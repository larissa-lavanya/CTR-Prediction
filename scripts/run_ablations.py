"""Ablations: embedding size, rare-value threshold, number of cross layers, permutation importance."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `ctr_pipeline` importable

import numpy as np
import pandas as pd
import torch

from ctr_pipeline.dataset import apply_min_count, load_meta, load_split
from ctr_pipeline.experiment import train_and_evaluate
from ctr_pipeline.metrics import auc
from ctr_pipeline.models import build_model
from ctr_pipeline.plotting import plot_ablation, plot_importance
from ctr_pipeline.runs import load_run_config
from ctr_pipeline.schema import N_DENSE, N_SPARSE, RAW_DENSE_COLS, RAW_SPARSE_COLS
from ctr_pipeline.trainer import predict_proba
from ctr_pipeline.utils import get_device, set_seed


def run_setting(model_name, hyperparams, train, val, test, meta, training, device, seed):
    """Train one configuration and return its headline numbers."""
    _, metrics, _, _ = train_and_evaluate(
        model_name, train, val, {"test": test}, meta["vocab_sizes"], hyperparams, training, device, seed,
        log=lambda s: None, measure_latency=False,
    )
    return {k: metrics[k] for k in ("test_auc", "test_logloss", "params_total", "train_seconds")}


def permutation_importance(model, test, device, n_repeats, seed):
    """AUC drop when one feature column is shuffled across rows (breaks its link to the label)."""
    rng = np.random.default_rng(seed)
    baseline = auc(test.label, predict_proba(model, test, device=device))

    rows = []
    names = list(RAW_DENSE_COLS) + list(RAW_SPARSE_COLS)
    for feature, name in enumerate(names):
        drops = []
        for _ in range(n_repeats):
            sparse, dense = test.sparse.copy(), test.dense.copy()
            if feature < N_DENSE:
                dense[:, feature] = rng.permutation(dense[:, feature])
            else:
                sparse[:, feature - N_DENSE] = rng.permutation(sparse[:, feature - N_DENSE])
            shuffled = type(test)(sparse, dense, test.label, test.row_id)
            drops.append(baseline - auc(test.label, predict_proba(model, shuffled, device=device)))
        rows.append({"feature": name, "auc_drop": float(np.mean(drops))})
    return baseline, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", default="results/ablations")
    parser.add_argument("--ablations", nargs="+", default=["emb_dim", "min_count", "n_cross", "importance"],
                        choices=["emb_dim", "min_count", "n_cross", "importance"])
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--importance-run", default="dcnv2", help="Which trained run to compute importance for")
    parser.add_argument("--importance-rows", type=int, default=50_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = get_device(args.device)

    meta = load_meta(args.data)
    train = load_split(args.data, "train", args.max_train_rows)
    val, test = load_split(args.data, "val"), load_split(args.data, "test")
    training = {"epochs": args.epochs}
    vocab_counts = np.load(Path(args.data) / "vocab_counts.npz")

    if "emb_dim" in args.ablations:
        print("\n== embedding dimension (DeepFM) ==")
        rows = []
        for dim in (4, 8, 16, 32):
            rows.append({"emb_dim": dim, **run_setting("deepfm", {"emb_dim": dim}, train, val, test, meta, training, device, args.seed)})
            print(rows[-1])
        frame = pd.DataFrame(rows)
        frame.to_csv(out / "ablation_emb_dim.csv", index=False)
        plot_ablation(frame, "emb_dim", "DeepFM: embedding dimension", out / "ablation_emb_dim.png")

    if "min_count" in args.ablations:
        print("\n== rare-value threshold (DeepFM) ==")
        rows = []
        thresholds = sorted({meta["min_count"], 20, 50, 200, 1000})
        for threshold in thresholds:
            # Values seen fewer than `threshold` times in training share one 'rare' embedding
            remapped = [d.with_sparse(apply_min_count(d.sparse, vocab_counts, threshold)) for d in (train, val, test)]
            rows.append({"min_count": threshold, "rare_share_train": float((remapped[0].sparse == 1).mean()),
                         **run_setting("deepfm", {}, *remapped, meta, training, device, args.seed)})
            print(rows[-1])
        frame = pd.DataFrame(rows)
        frame.to_csv(out / "ablation_min_count.csv", index=False)
        plot_ablation(frame, "min_count", "DeepFM: rare-value threshold", out / "ablation_min_count.png")

    if "n_cross" in args.ablations:
        print("\n== number of cross layers (DCN-v2) ==")
        rows = []
        for n_cross in (0, 1, 2, 3, 4):
            rows.append({"n_cross": n_cross, **run_setting("dcnv2", {"n_cross": n_cross}, train, val, test, meta, training, device, args.seed)})
            print(rows[-1])
        frame = pd.DataFrame(rows)
        frame.to_csv(out / "ablation_n_cross.csv", index=False)
        plot_ablation(frame, "n_cross", "DCN-v2: number of cross layers", out / "ablation_n_cross.png")

    if "importance" in args.ablations:
        print(f"\n== permutation importance ({args.importance_run}) ==")
        config = load_run_config(args.runs_dir, args.importance_run)
        hp = {k: tuple(v) if isinstance(v, list) else v for k, v in config["hyperparams"].items()}
        model = build_model(config["model"], meta["vocab_sizes"], N_DENSE, **hp)
        model.load_state_dict(torch.load(Path(args.runs_dir) / args.importance_run / "model.pt", map_location="cpu"))
        model.to(device)

        baseline, frame = permutation_importance(model, test.head(args.importance_rows), device, n_repeats=3, seed=args.seed)
        frame = frame.sort_values("auc_drop", ascending=False)
        frame.to_csv(out / "permutation_importance.csv", index=False)
        plot_importance(frame, out / "permutation_importance.png")
        print(f"baseline AUC on {args.importance_rows:,} test rows: {baseline:.4f}")
        print(frame.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()
