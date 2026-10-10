"""Train Logistic Regression, FM, DeepFM and DCN-v2 and save predictions, metrics and weights."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `ctr_pipeline` importable

from ctr_pipeline.dataset import load_meta, load_split
from ctr_pipeline.experiment import train_and_evaluate
from ctr_pipeline.runs import save_run
from ctr_pipeline.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed", help="Output folder of preprocess_spark.py")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--models", nargs="+", default=["lr", "fm", "deepfm", "dcnv2"], choices=["lr", "fm", "deepfm", "dcnv2"])
    parser.add_argument("--max-train-rows", type=int, default=None, help="Use only the earliest N training rows")
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--hidden", nargs="+", type=int, default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-cross", type=int, default=3, help="Cross layers in DCN-v2")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of the dense weights")
    parser.add_argument("--emb-lr", type=float, default=2e-3, help="Learning rate of the embedding tables")
    parser.add_argument("--emb-l2", type=float, default=0.0, help="Strength of the L2 penalty on used embeddings")
    parser.add_argument("--tag", default="", help="Suffix for the run name, e.g. _dim32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    # Load the splits (only the training split can be capped)
    meta = load_meta(args.data)
    print(f"Loading data from {args.data} (device: {device})")
    train = load_split(args.data, "train", args.max_train_rows)
    sets = {split: load_split(args.data, split) for split in ("val", "calib", "test")}
    print(f"train {train.n:,} rows | val {sets['val'].n:,} | calib {sets['calib'].n:,} | test {sets['test'].n:,}")

    hyperparams = {"emb_dim": args.emb_dim, "hidden": tuple(args.hidden), "dropout": args.dropout, "n_cross": args.n_cross}
    training = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "emb_lr": args.emb_lr, "emb_l2": args.emb_l2}

    for name in args.models:
        print(f"\n=== {name} ===")
        model, metrics, predictions, config = train_and_evaluate(
            name, train, sets["val"], sets, meta["vocab_sizes"], hyperparams, training, device, args.seed
        )
        save_run(args.runs_dir, name + args.tag, metrics, config, predictions, model.state_dict())
        print(
            f"{name}: test AUC {metrics['test_auc']:.4f}  log loss {metrics['test_logloss']:.4f}  "
            f"params {metrics['params_total']:,}  train {metrics['train_seconds']:.0f}s"
        )


if __name__ == "__main__":
    main()
