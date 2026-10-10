"""Train the LightGBM baseline (target + frequency encoded categoricals) and save its predictions."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `ctr_pipeline` importable

import numpy as np

from ctr_pipeline.dataset import load_meta, load_split
from ctr_pipeline.gbdt import TargetEncoder, build_features, train_lightgbm
from ctr_pipeline.metrics import auc, logloss
from ctr_pipeline.runs import save_run
from ctr_pipeline.trainer import time_calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    meta = load_meta(args.data)
    train = load_split(args.data, "train", args.max_train_rows)
    sets = {split: load_split(args.data, split) for split in ("val", "calib", "test")}

    # Encode categoricals (out-of-fold for training rows) and build the feature matrices
    started = time.perf_counter()
    encoder = TargetEncoder(meta["vocab_sizes"], seed=args.seed)
    x_train = build_features(encoder, train, encoder.fit_transform(train.sparse, train.label.astype(np.float64)))
    features = {split: build_features(encoder, data, encoder.transform(data.sparse)) for split, data in sets.items()}

    # Gradient boosting with early stopping on validation log loss
    booster = train_lightgbm(x_train, train.label, features["val"], sets["val"].label,
                             learning_rate=args.learning_rate, num_leaves=args.num_leaves, seed=args.seed)
    train_seconds = time.perf_counter() - started

    predictions = {split: booster.predict(x, num_iteration=booster.best_iteration) for split, x in features.items()}
    y_test = sets["test"].label
    metrics = {
        "model": "lightgbm",
        "train_rows": int(train.n),
        "train_seconds": train_seconds,
        "best_step": int(booster.best_iteration),
        "steps": int(booster.current_iteration()),
        "val_logloss_best": float(booster.best_score["valid_0"]["binary_logloss"]),
        "params_total": int(booster.num_trees() * args.num_leaves),  # leaf values (upper bound)
        "params_embedding": 0,
        "params_dense": int(booster.num_trees() * args.num_leaves),
        "test_auc": auc(y_test, predictions["test"]),
        "test_logloss": logloss(y_test, predictions["test"]),
    }
    for size in (1, 1024):
        batch = features["test"][:size]
        ms = time_calls(lambda: booster.predict(batch, num_iteration=booster.best_iteration), n_iters=30)
        metrics[f"latency_ms_bs{size}"] = ms
        metrics[f"us_per_row_bs{size}"] = 1000.0 * ms / size

    config = {"model": "lightgbm", "hyperparams": {"num_leaves": args.num_leaves, "learning_rate": args.learning_rate}, "seed": args.seed}
    save_run(args.runs_dir, "lightgbm", metrics, config, predictions)
    print(f"lightgbm: test AUC {metrics['test_auc']:.4f}  log loss {metrics['test_logloss']:.4f}  trees {booster.best_iteration}  train {train_seconds:.0f}s")


if __name__ == "__main__":
    main()
