"""One reusable 'train a model and measure everything' function used by scripts and ablations."""

import numpy as np

from ctr_pipeline.metrics import auc, logloss
from ctr_pipeline.models import build_model, count_parameters
from ctr_pipeline.schema import N_DENSE
from ctr_pipeline.trainer import fit, measure_model_latency, predict_proba

DEFAULT_HYPERPARAMS = {"emb_dim": 16, "hidden": (256, 128), "dropout": 0.1, "n_cross": 3}
DEFAULT_TRAINING = {
    "epochs": 2, "batch_size": 2048, "lr": 1e-3, "emb_lr": 2e-3, "emb_l2": 0.0, "evals_per_epoch": 4, "patience": 3,
}


def train_and_evaluate(name, train, val, evaluation_sets, vocab_sizes, hyperparams=None, training=None,
                       device="cpu", seed=0, log=print, measure_latency=True):
    """Train one model; return (model, metrics, predictions on every set in `evaluation_sets`)."""
    hp = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    cfg = {**DEFAULT_TRAINING, **(training or {})}

    model = build_model(name, vocab_sizes, N_DENSE, **hp)
    result = fit(model, train, val, device=device, seed=seed, log=log, **cfg)

    # Predictions (on the training scale) for every evaluation split
    predictions = {split: predict_proba(model, data, device=device) for split, data in evaluation_sets.items()}

    metrics = {
        "model": name,
        "train_rows": int(train.n),
        "train_seconds": result.train_seconds,
        "best_step": result.best_step,
        "steps": result.steps,
        "val_logloss_best": result.best_val_logloss,
        **count_parameters(model),
    }
    if "test" in evaluation_sets:
        metrics["test_auc"] = auc(evaluation_sets["test"].label, predictions["test"])
        metrics["test_logloss"] = logloss(evaluation_sets["test"].label, predictions["test"])
    if measure_latency:
        metrics.update(measure_model_latency(model, evaluation_sets.get("test", val)))

    config = {"model": name, "hyperparams": {k: list(v) if isinstance(v, tuple) else v for k, v in hp.items()},
              "training": cfg, "seed": seed}
    return model, metrics, predictions, config
