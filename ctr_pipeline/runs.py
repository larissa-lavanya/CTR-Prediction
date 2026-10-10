"""Saving and loading experiment runs (predictions, metrics, model weights) from disk."""

from pathlib import Path

import numpy as np
import torch

from ctr_pipeline.utils import load_json, save_json


def run_dir(runs_dir, name):
    return Path(runs_dir) / name


def save_run(runs_dir, name, metrics, config, predictions, state_dict=None):
    """Write metrics.json, config.json, preds_<split>.npy and (optionally) model.pt."""
    directory = run_dir(runs_dir, name)
    directory.mkdir(parents=True, exist_ok=True)
    save_json(metrics, directory / "metrics.json")
    save_json(config, directory / "config.json")
    for split, preds in predictions.items():
        np.save(directory / f"preds_{split}.npy", preds.astype(np.float32))
    if state_dict is not None:
        torch.save(state_dict, directory / "model.pt")


def list_runs(runs_dir):
    """Names of all runs that have a metrics.json, sorted alphabetically."""
    root = Path(runs_dir)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "metrics.json").exists())


def load_run_metrics(runs_dir, name):
    return load_json(run_dir(runs_dir, name) / "metrics.json")


def load_run_config(runs_dir, name):
    return load_json(run_dir(runs_dir, name) / "config.json")


def load_predictions(runs_dir, name, split):
    return np.load(run_dir(runs_dir, name) / f"preds_{split}.npy").astype(np.float64)
