"""Evaluation metrics: ranking quality (AUC) and probability quality (log loss, ECE, COPC)."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc(y, p):
    """Area under the ROC curve: chance that a clicked impression is scored above an unclicked one."""
    return float(roc_auc_score(y, p))


def logloss(y, p, eps=1e-7):
    """Cross-entropy; heavily punishes confident wrong probabilities."""
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def copc(y, p):
    """Clicks Over Predicted Clicks: 1.0 means the total predicted clicks match reality."""
    return float(np.sum(y) / np.sum(p))


def reliability_table(y, p, n_bins=20, strategy="quantile"):
    """Group predictions into bins and compare the mean prediction with the observed click rate.

    strategy='quantile' gives bins with equal numbers of impressions (best for skewed CTRs);
    strategy='uniform' gives equal-width probability bins.
    """
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    n_effective = len(edges) - 1
    bin_id = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_effective - 1)

    counts = np.bincount(bin_id, minlength=n_effective)
    nonempty = counts > 0
    mean_pred = np.bincount(bin_id, weights=p, minlength=n_effective)[nonempty] / counts[nonempty]
    mean_true = np.bincount(bin_id, weights=y, minlength=n_effective)[nonempty] / counts[nonempty]
    return pd.DataFrame({"mean_pred": mean_pred, "mean_true": mean_true, "count": counts[nonempty]})


def ece(y, p, n_bins=20, strategy="quantile"):
    """Expected calibration error: weighted average gap between predicted and observed rates."""
    table = reliability_table(y, p, n_bins, strategy)
    weights = table["count"] / table["count"].sum()
    return float(np.sum(weights * np.abs(table["mean_pred"] - table["mean_true"])))


def evaluate_predictions(y, p):
    """All headline metrics for one set of predictions."""
    return {"auc": auc(y, p), "logloss": logloss(y, p), "ece": ece(y, p), "copc": copc(y, p)}
