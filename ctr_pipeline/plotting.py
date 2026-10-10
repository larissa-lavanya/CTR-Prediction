"""Matplotlib figures (saved to files, no display needed)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_reliability(tables, title, path):
    """Reliability diagram: points on the diagonal mean 'predicted probability = observed click rate'.

    `tables` maps a label (e.g. 'raw', 'platt') to a reliability_table DataFrame.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    top = 0.0
    for label, table in tables.items():
        ax.plot(table["mean_pred"], table["mean_true"], marker="o", markersize=4, label=label)
        top = max(top, table["mean_pred"].max(), table["mean_true"].max())

    ax.plot([0, top], [0, top], color="gray", linestyle="--", label="perfect calibration")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed click rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric_bars(labels, values, ylabel, title, path, errors=None, zero_line=False):
    """Simple bar chart (one bar per model or variant)."""
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels)), 4.5))
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, capsize=3, color="tab:blue", alpha=0.85)
    if zero_line:
        ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ablation(frame, x_col, title, path):
    """AUC (left axis) and log loss (right axis) as one setting is varied."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(frame[x_col].astype(str), frame["test_auc"], marker="o", color="tab:blue", label="test AUC")
    ax.set_xlabel(x_col)
    ax.set_ylabel("test AUC", color="tab:blue")

    twin = ax.twinx()
    twin.plot(frame[x_col].astype(str), frame["test_logloss"], marker="s", color="tab:red", label="test log loss")
    twin.set_ylabel("test log loss", color="tab:red")

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(frame, path, top=20):
    """Horizontal bars of the AUC drop when a feature is shuffled."""
    top_frame = frame.sort_values("auc_drop", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(top_frame) + 1.5))
    ax.barh(top_frame["feature"], top_frame["auc_drop"], color="tab:blue", alpha=0.85)
    ax.set_xlabel("drop in test AUC when the feature is shuffled")
    ax.set_title("Permutation importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
