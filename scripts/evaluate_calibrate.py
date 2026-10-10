"""Compare all trained runs and measure what Platt scaling and isotonic regression do to them."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `ctr_pipeline` importable

import pandas as pd

from ctr_pipeline.calibration import CALIBRATORS
from ctr_pipeline.dataset import load_labels
from ctr_pipeline.metrics import evaluate_predictions, reliability_table
from ctr_pipeline.plotting import plot_metric_bars, plot_reliability
from ctr_pipeline.runs import list_runs, load_predictions, load_run_metrics


def markdown_table(frame):
    """Render a DataFrame as a GitHub-flavored markdown table (no extra dependencies)."""
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "|" + "|".join(["---"] * len(frame.columns)) + "|"
    rows = ["| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider] + rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", default="results")
    parser.add_argument("--n-bins", type=int, default=20, help="Bins of the reliability diagram and ECE")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    runs = list_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"No runs found in {args.runs_dir}. Run train_models.py first.")

    # Calibrators are fitted on the calibration split and judged on the (later) test split
    y_calib = load_labels(args.data, "calib")
    y_test = load_labels(args.data, "test")

    calibration_rows, comparison_rows = [], []
    for name in runs:
        p_calib, p_test = load_predictions(args.runs_dir, name, "calib"), load_predictions(args.runs_dir, name, "test")

        variants = {"raw": p_test}
        for label, cls in CALIBRATORS.items():
            variants[label] = cls().fit(p_calib, y_calib).transform(p_test)

        diagrams = {}
        for variant, p in variants.items():
            calibration_rows.append({"model": name, "calibration": variant, **evaluate_predictions(y_test, p)})
            diagrams[variant] = reliability_table(y_test, p, args.n_bins)
        plot_reliability(diagrams, f"{name}: reliability on the test split", out / f"reliability_{name}.png")

        # One row per model for the headline comparison table
        run_metrics = load_run_metrics(args.runs_dir, name)
        raw = evaluate_predictions(y_test, p_test)
        comparison_rows.append(
            {
                "model": name,
                "test_auc": raw["auc"],
                "test_logloss": raw["logloss"],
                "test_ece": raw["ece"],
                "train_seconds": run_metrics.get("train_seconds", float("nan")),
                "params": run_metrics.get("params_total", 0),
                "latency_ms_bs1": run_metrics.get("latency_ms_bs1", float("nan")),
                "us_per_row_bs1024": run_metrics.get("us_per_row_bs1024", float("nan")),
            }
        )

    calibration = pd.DataFrame(calibration_rows)
    comparison = pd.DataFrame(comparison_rows).sort_values("test_auc", ascending=False)
    calibration.to_csv(out / "calibration_results.csv", index=False)
    comparison.to_csv(out / "model_comparison.csv", index=False)
    (out / "model_comparison.md").write_text(markdown_table(comparison) + "\n")

    plot_metric_bars(comparison["model"], comparison["test_auc"], "test AUC", "Ranking quality", out / "comparison_auc.png")
    plot_metric_bars(comparison["model"], comparison["test_logloss"], "test log loss", "Probability quality", out / "comparison_logloss.png")

    print("\nModel comparison (raw predictions on the test split):")
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nEffect of calibration (fitted on calib split, evaluated on test):")
    print(calibration.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()
