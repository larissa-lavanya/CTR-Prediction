"""Bidding simulation: how much money do calibration and downsampling correction make (or lose)?"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `ctr_pipeline` importable

import numpy as np
import pandas as pd

from ctr_pipeline.auction_sim import simulate_second_price
from ctr_pipeline.calibration import CALIBRATORS
from ctr_pipeline.dataset import load_labels
from ctr_pipeline.downsampling import correct_downsampled_probability
from ctr_pipeline.plotting import plot_metric_bars
from ctr_pipeline.runs import list_runs, load_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", default="results")
    parser.add_argument("--neg-keep-rate", type=float, default=1.0,
                        help="Fraction of negatives kept when the data was downsampled (1.0 = not downsampled)")
    parser.add_argument("--value-per-click", type=float, default=1.0)
    parser.add_argument("--competitor-factor", type=float, default=1.0,
                        help="Median competing price = factor * value * average true CTR")
    parser.add_argument("--competitor-sigma", type=float, default=1.0, help="Spread (lognormal sigma) of competing prices")
    parser.add_argument("--n-sims", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    keep = args.neg_keep_rate

    # Labels, and the weights that undo the negative downsampling (each kept negative stands for 1/keep)
    y_calib, y_test = load_labels(args.data, "calib"), load_labels(args.data, "test")
    weight = np.where(y_test == 1, 1.0, 1.0 / keep)

    # Average TRUE click rate (from the calibration split) sets the scale of competing prices
    base_ctr = float(correct_downsampled_probability(y_calib.mean(), keep))
    competitor_median = args.competitor_factor * args.value_per_click * base_ctr
    print(f"average true CTR ~ {base_ctr:.4f}; median competing price {competitor_median:.4f} (value per click {args.value_per_click})")

    def simulate(p_true_scale):
        return simulate_second_price(
            p_true_scale, y_test, weight, args.value_per_click, competitor_median, args.competitor_sigma, args.n_sims, args.seed
        )

    rows = []

    # Reference: bid the same average CTR on every impression (no model at all)
    result = simulate(np.full(len(y_test), base_ctr))
    rows.append({"model": "no_model", "variant": "constant_bid", **result["mean"].to_dict(), "surplus_std": result.loc["surplus_per_1k", "std"]})

    for name in list_runs(args.runs_dir):
        p_calib, p_test = load_predictions(args.runs_dir, name, "calib"), load_predictions(args.runs_dir, name, "test")

        # Four ways of turning model output into a bid probability
        variants = {"raw (no correction)": p_test, "downsampling corrected": correct_downsampled_probability(p_test, keep)}
        for label, cls in CALIBRATORS.items():
            calibrated = cls().fit(p_calib, y_calib).transform(p_test)  # calibrate on the training scale first
            variants[f"{label} + corrected"] = correct_downsampled_probability(calibrated, keep)

        for variant, p_bid in variants.items():
            result = simulate(np.clip(p_bid, 0.0, 1.0))
            rows.append({"model": name, "variant": variant, **result["mean"].to_dict(), "surplus_std": result.loc["surplus_per_1k", "std"]})

    table = pd.DataFrame(rows)
    table.to_csv(out / "auction_simulation.csv", index=False)
    print("\nSimulated second-price auctions (per 1000 real impressions; surplus = value won - price paid):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    labels = [f"{m}\n{v}" for m, v in zip(table["model"], table["variant"])]
    plot_metric_bars(labels, table["surplus_per_1k"], "surplus per 1000 impressions", "Auction surplus by bidding probability",
                     out / "auction_surplus.png", errors=table["surplus_std"], zero_line=True)
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()
