"""Generate a Criteo-format TSV file from a synthetic click model (for testing and demos)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `ctr_pipeline` importable

from ctr_pipeline.synthetic import write_criteo_like_tsv
from ctr_pipeline.utils import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=1_000_000, help="Rows to write (after negative downsampling)")
    parser.add_argument("--output", default="data/criteo_like.tsv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-ctr", type=float, default=0.03, help="True average click-through rate")
    parser.add_argument("--neg-keep-rate", type=float, default=0.1, help="Fraction of negatives that are kept")
    parser.add_argument("--drift", type=float, default=0.3, help="Change of the click logit over the whole period")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating {args.n_rows:,} rows -> {args.output}")
    stats = write_criteo_like_tsv(
        args.output, args.n_rows, args.seed, args.target_ctr, args.neg_keep_rate, args.drift
    )

    # The oracle numbers are the best any model could do: use them to judge your results
    save_json(stats, Path(args.output).with_suffix(".oracle.json"))
    print(
        f"\nclick rate in the file: {stats['click_rate_kept_rows']:.3f} "
        f"(true CTR is about {stats['true_ctr_estimate']:.3f}, negatives kept with prob {args.neg_keep_rate})"
        f"\noracle AUC: {stats['oracle_auc']:.4f}   oracle log loss: {stats['oracle_logloss']:.4f}"
    )


if __name__ == "__main__":
    main()
