"""Run the PySpark preprocessing pipeline on a Criteo-format TSV file."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `ctr_pipeline` importable

from ctr_pipeline.spark_preprocess import run_preprocessing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw TSV (label, 13 ints, 26 hex categoricals)")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--min-count", type=int, default=10, help="Values seen fewer times in train are 'rare'")
    parser.add_argument("--max-vocab", type=int, default=200_000, help="Keep at most this many values per feature")
    parser.add_argument("--ratios", nargs=4, type=float, default=[0.8, 0.05, 0.05, 0.1],
                        help="Chronological split: train val calib test")
    parser.add_argument("--master", default="local[*]", help="Spark master URL")
    parser.add_argument("--driver-memory", default="4g")
    parser.add_argument("--shuffle-partitions", type=int, default=16)
    args = parser.parse_args()

    meta = run_preprocessing(
        args.input,
        args.output,
        min_count=args.min_count,
        max_vocab=args.max_vocab,
        ratios=tuple(args.ratios),
        master=args.master,
        driver_memory=args.driver_memory,
        shuffle_partitions=args.shuffle_partitions,
    )
    print(f"\nProcessed {meta['n_rows']:,} rows -> {args.output}")
    for split, info in meta["splits"].items():
        print(f"  {split:6s} {info['rows']:>12,} rows   click rate {info['ctr']:.4f}")


if __name__ == "__main__":
    main()
