#!/usr/bin/env bash
# End-to-end demo on synthetic Criteo-format data. A few minutes with a GPU, 10-20 minutes on 2 CPU cores (500k rows).
# Usage: bash run_demo.sh            (or: N_ROWS=2000000 bash run_demo.sh)
set -euo pipefail

# Ignore packages in ~/.local (they can be built for another Python and break imports)
export PYTHONNOUSERSITE=1
unset PYTHONPATH   # a PYTHONPATH pointing to another Python's packages also breaks imports

N_ROWS=${N_ROWS:-500000}
NEG_KEEP=0.1   # must match the generator's --neg-keep-rate (default 0.1)

# 1) Synthetic data in the Criteo layout (clicks kept, 90% of non-clicks dropped)
python3 scripts/generate_synthetic_data.py --n-rows "$N_ROWS" --neg-keep-rate "$NEG_KEEP" --output data/criteo_like.tsv

# 2) PySpark preprocessing: vocabularies, rare bucket, chronological split, Parquet
python3 scripts/preprocess_spark.py --input data/criteo_like.tsv --output data/processed --min-count 5

# 3) Train the models and the gradient-boosting baseline
python3 scripts/train_models.py --data data/processed
python3 scripts/train_gbdt.py --data data/processed

# 4) Compare models, then calibrate (Platt / isotonic) and draw reliability diagrams
python3 scripts/evaluate_calibrate.py --data data/processed

# 5) Simulate second-price auctions with different bidding probabilities
python3 scripts/simulate_auction.py --data data/processed --neg-keep-rate "$NEG_KEEP"

# 6) Ablations: embedding size, rare threshold, cross layers, permutation importance
python3 scripts/run_ablations.py --data data/processed
