"""Column names and constants shared across the project (Criteo Display Advertising layout)."""

N_DENSE = 13  # integer count features I1..I13
N_SPARSE = 26  # hashed categorical features C1..C26

# Raw TSV layout: label, 13 dense, 26 sparse (no header, tab separated)
RAW_DENSE_COLS = [f"I{i}" for i in range(1, N_DENSE + 1)]
RAW_SPARSE_COLS = [f"C{i}" for i in range(1, N_SPARSE + 1)]
RAW_COLUMNS = ["label"] + RAW_DENSE_COLS + RAW_SPARSE_COLS

# Column names in the processed Parquet files
DENSE_COLS = [f"dense_{i}" for i in range(N_DENSE)]
SPARSE_COLS = [f"sparse_{i}" for i in range(N_SPARSE)]

# Chronological splits: train < val (early stopping) < calib (calibration) < test
SPLITS = ("train", "val", "calib", "test")

# Reserved embedding indices written by the preprocessing
MISSING_INDEX = 0  # the value was empty in the raw data
RARE_INDEX = 1  # the value is rare (below min_count) or was not in the training vocabulary
FIRST_VOCAB_INDEX = 2  # frequent values get indices 2, 3, ... in order of decreasing frequency
