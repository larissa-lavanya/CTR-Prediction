"""Load the processed Parquet splits into numpy arrays for PyTorch training."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.dataset as pa_ds

from ctr_pipeline.schema import DENSE_COLS, FIRST_VOCAB_INDEX, RARE_INDEX, SPARSE_COLS
from ctr_pipeline.utils import load_json


@dataclass
class CTRData:
    """One split held in memory: about 150 bytes per row."""

    sparse: np.ndarray  # (n, 26) int32 vocabulary indices
    dense: np.ndarray  # (n, 13) float32 standardized counts
    label: np.ndarray  # (n,) float32 click labels
    row_id: np.ndarray  # (n,) int64 position in the original file (increasing = later in time)

    @property
    def n(self):
        return len(self.label)

    def head(self, n):
        """First n rows (the earliest in time)."""
        return CTRData(self.sparse[:n], self.dense[:n], self.label[:n], self.row_id[:n])

    def with_sparse(self, sparse):
        return CTRData(sparse, self.dense, self.label, self.row_id)


def load_meta(processed_dir):
    return load_json(Path(processed_dir) / "meta.json")


def _split_dataset(processed_dir, split):
    return pa_ds.dataset(str(Path(processed_dir) / "data" / f"split={split}"), format="parquet")


def load_split(processed_dir, split, max_rows=None):
    """Read one split, sorted by time. `max_rows` keeps only the earliest rows of the split."""
    dataset = _split_dataset(processed_dir, split)

    # With a row cap, first find the row_id cutoff so only the needed rows are read in full
    row_filter = None
    if max_rows is not None:
        row_ids = dataset.to_table(columns=["row_id"]).column("row_id").to_numpy()
        if max_rows < len(row_ids):
            cutoff = np.partition(row_ids, max_rows - 1)[max_rows - 1]
            row_filter = pa_ds.field("row_id") <= int(cutoff)

    table = dataset.to_table(columns=["row_id", "label"] + DENSE_COLS + SPARSE_COLS, filter=row_filter)
    order = np.argsort(table.column("row_id").to_numpy())

    sparse = np.stack([table.column(c).to_numpy() for c in SPARSE_COLS], axis=1).astype(np.int32)[order]
    dense = np.stack([table.column(c).to_numpy() for c in DENSE_COLS], axis=1).astype(np.float32)[order]
    label = table.column("label").to_numpy().astype(np.float32)[order]
    row_id = table.column("row_id").to_numpy()[order]
    return CTRData(sparse, dense, label, row_id)


def load_labels(processed_dir, split, max_rows=None):
    """Only the labels of a split, in time order (cheap: reads two columns)."""
    dataset = _split_dataset(processed_dir, split)
    table = dataset.to_table(columns=["row_id", "label"])
    order = np.argsort(table.column("row_id").to_numpy())
    labels = table.column("label").to_numpy().astype(np.float32)[order]
    return labels if max_rows is None else labels[:max_rows]


def apply_min_count(sparse, vocab_counts, threshold):
    """Send values seen fewer than `threshold` times in training to the shared 'rare' index.

    The Spark job keeps every value above its (low) min_count, so a stricter threshold can be
    tried without re-running Spark. Indices 0 (missing) and 1 (rare) are left unchanged.
    """
    remapped = sparse.copy()
    for i in range(sparse.shape[1]):
        counts = vocab_counts[f"sparse_{i}"]
        column = sparse[:, i]
        too_rare = (column >= FIRST_VOCAB_INDEX) & (counts[column] < threshold)
        remapped[too_rare, i] = RARE_INDEX
    return remapped
