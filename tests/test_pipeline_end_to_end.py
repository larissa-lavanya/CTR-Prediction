"""Generator format checks, plus an end-to-end Spark preprocessing test (slow, needs Java)."""

import shutil

import numpy as np
import pytest

from ctr_pipeline.dataset import load_meta, load_split
from ctr_pipeline.schema import N_DENSE, N_SPARSE, SPLITS
from ctr_pipeline.synthetic import write_criteo_like_tsv


@pytest.fixture(scope="module")
def raw_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("raw") / "tiny.tsv"
    stats = write_criteo_like_tsv(str(path), n_rows=20_000, seed=3, log=lambda s: None)
    return path, stats


def test_generator_writes_the_criteo_layout(raw_file):
    path, stats = raw_file
    lines = path.read_text().splitlines()
    assert len(lines) == 20_000

    fields = lines[0].split("\t")
    assert len(fields) == 1 + N_DENSE + N_SPARSE
    assert fields[0] in ("0", "1")

    # Click rate looks like the downsampled public data, and the oracle is a realistic ceiling
    assert 0.15 < stats["click_rate_kept_rows"] < 0.35
    assert 0.70 < stats["oracle_auc"] < 0.90


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("java") is None, reason="Spark needs Java")
def test_spark_pipeline_produces_valid_chronological_splits(raw_file, tmp_path):
    pytest.importorskip("pyspark")
    from ctr_pipeline.spark_preprocess import run_preprocessing

    path, _ = raw_file
    out = tmp_path / "processed"
    meta = run_preprocessing(str(path), str(out), min_count=3, shuffle_partitions=4, driver_memory="2g", log=lambda s: None)

    assert sum(info["rows"] for info in meta["splits"].values()) == 20_000
    previous_last_row = -1
    for split in SPLITS:
        data = load_split(out, split)
        assert data.n == meta["splits"][split]["rows"]

        # Indices stay inside each feature's vocabulary, and splits follow each other in time
        assert np.all(data.sparse >= 0)
        assert np.all(data.sparse.max(axis=0) < np.array(meta["vocab_sizes"]))
        assert data.row_id.min() > previous_last_row
        previous_last_row = data.row_id.max()

    # Dense features are standardized with TRAIN statistics
    train = load_split(out, "train")
    assert np.allclose(train.dense.mean(axis=0), 0.0, atol=0.05)
    assert np.allclose(train.dense.std(axis=0), 1.0, atol=0.05)
    assert load_meta(out)["timings_sec"]["build_vocabulary"] > 0
