"""PySpark preprocessing: raw Criteo-style TSV -> indexed, chronologically split Parquet files.

Pipeline
    1. read the TSV with an explicit schema and tag every row with its position in the file
    2. split by position into train / val / calib / test (rows are in time order)
    3. build a vocabulary per categorical feature from the TRAIN split only (no leakage):
       frequent values get indices 2, 3, ...; rare values share index 1; missing values are index 0
    4. replace categorical values by indices (broadcast joins) and standardize log-scaled counts
    5. write Parquet files and a meta.json with sizes, statistics and timings
"""

from pathlib import Path

import numpy as np
import pyarrow.dataset as pa_ds
from pyspark import StorageLevel
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from ctr_pipeline.schema import (
    FIRST_VOCAB_INDEX,
    MISSING_INDEX,
    N_DENSE,
    N_SPARSE,
    RARE_INDEX,
    RAW_DENSE_COLS,
    RAW_SPARSE_COLS,
    SPLITS,
)
from ctr_pipeline.utils import save_json, timer


def create_spark(master="local[*]", driver_memory="4g", shuffle_partitions=16):
    """Start a local Spark session (use a cluster master URL to scale out)."""
    return (
        SparkSession.builder.master(master)
        .appName("ctr-preprocess")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def raw_schema():
    """label, 13 integer features, 26 string features."""
    fields = [T.StructField("label", T.IntegerType())]
    fields += [T.StructField(c, T.LongType()) for c in RAW_DENSE_COLS]
    fields += [T.StructField(c, T.StringType()) for c in RAW_SPARSE_COLS]
    return T.StructType(fields)


def signed_log(column):
    """sign(x) * log(1 + |x|): compresses heavy-tailed counts and keeps the sign of negatives."""
    x = F.col(column).cast("double")
    return F.signum(x) * F.log1p(F.abs(x))


def filled_signed_log(column):
    """Signed log with missing values replaced by 0."""
    return F.coalesce(signed_log(column), F.lit(0.0))


def assign_splits(df, ratios):
    """Tag each row with train/val/calib/test by its position in the file (chronological)."""
    cumulative = np.cumsum(ratios)[:-1] / np.sum(ratios)

    # Cut points are approximate quantiles of the row position (error 0.01% of the rows)
    cuts = df.approxQuantile("row_id", [float(c) for c in cumulative], 0.0001)

    split = F.lit(SPLITS[-1])
    for name, cut in reversed(list(zip(SPLITS[:-1], cuts))):
        split = F.when(F.col("row_id") <= cut, F.lit(name)).otherwise(split)
    return df.withColumn("split", split)


def build_vocabulary(train_df, min_count, max_vocab):
    """Return a DataFrame (field, value, index, count) of the frequent values of each feature."""
    # Melt the 26 columns into (field, value) pairs so ONE pass counts every feature
    pairs = ", ".join(f"'{c}', {c}" for c in RAW_SPARSE_COLS)
    long_df = train_df.select(F.expr(f"stack({N_SPARSE}, {pairs}) as (field, value)")).where(F.col("value").isNotNull())

    counts = long_df.groupBy("field", "value").count().where(F.col("count") >= min_count)

    # Rank values inside each feature by frequency (ties broken by value for determinism)
    ranking = Window.partitionBy("field").orderBy(F.desc("count"), F.asc("value"))
    vocab = (
        counts.withColumn("rank", F.row_number().over(ranking))
        .where(F.col("rank") <= max_vocab)
        .withColumn("index", F.col("rank") + (FIRST_VOCAB_INDEX - 1))
        .select("field", "value", "index", "count")
    )
    return vocab


def apply_vocabulary(df, vocab):
    """Replace every categorical column by an integer index column sparse_i."""
    for i, column in enumerate(RAW_SPARSE_COLS):
        # Small per-feature lookup table, broadcast to every executor (no shuffle of the big table)
        lookup = vocab.where(F.col("field") == column).select(F.col("value").alias(column), F.col("index").alias("_idx"))
        df = df.join(F.broadcast(lookup), on=column, how="left")
        df = df.withColumn(
            f"sparse_{i}",
            F.when(F.col(column).isNull(), F.lit(MISSING_INDEX))
            .when(F.col("_idx").isNull(), F.lit(RARE_INDEX))
            .otherwise(F.col("_idx"))
            .cast("int"),
        ).drop("_idx")
    return df


def run_preprocessing(
    input_path,
    output_dir,
    min_count=10,
    max_vocab=200_000,
    ratios=(0.8, 0.05, 0.05, 0.1),
    master="local[*]",
    driver_memory="4g",
    shuffle_partitions=16,
    log=print,
):
    """Run the whole pipeline and return the metadata dictionary (also saved as meta.json)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings = {}

    spark = create_spark(master, driver_memory, shuffle_partitions)
    spark.sparkContext.setLogLevel("ERROR")

    # 1) Read the raw file and remember each row's position in the file
    with timer(timings, "read_and_count_rows"):
        df = spark.read.csv(str(input_path), sep="\t", schema=raw_schema(), nullValue="")
        df = df.withColumn("row_id", F.monotonically_increasing_id())
        df = assign_splits(df, ratios)
        df = df.persist(StorageLevel.MEMORY_AND_DISK)
        split_stats = {r["split"]: (r["n"], r["ctr"]) for r in df.groupBy("split").agg(F.count("*").alias("n"), F.mean("label").alias("ctr")).collect()}
    n_rows = sum(n for n, _ in split_stats.values())
    log(f"read {n_rows:,} rows: " + ", ".join(f"{s}={split_stats[s][0]:,}" for s in SPLITS if s in split_stats))

    train_df = df.where(F.col("split") == "train")

    # 2) Vocabulary from the training split only
    with timer(timings, "build_vocabulary"):
        vocab = build_vocabulary(train_df, min_count, max_vocab)
        vocab.write.mode("overwrite").parquet(str(out / "vocab"))
        vocab = spark.read.parquet(str(out / "vocab")).persist(StorageLevel.MEMORY_AND_DISK)
        max_index = {r["field"]: r["max_index"] for r in vocab.groupBy("field").agg(F.max("index").alias("max_index")).collect()}
    vocab_sizes = [int(max_index.get(c, RARE_INDEX)) + 1 for c in RAW_SPARSE_COLS]
    log(f"vocabulary sizes (incl. missing and rare): min={min(vocab_sizes)}, max={max(vocab_sizes):,}, total={sum(vocab_sizes):,}")

    # 3) Standardization statistics of the dense features (training split only)
    with timer(timings, "dense_statistics"):
        exprs = []
        for i, column in enumerate(RAW_DENSE_COLS):
            exprs += [F.mean(filled_signed_log(column)).alias(f"mean_{i}"), F.stddev(filled_signed_log(column)).alias(f"std_{i}")]
        stats = train_df.agg(*exprs).collect()[0]
    dense_mean = [float(stats[f"mean_{i}"]) for i in range(N_DENSE)]
    dense_std = [float(stats[f"std_{i}"]) or 1.0 for i in range(N_DENSE)]

    # 4) Transform: dense -> standardized signed log; sparse -> vocabulary indices
    with timer(timings, "transform_and_write"):
        result = apply_vocabulary(df, vocab)
        for i, column in enumerate(RAW_DENSE_COLS):
            result = result.withColumn(f"dense_{i}", ((filled_signed_log(column) - dense_mean[i]) / dense_std[i]).cast("float"))
        keep = ["row_id", "split", "label"] + [f"dense_{i}" for i in range(N_DENSE)] + [f"sparse_{i}" for i in range(N_SPARSE)]
        result.select(*keep).write.mode("overwrite").partitionBy("split").parquet(str(out / "data"))

    spark_version = spark.version
    spark.stop()

    # 5) Per-feature value counts (used by the min-count ablation) and the metadata file
    vocab_counts = _vocab_counts_from_parquet(out / "vocab", vocab_sizes)
    np.savez_compressed(out / "vocab_counts.npz", **{f"sparse_{i}": c for i, c in enumerate(vocab_counts)})

    meta = {
        "input": str(input_path),
        "n_rows": int(n_rows),
        "splits": {s: {"rows": int(split_stats[s][0]), "ctr": float(split_stats[s][1])} for s in SPLITS if s in split_stats},
        "vocab_sizes": vocab_sizes,
        "min_count": min_count,
        "max_vocab": max_vocab,
        "ratios": list(ratios),
        "dense_mean": dense_mean,
        "dense_std": dense_std,
        "timings_sec": timings,
        "spark_version": spark_version,
        "master": master,
    }
    save_json(meta, out / "meta.json")
    log(f"timings (seconds): {timings}")
    return meta


def _vocab_counts_from_parquet(vocab_dir, vocab_sizes):
    """counts[i][index] = how often that value appears in the training split (0 for reserved indices)."""
    table = pa_ds.dataset(str(vocab_dir), format="parquet").to_table(columns=["field", "index", "count"])
    field = np.asarray(table.column("field").to_pylist())
    index = table.column("index").to_numpy()
    count = table.column("count").to_numpy()

    counts = []
    for column, size in zip(RAW_SPARSE_COLS, vocab_sizes):
        array = np.zeros(size, dtype=np.int64)
        mask = field == column
        array[index[mask]] = count[mask]
        counts.append(array)
    return counts
