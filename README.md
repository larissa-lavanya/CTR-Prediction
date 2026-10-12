# CTR Prediction at Scale: PySpark Pipeline, Deep CTR Models, Calibration and Auction Simulation

An end-to-end click-through-rate (CTR) prediction project in the Criteo display-advertising format. It covers the whole path an ad-tech ML team cares about:

1. **PySpark preprocessing** of a 40-column hashed-feature log (vocabularies, rare-value bucket, chronological split, Parquet).
2. **Models**: Logistic Regression, Factorization Machine, DeepFM, DCN-v2 (PyTorch) and a LightGBM baseline with target encoding.
3. **Evaluation beyond AUC**: log loss, expected calibration error (ECE), COPC, reliability diagrams, inference latency.
4. **Calibration**: negative-downsampling correction, Platt scaling, isotonic regression.
5. **Business impact**: a second-price auction simulation that shows what miscalibration costs in money.
6. **Ablations**: embedding size, rare-value threshold, number of cross layers, permutation importance.

Everything runs on a laptop with a synthetic Criteo-like generator, and the same code runs on the real Criteo dataset.

---

## Contents

1. [Read this first: honest scope](#1-read-this-first-honest-scope)
2. [Background from the basics](#2-background-from-the-basics)
3. [Dataset](#3-dataset)
4. [Project structure](#4-project-structure)
5. [Setup](#5-setup)
6. [Quick start](#6-quick-start)
7. [Step-by-step commands](#7-step-by-step-commands)
8. [How the code works](#8-how-the-code-works)
9. [Design decisions and caveats](#9-design-decisions-and-caveats)
10. [Example results (synthetic data)](#10-example-results-synthetic-data)
11. [Running on real Criteo](#11-running-on-real-criteo)
12. [Troubleshooting](#12-troubleshooting)
13. [Extensions](#13-extensions)
14. [Resume bullet template](#14-resume-bullet-template)
15. [References](#15-references)

---

## 1. Read this first: honest scope

* The results in section 10 come from a **synthetic** dataset (600k rows, 2-CPU sandbox). They show that the pipeline works and that the qualitative effects (calibration helps, the downsampling correction matters, and so on) appear. They are **not** benchmark numbers and must not be quoted as real-Criteo results.
* At this small scale **LightGBM beats the deep models and FM is worse than plain LR**. That is expected: interaction models need many millions of rows to beat a well-tuned linear model. This was not verified at larger scale here. Run on real Criteo before drawing conclusions.
* Real Criteo (Kaggle) has **no timestamp**. The code assumes the file order is chronological. If that is not true for your copy, the "chronological" split is just a split by file position.
* The true negative-keep-rate of the Kaggle Criteo data is **unknown**. The downsampling correction therefore takes it as an input (`--neg-keep-rate`, default 1.0 = no correction).
* Fill in your own numbers in the resume template (section 14) only after you have run the real data.

---

## 2. Background from the basics

### 2.1 What is CTR prediction?

An ad exchange has an impression opportunity (a user opens a page). Advertisers bid. To bid sensibly, a bidder needs the probability the user will click:

    pCTR = P(click | user, page, ad, context)

It is a binary classification problem: label 1 = click, 0 = no click. Typical CTR is 1 to 5%, so the data is highly imbalanced.

### 2.2 Why is the data shaped like Criteo?

Each row has 40 columns: `label`, 13 integer features (`I1..I13`, counts such as "number of visits") and 26 categorical features (`C1..C26`, hashed ids such as site, ad, device). Categorical features can have millions of distinct values and most values are rare.

### 2.3 Negative downsampling and why it breaks probabilities

Logs contain far more non-clicks than clicks, so teams keep every click and a fraction `w` of the non-clicks. That speeds up training but changes the base rate, so a model trained on it predicts probabilities that are too high. If the model outputs `p` on downsampled data, the corrected probability is

    q = p / (p + (1 - p) / w)

This is implemented in `downsampling.py` and tested. In a bidder, using `p` instead of `q` makes you overbid. The auction simulation measures exactly this.

### 2.4 Embeddings for categorical features

A categorical value (for example `C7 = "8a3f..."`) becomes an integer id, and the model learns a small vector (the embedding, 16 numbers by default) for each id. Vectors of similar values end up close together. Because tables are huge and each batch touches few rows, we use **sparse gradients** and the **SparseAdam** optimizer so only the used rows are updated.

### 2.5 Models

* **Logistic regression (LR)**: logit = bias + sum of one weight per categorical value + linear term on the dense features. Strong baseline, very fast.
* **Factorization machine (FM)**: LR plus all pairwise interactions between fields, each approximated by a dot product of embeddings. The O(fields x dim) identity used is `sum_{i<j} <v_i, v_j> = 0.5 * ((sum v)^2 - sum v^2)`.
* **DeepFM**: FM for low-order interactions plus an MLP on the same embeddings for high-order ones.
* **DCN-v2**: cross layers `x_next = x0 * (W x + b) + x` build explicit feature crosses of growing order, in parallel with an MLP.
* **LightGBM + target encoding**: gradient-boosted trees cannot use a 100k-level id directly, so each categorical is replaced by its (out-of-fold) mean click rate and its frequency.

### 2.6 Metrics

* **AUC**: probability a random click is ranked above a random non-click. Measures ranking only.
* **Log loss**: penalizes confident wrong probabilities. Sensitive to calibration.
* **ECE** (expected calibration error): sort predictions into quantile bins, compare mean prediction to observed click rate in each bin, average the gaps weighted by bin size.
* **COPC** (clicks over predicted clicks): total actual clicks / total predicted probability. 1.0 is ideal; above 1 means under-prediction.

Why all of them? In ad auctions the *value* of the probability matters, not only the order. A model can have good AUC and still lose money through miscalibration.

### 2.7 Calibration methods

* **Platt scaling**: fit `sigmoid(a * logit(p) + b)` on a held-out split. Two parameters, smooth.
* **Isotonic regression**: fit a monotone step function. More flexible, can overfit on small sets, may produce ties that slightly reduce AUC.

Both are fit on the **calib split** and judged on the **test split** (never fit and evaluate on the same rows).

### 2.8 Second-price auctions

In a second-price auction the winner pays the highest competing bid. The best strategy is **truthful bidding**: bid your true value per impression, `value_per_click * pCTR`. The simulation draws competing bids, lets each bidding strategy play, and reports wins, spend, clicks and surplus (value of clicks won minus price paid). Overestimated pCTR means winning auctions you should lose, so surplus goes negative.

### 2.9 Spark concepts used

* **Explicit schema** when reading the TSV (no slow inference).
* **Melt with `stack`** to count all 26 features in one pass.
* **Window `row_number`** to rank values by frequency inside each feature.
* **Broadcast joins**: each per-feature vocabulary is small, so it is shipped to every executor and the big table is never shuffled.
* **`partitionBy("split")`** when writing Parquet, so each split is its own folder.
* **Fit on train only**: vocabularies and standardization statistics come from the training split to avoid leakage.

---

## 3. Dataset

### 3.1 Real data

The Criteo display-advertising challenge data (Kaggle, `train.txt`, tab separated, about 45M rows, about 11 GB) has the layout:

    label  I1..I13 (integers, may be empty)  C1..C26 (hashed strings, may be empty)

Download it yourself (license terms apply) and pass its path to `preprocess_spark.py`.

### 3.2 Synthetic data (default)

`generate_synthetic_data.py` writes a file in the same layout, with:

* Zipf-distributed categorical values (a few common, a long tail of rare) and cardinalities from 3 to 100,000.
* A hidden ground-truth CTR model with latent-factor interactions, plus a mild time drift.
* Negative downsampling (`--neg-keep-rate`, default 0.1), as in real logs.
* The **oracle AUC** (about 0.807 here), the best any model can possibly achieve, a useful ceiling and sanity check.

### 3.3 Memory

Training data is held in memory as arrays: about **150 bytes per row** (26 int32, 13 float32, label, row id). 10M rows is about 1.5 GB. Use `--max-train-rows` for less.

---

## 4. Project structure

    ctr-prediction-spark/
    ├── README.md
    ├── requirements.txt
    ├── pytest.ini
    ├── run_demo.sh                      # whole pipeline in one command
    ├── ctr_pipeline/
    │   ├── schema.py                    # column names, split names, reserved indices
    │   ├── utils.py                     # sigmoid/logit, seeding, timers, json
    │   ├── synthetic.py                 # synthetic Criteo-like generator
    │   ├── spark_preprocess.py          # PySpark pipeline
    │   ├── dataset.py                   # load Parquet splits into arrays
    │   ├── models.py                    # LR, FM, DeepFM, DCN-v2
    │   ├── trainer.py                   # training loop, early stopping, latency
    │   ├── experiment.py                # train one model and measure everything
    │   ├── gbdt.py                      # target encoding + LightGBM
    │   ├── metrics.py                   # AUC, log loss, ECE, COPC, reliability
    │   ├── calibration.py               # Platt, isotonic
    │   ├── downsampling.py              # negative-downsampling correction
    │   ├── auction_sim.py               # second-price auction simulator
    │   ├── runs.py                      # save/load run artifacts
    │   └── plotting.py                  # figures
    ├── scripts/
    │   ├── generate_synthetic_data.py
    │   ├── preprocess_spark.py
    │   ├── train_models.py
    │   ├── train_gbdt.py
    │   ├── evaluate_calibrate.py
    │   ├── simulate_auction.py
    │   └── run_ablations.py
    └── tests/                           # 29 tests (one slow Spark end-to-end test)

---

## 5. Setup

Requirements: Python 3.9+ (verified with 3.14) and **Java 17 or newer** (needed by Spark; PySpark 4 dropped Java 8 and 11, while PySpark 3.5 still works with Java 11). Always use a **fresh virtual environment**. These exact steps were verified on Ubuntu with Python 3.14, PySpark 4.2, PyTorch 2.14 and pandas 3.0:

    cd ctr-prediction-spark
    python3 -m venv .venv               # Debian/Ubuntu: sudo apt install python3-venv
    source .venv/bin/activate
    unset PYTHONPATH                    # an inherited PYTHONPATH overrides the venv
    export PYTHONNOUSERSITE=1           # stops packages in ~/.local from leaking into the venv
    pip install -r requirements.txt
    python -c "import pandas, torch; print(pandas.__file__)"   # path must be inside .venv
    java -version                       # must work; otherwise: sudo apt install default-jre-headless

Run the tests (about 10 seconds, 28 tests):

    pytest -q -m "not slow"             # skips the Spark end-to-end test
    pytest -q                           # all 29 tests (the Spark test is skipped if Java is missing)

Notes:

* Do all of the above in a **new terminal** after `unset PYTHONPATH` and `PYTHONNOUSERSITE` are set. `run_demo.sh` sets both again itself, but the venv must already be activated.
* Debian/Ubuntu block `pip install` outside a venv (`externally-managed-environment`). Use the venv, not `--break-system-packages`.
* The first install downloads several GB because PyTorch pulls in CUDA libraries. For a CPU-only install: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install -r requirements.txt`.
* If you see `No module named 'pandas._libs...'`, packages from another Python version (`~/.local/lib/python3.x` or a `PYTHONPATH`) are being picked up. Run `echo $PYTHONPATH`, `unset` it, delete `.venv`, and repeat the steps above.
* `deactivate: command not found` just means no venv was active; ignore it.
* Activate the venv in every new terminal before running `run_demo.sh`.

---

## 6. Quick start

    bash run_demo.sh                       # 500k synthetic rows
    N_ROWS=2000000 bash run_demo.sh        # bigger

Runtime: about 2 to 5 minutes on a recent laptop with a GPU (the neural models use CUDA automatically when available); expect 10 to 20 minutes on a 2-core CPU-only machine. Spark prints some warnings at start-up (see Troubleshooting); they are harmless.

Outputs: `data/` (raw and Parquet), `runs/` (per-model predictions, metrics, weights), `results/` (tables and plots), `results/ablations/`.

---

## 7. Step-by-step commands

### 7.1 Generate data

    python scripts/generate_synthetic_data.py --n-rows 500000 --neg-keep-rate 0.1 --output data/criteo_like.tsv

Prints and saves oracle AUC and log loss.

### 7.2 Preprocess with Spark

    python scripts/preprocess_spark.py --input data/criteo_like.tsv --output data/processed --min-count 5

Key options: `--min-count` (values rarer than this share the rare bucket), `--max-vocab`, `--master` (use `spark://...` or `yarn` to scale out), `--driver-memory`. Produces `data/split=train|val|calib|test/` Parquet, `vocab/`, `vocab_counts.npz` and `meta.json`.

Split sizes default to 80% train, 5% val, 5% calib, 10% test **in file order**.

### 7.3 Train the neural models

    python scripts/train_models.py --data data/processed --models lr fm deepfm dcnv2

Options: `--emb-dim`, `--hidden`, `--dropout`, `--n-cross`, `--epochs`, `--batch-size`, `--lr`, `--emb-lr`, `--emb-l2`, `--max-train-rows`, `--tag`, `--device`, `--seed`. Early stopping on validation log loss restores the best checkpoint.

### 7.4 LightGBM baseline

    python scripts/train_gbdt.py --data data/processed

### 7.5 Compare and calibrate

    python scripts/evaluate_calibrate.py --data data/processed

Writes `results/model_comparison.{csv,md}`, `calibration_results.csv`, `reliability_*.png`, `comparison_*.png`.

### 7.6 Auction simulation

    python scripts/simulate_auction.py --data data/processed --neg-keep-rate 0.1

`--neg-keep-rate` **must match** how your data was downsampled. For real Criteo, set it to your best estimate (or leave 1.0 and treat results as relative only). Other options: `--value-per-click`, `--competitor-factor`, `--competitor-sigma`, `--n-sims`.

### 7.7 Ablations

    python scripts/run_ablations.py --data data/processed --ablations emb_dim min_count n_cross importance

The min-count ablation re-maps rare values on the fly using `vocab_counts.npz`, so Spark need not be re-run.

---

## 8. How the code works

| Module | What to look at |
|---|---|
| `spark_preprocess.py` | `build_vocabulary` (stack-melt + window ranking), `apply_vocabulary` (broadcast left joins; index 0 missing, 1 rare, 2+ frequent), `assign_splits` (approxQuantile cuts on `row_id`) |
| `dataset.py` | `load_split` reads Parquet via a row-id filter and sorts by time; `apply_min_count` remaps rare ids for the ablation |
| `models.py` | `FieldEmbeddings` (one table for all fields with per-field offsets), `FactorizationMachine.pairwise_interactions`, `CrossLayer`, `DCNv2` |
| `trainer.py` | SparseAdam for embeddings and Adam for dense weights; validation four times per epoch; early stopping; CPU latency measurement |
| `gbdt.py` | `TargetEncoder` is **out-of-fold** on the training split (prevents label leakage), plain on val/test |
| `metrics.py` | `ece` with quantile bins, `copc`, `reliability_table` |
| `calibration.py` | `PlattScaler`, `IsotonicCalibrator` share a `fit/transform` interface |
| `auction_sim.py` | `simulate_second_price`: competitor bids are log-normal around the true value times a factor |

---

## 9. Design decisions and caveats

* **Chronological split** (not random) mimics deployment: train on the past, test on the future. Rows are assumed to be in time order.
* **Vocabulary from train only**; unseen test values fall into the rare bucket.
* **Separate calib split** so calibrators are never fit on test rows.
* **Test AUC is unchanged by Platt** (monotone transform). Isotonic can lower it slightly through ties.
* **Parameter counts**: `params_total` does not change with `min_count`, because the embedding tables stay allocated. For LightGBM "params" are leaf counts (an upper bound).
* **L2 on embeddings** is implemented as an optional flag (`--emb-l2`). It barely helped on the synthetic data; default is 0.
* **Latency** is measured on CPU at batch sizes 1 and 1024. It is hardware dependent, so compare models to each other, not to any production number.
* **Synthetic ceiling**: no model can exceed the oracle AUC (about 0.807).

---

## 10. Example results (synthetic data)

Setup: 600k synthetic rows (neg-keep 0.1), 2-CPU sandbox, `min-count 5`, default hyperparameters, 2 epochs. Single seed, so differences of about 0.002 AUC are within noise.

### 10.1 Model comparison (test split)

| model | AUC | log loss | ECE | train s | params |
|---|---|---|---|---|---|
| LightGBM | 0.7235 | 0.5089 | 0.0228 | 26 | 16,569 (leaves) |
| DeepFM | 0.7101 | 0.5163 | 0.0183 | 20 | 1,515,274 |
| DCN-v2 | 0.7093 | 0.5172 | 0.0173 | 39 | 1,988,384 |
| LR | 0.7028 | 0.5221 | 0.0281 | 4 | 80,729 |
| FM | 0.6932 | 0.5302 | 0.0367 | 11 | 1,372,169 |

Oracle AUC is about 0.807. FM below LR is an overfitting effect at this data size.

### 10.2 Calibration (test ECE, calibrators fit on the calib split)

| model | raw | Platt | isotonic |
|---|---|---|---|
| LR | 0.0281 | 0.0072 | 0.0120 |
| FM | 0.0367 | 0.0164 | 0.0096 |
| DeepFM | 0.0183 | 0.0095 | 0.0130 |
| DCN-v2 | 0.0173 | 0.0116 | 0.0096 |
| LightGBM | 0.0228 | 0.0072 | 0.0078 |

Both methods cut ECE substantially; COPC moves from about 1.07-1.11 to about 1.03.

### 10.3 Auction simulation (surplus per 1,000 impressions)

| strategy | LR | DeepFM | DCN-v2 | LightGBM |
|---|---|---|---|---|
| raw pCTR (no downsampling correction) | -7.98 | -6.86 | -6.66 | -6.05 |
| downsampling corrected | 11.21 | 11.56 | 11.52 | 11.94 |
| Platt + corrected | 11.38 | 11.61 | 11.56 | 12.03 |
| isotonic + corrected | 11.37 | 11.58 | 11.55 | 12.03 |

A constant-bid baseline earns about 8.2. Bidding on uncorrected probabilities wins about 92% of auctions and loses money; correcting wins about 40% and is profitable. Calibration adds a smaller further gain.

### 10.4 Ablations (DeepFM for embedding/min-count, DCN-v2 for cross layers)

| ablation | setting -> test AUC |
|---|---|
| embedding dim | 4: 0.7069, 8: 0.7073, 16: 0.7092, 32: 0.7146 |
| min count | 5: 0.7092, 20: 0.7094, 50: 0.7190, 200: 0.7187, 1000: 0.7112 |
| cross layers | 0: 0.7052, 1: 0.7073, 2: 0.7088, 3: 0.7093, 4: 0.7095 |

Takeaways: larger embeddings help up to the largest size tried; a stricter rare threshold regularizes the long tail (best at 50 to 200, too aggressive at 1000); cross layers help with diminishing returns. Permutation importance (`permutation_importance.csv`) ranks the strongest categorical features (C23, C4, C13, C6) above most integer features.

---

### 10.5 A second, independent run

The same demo (`bash run_demo.sh`, 500k rows, GPU, Python 3.14, different random draws than the table above) gave the same picture, which shows the findings are not one-off:

| model | test AUC | test log loss |
|---|---|---|
| LightGBM | 0.7239 | 0.5088 |
| DCN-v2 | 0.7067 | 0.5190 |
| DeepFM | 0.7060 | 0.5185 |
| LR | 0.6992 | 0.5263 |
| FM | 0.6931 | 0.5279 |

Platt and isotonic cut ECE from about 0.017 to 0.035 down to about 0.007 to 0.015; COPC moved from about 1.04 to 1.09 to about 0.99. In the auction simulation, bidding on uncorrected probabilities lost 7 to 10 per 1,000 impressions, while corrected bids earned 10.5 to 11.6. Embedding size 32 beat 4 (0.7079 vs 0.7036 AUC), `min_count` of 50 to 200 was best, and more cross layers helped slightly. Exact numbers vary by hardware and seed, so judge by the ranking and the direction of each effect.

---

## 11. Running on real Criteo

1. Put `train.txt` somewhere with enough disk. Optionally take the first N rows: `head -n 5000000 train.txt > data/criteo_5m.tsv`.
2. `python scripts/preprocess_spark.py --input data/criteo_5m.tsv --output data/processed --min-count 10 --driver-memory 8g`
3. `python scripts/train_models.py --data data/processed --epochs 1`
4. Everything else is unchanged. For `simulate_auction.py` pass your best estimate of `--neg-keep-rate`.
5. On a cluster, change `--master` and give Spark more shuffle partitions (`--shuffle-partitions`).

Expect interaction models (DeepFM, DCN-v2) to overtake LR at millions of rows.

---

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'pandas._libs...'` / `externally-managed-environment` | Use a fresh venv with `PYTHONNOUSERSITE=1` (see Setup) |
| `JAVA_HOME is not set` / Spark fails to start | Install JDK 17 or newer; check `java -version` |
| Spark prints `WARN Utils: Your hostname ... resolves to a loopback address`, `jdk.incubator.vector`, `package sun.security.action not in java.base`, or `NativeCodeLoader` warnings | Harmless. Set `SPARK_LOCAL_IP=127.0.0.1` to silence the hostname one |
| `FutureWarning: PySpark does not yet fully support pandas >= 3.0.0` | Harmless here: the pipeline only uses pandas outside Spark. Install `pandas<3` to silence it |
| Spark out of memory | Raise `--driver-memory`; lower `--shuffle-partitions` for small data, raise it for big data |
| Python out of memory when training | Use `--max-train-rows`; about 150 B per row |
| FM or DeepFM overfits quickly | Lower `--emb-lr`, use a smaller `--batch-size`, raise `--min-count` |
| Slow on CPU | Use fewer rows, fewer epochs, or `--device cuda` (used automatically when a GPU is available) |
| `pip install pyspark` conflicts | Use a clean virtualenv; match your Java version |
| Auction numbers look wrong | `--neg-keep-rate` must match the downsampling of your data |

---

## 13. Extensions

* Multi-task heads (CTR + conversion) with a shared embedding table.
* Hashing trick instead of vocabularies to bound memory.
* Time-aware features and online/incremental training with a rolling window.
* Delayed-feedback conversion modeling.
* Quantile-aware or distribution-aware calibration per segment (device, site).
* Model distillation or int8 quantization for latency.
* Replace the log-normal competitor model with a learned bid-landscape model.

---

## 14. Resume bullet template

Replace the placeholders with numbers **you measured**:

* Built a PySpark pipeline (vocabulary construction, rare-value bucketing, chronological split) for `<N>`M-row Criteo-format ad logs, and trained LR, FM, DeepFM, DCN-v2 and LightGBM CTR models, reaching `<AUC>` test AUC.
* Diagnosed and fixed probability miscalibration with negative-downsampling correction and Platt/isotonic calibration, reducing ECE from `<x>` to `<y>`.
* Simulated second-price auctions to quantify the revenue impact of calibration, showing uncorrected predictions lose money while calibrated bids earn `<z>` surplus per 1k impressions.

---

## 15. References

* Rendle, *Factorization Machines* (2010).
* Guo et al., *DeepFM* (2017).
* Wang et al., *DCN V2* (2021).
* He et al., *Practical Lessons from Predicting Clicks on Ads at Facebook* (2014): downsampling and calibration.
* Platt, *Probabilistic Outputs for SVMs* (1999); Niculescu-Mizil and Caruana, *Predicting Good Probabilities with Supervised Learning* (2005).
* Vickrey, *Counterspeculation, Auctions, and Competitive Sealed Tenders* (1961).
* Criteo Display Advertising Challenge (Kaggle).
