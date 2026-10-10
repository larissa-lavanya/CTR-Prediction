"""Generate Criteo-format data (label, 13 ints, 26 hex categoricals) from a known click model.

Why synthetic data? It lets you run and test the whole pipeline without downloading 11 GB,
and because the true click model is known we can sanity-check results (oracle AUC).
The generator mimics the properties of the real data that matter for this project:
    * heavy-tailed (Zipf) categorical values with a huge range of cardinalities,
    * missing values in both dense and categorical features,
    * feature interactions plus a non-linear term (so deep models have something to learn),
    * negatives downsampled so that ~25% of rows are clicks while the true CTR is a few percent,
    * rows written in time order, with a slow drift in the click rate.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ctr_pipeline.schema import N_DENSE, N_SPARSE
from ctr_pipeline.utils import sigmoid

# Number of distinct values of each categorical feature (spans 3 to 100,000)
CARDINALITIES = [
    1000, 500, 100000, 50000, 200, 15, 10000, 400, 3, 40000, 5000, 80000, 3000,
    25, 8000, 60000, 10, 3000, 1500, 4, 70000, 15, 20, 30000, 60, 20000,
]
assert len(CARDINALITIES) == N_SPARSE


class SyntheticCTRModel:
    """Ground-truth click model: linear + pairwise interactions + a non-linear dense term."""

    def __init__(self, seed=0, target_ctr=0.03, drift=0.3, zipf_exponent=1.1, latent_dim=4, signal=0.62):
        rng = np.random.default_rng(seed)
        self.drift = drift

        # `signal` scales every effect; 0.62 gives an oracle AUC of about 0.80 (similar to real Criteo)

        # Zipf-like popularity of the values of each categorical feature
        self.cdfs = []
        for card in CARDINALITIES:
            weights = 1.0 / np.arange(1, card + 1) ** zipf_exponent
            self.cdfs.append(np.cumsum(weights) / weights.sum())

        # A fixed 8-character hex string for every categorical value (like the real data)
        self.hex_tables = [
            np.array([f"{v:08x}" for v in rng.integers(0, 2**32, size=card)]) for card in CARDINALITIES
        ]

        # Missing-value rates and distributions of the dense count features
        self.cat_missing_rate = rng.uniform(0.0, 0.05, N_SPARSE)
        self.dense_missing_rate = rng.uniform(0.0, 0.4, N_DENSE)
        self.dense_mu = rng.uniform(0.5, 3.0, N_DENSE)
        self.dense_sigma = rng.uniform(0.8, 1.6, N_DENSE)

        # Per-value linear effects: a few "important" features, the rest nearly irrelevant
        important = rng.permutation(N_SPARSE)[:12]
        scale = np.full(N_SPARSE, 0.08)
        scale[important] = 0.35
        self.linear_effects = [signal * rng.normal(0.0, scale[f], card) for f, card in enumerate(CARDINALITIES)]

        # Pairwise interactions between 10 random pairs of features (what FM-style models capture)
        self.latents = [np.sqrt(signal) * rng.normal(0.0, 0.5, (card, latent_dim)) for card in CARDINALITIES]
        self.pairs = [tuple(rng.choice(N_SPARSE, size=2, replace=False)) for _ in range(10)]

        # Dense effects (linear and one non-linear interaction that an MLP can learn)
        self.dense_weights = signal * rng.normal(0.0, 0.25, N_DENSE)
        self.intercept = 0.0
        self.intercept = self._calibrate_intercept(target_ctr, rng)

    # ------------------------------------------------------------------ sampling

    def sample_features(self, rng, n):
        """Draw n impressions: dense counts (NaN = missing) and category indices (-1 = missing)."""
        # Dense features: log-normal counts, some missing, I2 can be negative like the real data
        dense = np.floor(np.exp(self.dense_mu + self.dense_sigma * rng.standard_normal((n, N_DENSE))))
        dense[:, 1] = np.floor(rng.normal(20.0, 60.0, n))
        dense[rng.random((n, N_DENSE)) < self.dense_missing_rate] = np.nan

        # Categorical features: Zipf-distributed value ids, some missing
        sparse = np.empty((n, N_SPARSE), dtype=np.int64)
        for f in range(N_SPARSE):
            sparse[:, f] = np.minimum(np.searchsorted(self.cdfs[f], rng.random(n)), CARDINALITIES[f] - 1)
        sparse[rng.random((n, N_SPARSE)) < self.cat_missing_rate] = -1
        return dense, sparse

    def logit(self, dense, sparse, t):
        """True click logit; t in [0, 1] is the position in time (drives the drift)."""
        n = len(dense)

        # Standardized log-counts (missing -> 0)
        z = (np.log1p(np.maximum(np.nan_to_num(dense, nan=0.0), 0.0)) - self.dense_mu) / self.dense_sigma
        z[np.isnan(dense)] = 0.0
        score = z @ self.dense_weights + 0.5 * np.tanh(z[:, 0] * z[:, 2])  # linear + non-linear

        # Linear effect of each categorical value
        for f in range(N_SPARSE):
            present = sparse[:, f] >= 0
            score[present] += self.linear_effects[f][sparse[present, f]]

        # Pairwise interactions through latent vectors
        for f, g in self.pairs:
            both = (sparse[:, f] >= 0) & (sparse[:, g] >= 0)
            score[both] += np.sum(self.latents[f][sparse[both, f]] * self.latents[g][sparse[both, g]], axis=1)

        return score + self.intercept + self.drift * (t - 0.5)

    def _calibrate_intercept(self, target_ctr, rng, n_pilot=200_000):
        """Bisection on the intercept so the average true CTR (mid-period) equals target_ctr."""
        dense, sparse = self.sample_features(rng, n_pilot)
        base = self.logit(dense, sparse, np.full(n_pilot, 0.5))
        low, high = -15.0, 5.0
        for _ in range(40):
            mid = 0.5 * (low + high)
            if sigmoid(base + mid).mean() > target_ctr:
                high = mid
            else:
                low = mid
        return 0.5 * (low + high)

    # ------------------------------------------------------------------ output

    def to_frame(self, dense, sparse, labels):
        """Build the raw-format DataFrame: label, I1..I13 (empty = missing), C1..C26 (hex, empty = missing)."""
        columns = {"label": labels.astype(np.int64)}
        for j in range(N_DENSE):
            columns[f"I{j + 1}"] = pd.array(dense[:, j], dtype="Float64").astype("Int64")
        for f in range(N_SPARSE):
            values = self.hex_tables[f][np.maximum(sparse[:, f], 0)].astype(object)
            values[sparse[:, f] < 0] = ""
            columns[f"C{f + 1}"] = values
        return pd.DataFrame(columns)


def write_criteo_like_tsv(path, n_rows, seed=0, target_ctr=0.03, neg_keep_rate=0.1, drift=0.3,
                          chunk_size=200_000, log=print):
    """Write `n_rows` rows to a tab-separated file in the Criteo layout.

    Impressions are generated at the true CTR, then every negative is kept only with
    probability `neg_keep_rate` (positives are always kept), exactly like the downsampling
    behind the public Criteo data. Returns oracle statistics on the kept rows.
    """
    model = SyntheticCTRModel(seed=seed, target_ctr=target_ctr, drift=drift)
    rng = np.random.default_rng(seed + 1)

    # Expected number of raw impressions we need to generate (used to place rows in time)
    keep_fraction = target_ctr + (1.0 - target_ctr) * neg_keep_rate
    expected_raw = n_rows / keep_fraction

    written, raw_seen = 0, 0
    kept_p, kept_y = [], []
    first_chunk = True
    while written < n_rows:
        dense, sparse = model.sample_features(rng, chunk_size)
        t = np.minimum((raw_seen + np.arange(chunk_size)) / expected_raw, 1.0)
        p_true = sigmoid(model.logit(dense, sparse, t))
        clicks = (rng.random(chunk_size) < p_true).astype(np.int64)
        raw_seen += chunk_size

        # Keep every click, and a fraction of the non-clicks
        keep = (clicks == 1) | (rng.random(chunk_size) < neg_keep_rate)
        keep = np.flatnonzero(keep)[: n_rows - written]
        frame = model.to_frame(dense[keep], sparse[keep], clicks[keep])
        frame.to_csv(path, sep="\t", header=False, index=False, mode="w" if first_chunk else "a", na_rep="")
        first_chunk = False

        written += len(keep)
        kept_p.append(p_true[keep])
        kept_y.append(clicks[keep])
        log(f"  wrote {written:,} / {n_rows:,} rows")

    p, y = np.concatenate(kept_p), np.concatenate(kept_y)
    p_kept_scale = p / (p + (1.0 - p) * neg_keep_rate)  # what a perfect model predicts on the kept rows
    return {
        "rows": int(written),
        "click_rate_kept_rows": float(y.mean()),
        "true_ctr_estimate": float(y.mean() * neg_keep_rate / (y.mean() * neg_keep_rate + (1 - y.mean()))),
        "oracle_auc": float(roc_auc_score(y, p)),
        "oracle_logloss": float(-np.mean(y * np.log(p_kept_scale) + (1 - y) * np.log(1 - p_kept_scale))),
        "neg_keep_rate": neg_keep_rate,
    }
