"""Gradient-boosted trees baseline: target-encode the categoricals, then train LightGBM.

Trees cannot use hashed ids directly, so each categorical field is turned into two numbers:
    * a smoothed click rate of its value (target encoding), computed OUT-OF-FOLD on the training
      rows so a row never sees its own label, and
    * the log of how often the value occurs in the training split (frequency encoding).
"""

import lightgbm as lgb
import numpy as np

from ctr_pipeline.schema import FIRST_VOCAB_INDEX


class TargetEncoder:
    """Smoothed per-value click rates: (clicks + prior * m) / (impressions + m)."""

    def __init__(self, vocab_sizes, n_folds=5, smoothing=20.0, seed=0):
        self.vocab_sizes = vocab_sizes
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.seed = seed

    def fit_transform(self, sparse, y):
        """Fit on the training rows and return their out-of-fold encodings."""
        n, n_fields = sparse.shape
        self.prior = float(y.mean())
        self.tables, self.log_counts = [], []
        folds = np.random.default_rng(self.seed).integers(0, self.n_folds, size=n)

        encoded = np.zeros((n, n_fields), dtype=np.float32)
        for f in range(n_fields):
            idx, size = sparse[:, f], self.vocab_sizes[f]
            total_pos = np.bincount(idx, weights=y, minlength=size)
            total_cnt = np.bincount(idx, minlength=size).astype(np.float64)

            # Out-of-fold: statistics of all OTHER folds only
            for k in range(self.n_folds):
                rows = folds == k
                fold_pos = np.bincount(idx[rows], weights=y[rows], minlength=size)
                fold_cnt = np.bincount(idx[rows], minlength=size)
                rate = (total_pos - fold_pos + self.prior * self.smoothing) / (total_cnt - fold_cnt + self.smoothing)
                encoded[rows, f] = rate[idx[rows]]

            # Full-training statistics, used later for validation and test rows
            self.tables.append(((total_pos + self.prior * self.smoothing) / (total_cnt + self.smoothing)).astype(np.float32))
            self.log_counts.append(np.log1p(total_cnt).astype(np.float32))
        return encoded

    def transform(self, sparse):
        """Encode unseen rows with the full-training statistics."""
        columns = [self.tables[f][sparse[:, f]] for f in range(sparse.shape[1])]
        return np.stack(columns, axis=1)

    def frequency_features(self, sparse):
        columns = [self.log_counts[f][sparse[:, f]] for f in range(sparse.shape[1])]
        return np.stack(columns, axis=1)


def build_features(encoder, data, target_encoding):
    """Dense features + target-encoded categoricals + log-frequency of each categorical value."""
    return np.hstack([data.dense, target_encoding, encoder.frequency_features(data.sparse)]).astype(np.float32)


def train_lightgbm(x_train, y_train, x_val, y_val, max_rounds=1000, learning_rate=0.1, num_leaves=63, seed=0):
    """LightGBM with early stopping on validation log loss. Returns the booster."""
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": seed,
        "verbose": -1,
    }
    train_set = lgb.Dataset(x_train, label=y_train)
    val_set = lgb.Dataset(x_val, label=y_val, reference=train_set)
    return lgb.train(
        params,
        train_set,
        num_boost_round=max_rounds,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
