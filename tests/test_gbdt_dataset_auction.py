"""Target encoding has no label leakage; min-count remapping; the auction simulator behaves."""

import numpy as np

from ctr_pipeline.auction_sim import simulate_second_price
from ctr_pipeline.dataset import apply_min_count
from ctr_pipeline.gbdt import TargetEncoder


def test_out_of_fold_target_encoding_never_sees_its_own_label():
    n = 2000
    rng = np.random.default_rng(0)

    # Every row has a UNIQUE category value, so a leaky encoder could read the label back
    sparse = np.arange(n, dtype=np.int64).reshape(-1, 1)
    y = (rng.random(n) < 0.3).astype(np.float64)

    encoder = TargetEncoder(vocab_sizes=[n], n_folds=5)
    encoded = encoder.fit_transform(sparse, y)

    # Out-of-fold, the value has never been seen, so the encoding is just the global prior
    assert np.allclose(encoded[:, 0], y.mean(), atol=1e-6)


def test_target_encoding_learns_an_informative_category():
    rng = np.random.default_rng(1)
    sparse = rng.integers(0, 2, (50_000, 1)).astype(np.int64)
    y = (rng.random(50_000) < np.where(sparse[:, 0] == 1, 0.5, 0.05)).astype(np.float64)

    encoder = TargetEncoder(vocab_sizes=[2])
    encoder.fit_transform(sparse, y)
    assert encoder.transform(np.array([[1]]))[0, 0] > 0.45
    assert encoder.transform(np.array([[0]]))[0, 0] < 0.08


def test_min_count_sends_rare_values_to_the_rare_index():
    counts = {"sparse_0": np.array([0, 0, 100, 5, 50, 1])}
    sparse = np.array([[0], [1], [2], [3], [4], [5]], dtype=np.int32)
    remapped = apply_min_count(sparse, counts, threshold=10)
    assert remapped[:, 0].tolist() == [0, 1, 2, 1, 4, 1]  # values 3 and 5 are too rare


def make_market(n=300_000, seed=0):
    """Impressions with a known click probability and a competing-price scale."""
    rng = np.random.default_rng(seed)
    p_true = np.clip(rng.lognormal(np.log(0.03), 0.8, n), 1e-4, 0.9)
    label = (rng.random(n) < p_true).astype(float)
    return p_true, label, np.ones(n)


def test_truthful_bidding_beats_overbidding_and_underbidding():
    p_true, label, weight = make_market()
    surplus = {}
    scales = {"truthful": 1.0, "over x3": 3.0, "over x30": 30.0, "under x0.3": 0.3}
    for name, scale in scales.items():
        result = simulate_second_price(np.clip(scale * p_true, 0, 1), label, weight, competitor_median=0.03, n_sims=5)
        surplus[name] = result.loc["surplus_per_1k", "mean"]

    # Truthful bidding is optimal in a second-price auction; any distortion earns less
    assert surplus["truthful"] > surplus["over x3"] > surplus["over x30"]
    assert surplus["truthful"] > surplus["under x0.3"]

    # Winning nearly everything at inflated bids pays more than the clicks are worth
    assert surplus["over x30"] < 0


def test_simulation_is_reproducible():
    p_true, label, weight = make_market(n=20_000)
    first = simulate_second_price(p_true, label, weight, n_sims=3, seed=7)
    second = simulate_second_price(p_true, label, weight, n_sims=3, seed=7)
    assert first.equals(second)
