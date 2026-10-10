"""Metrics are correct, and the calibrators fix a deliberately miscalibrated model."""

import numpy as np

from ctr_pipeline.calibration import IsotonicCalibrator, PlattScaler
from ctr_pipeline.metrics import auc, copc, ece, logloss, reliability_table
from ctr_pipeline.utils import sigmoid


def make_overconfident_model(n=200_000, seed=0):
    """True click probabilities, and a model whose logits are exaggerated and shifted."""
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(-3.0, 1.5, n)
    y = (rng.random(n) < sigmoid(true_logit)).astype(float)
    return y, sigmoid(true_logit), sigmoid(2.0 * true_logit + 1.0)


def test_logloss_matches_the_formula():
    assert np.isclose(logloss(np.array([1.0, 0.0]), np.array([0.8, 0.2])), -np.log(0.8))


def test_auc_perfect_and_random():
    assert auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_copc_is_one_for_unbiased_totals():
    y = np.array([1.0, 0.0, 0.0, 1.0])
    assert np.isclose(copc(y, np.full(4, 0.5)), 1.0)
    assert np.isclose(copc(y, np.full(4, 1.0)), 0.5)


def test_reliability_table_counts_every_row():
    rng = np.random.default_rng(0)
    p = rng.random(10_000)
    table = reliability_table((rng.random(10_000) < p).astype(float), p, n_bins=20)
    assert table["count"].sum() == 10_000


def test_ece_is_small_for_calibrated_and_large_for_miscalibrated():
    y, p_true, p_model = make_overconfident_model()
    assert ece(y, p_true) < 0.003
    assert ece(y, p_model) > 0.03


def test_platt_scaling_fixes_calibration_and_keeps_the_ranking():
    y, _, p_model = make_overconfident_model()
    half = len(y) // 2
    scaler = PlattScaler().fit(p_model[:half], y[:half])
    fixed = scaler.transform(p_model[half:])

    assert ece(y[half:], fixed) < 0.2 * ece(y[half:], p_model[half:])
    assert np.isclose(auc(y[half:], fixed), auc(y[half:], p_model[half:]))


def test_isotonic_is_monotone_and_improves_calibration():
    y, _, p_model = make_overconfident_model()
    half = len(y) // 2
    calibrator = IsotonicCalibrator().fit(p_model[:half], y[:half])

    grid = np.linspace(p_model.min(), p_model.max(), 500)
    assert np.all(np.diff(calibrator.transform(grid)) >= -1e-12)
    assert ece(y[half:], calibrator.transform(p_model[half:])) < 0.2 * ece(y[half:], p_model[half:])
