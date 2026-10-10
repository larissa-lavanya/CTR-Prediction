"""The negative-downsampling correction must invert the downsampling exactly."""

import numpy as np

from ctr_pipeline.downsampling import correct_downsampled_probability, downsample_probability


def test_round_trip():
    p = np.array([0.0005, 0.03, 0.2, 0.5, 0.9])
    for keep in (0.01, 0.1, 0.5):
        recovered = correct_downsampled_probability(downsample_probability(p, keep), keep)
        assert np.allclose(recovered, p)


def test_keeping_every_negative_changes_nothing():
    p = np.array([0.01, 0.3, 0.8])
    assert np.allclose(correct_downsampled_probability(p, 1.0), p)


def test_correction_lowers_probabilities():
    p_downsampled = np.array([0.1, 0.25, 0.6])
    assert np.all(correct_downsampled_probability(p_downsampled, 0.1) < p_downsampled)


def test_recovers_the_true_ctr_from_simulated_downsampling():
    rng = np.random.default_rng(0)
    true_ctr, keep = 0.03, 0.1

    # Generate impressions, then keep all clicks and a fraction of non-clicks
    clicks = rng.random(3_000_000) < true_ctr
    kept = clicks | (rng.random(len(clicks)) < keep)
    ctr_in_kept_rows = clicks[kept].mean()

    assert ctr_in_kept_rows > 0.2  # looks nothing like the true CTR
    corrected = correct_downsampled_probability(ctr_in_kept_rows, keep)
    assert abs(corrected - true_ctr) / true_ctr < 0.03
