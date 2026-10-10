"""Negative-downsampling correction.

Public CTR datasets keep every click but only a fraction `keep_rate` of the non-clicks.
A model trained on such data predicts p_s = P(click | row was kept), which is much larger than
the real click probability p. Bayes' rule links them:

        p_s = p / (p + (1 - p) * keep_rate)          (downsample_probability)
        p   = keep_rate * p_s / (keep_rate * p_s + 1 - p_s)   (correct_downsampled_probability)
"""


def downsample_probability(p, keep_rate):
    """True click probability -> probability on the downsampled data."""
    return p / (p + (1.0 - p) * keep_rate)


def correct_downsampled_probability(p_s, keep_rate):
    """Probability on the downsampled data -> true click probability."""
    return keep_rate * p_s / (keep_rate * p_s + 1.0 - p_s)
