"""A simple second-price auction simulation that shows why calibrated probabilities matter.

For every impression we bid  bid = value_per_click * predicted_CTR  against one competing price.
If we win we pay the competing price (second-price rule) and earn value_per_click if the user
clicked. In a second-price auction the best strategy is to bid the TRUE expected value, so a model
whose probabilities are too high overbids (wins impressions that cost more than they are worth)
and a model whose probabilities are too low underbids (misses profitable impressions).
"""

import numpy as np
import pandas as pd


def simulate_second_price(
    p_bid,
    label,
    sample_weight,
    value_per_click=1.0,
    competitor_median=0.03,
    competitor_sigma=1.0,
    n_sims=20,
    seed=0,
):
    """Replay the logged impressions through random auctions and summarize the outcome.

    p_bid         : the CTR our model predicts on the TRUE (not downsampled) scale
    label         : realized click (0/1) of each logged impression
    sample_weight : how many real impressions each logged row represents (1 for clicks,
                    1/keep_rate for kept non-clicks), which undoes the downsampling
    competitor_*  : the competing price is lognormal with this median and sigma (in value units)
    Returns a one-row-per-metric table: mean and standard deviation over `n_sims` random price draws.
    The same seed gives the same competing prices, so different models are compared fairly.
    """
    rng = np.random.default_rng(seed)
    per_thousand = 1000.0 / sample_weight.sum()  # report everything per 1000 real impressions
    rows = []

    for _ in range(n_sims):
        competitor_price = competitor_median * rng.lognormal(0.0, competitor_sigma, size=len(p_bid))

        # We win when our bid beats the competitor, and then pay the competitor's price
        win = value_per_click * p_bid > competitor_price
        w = sample_weight * win

        clicks = np.sum(w * label)
        spend = np.sum(w * competitor_price)
        value = value_per_click * clicks
        rows.append(
            {
                "win_rate": np.sum(w) / sample_weight.sum(),
                "clicks_per_1k": clicks * per_thousand,
                "spend_per_1k": spend * per_thousand,
                "value_per_1k": value * per_thousand,
                "surplus_per_1k": (value - spend) * per_thousand,
                "roi": value / spend if spend > 0 else np.nan,
            }
        )

    frame = pd.DataFrame(rows)
    return pd.DataFrame({"mean": frame.mean(), "std": frame.std()})
