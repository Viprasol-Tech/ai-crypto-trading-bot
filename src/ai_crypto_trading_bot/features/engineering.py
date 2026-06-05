"""Feature engineering — turn a price window into model-ready features.

These are the inputs an alpha model consumes. Keeping feature extraction separate
means you can swap the model (rule-based, sklearn, gradient boosting, an LLM
score) without touching the rest of the pipeline.

Part of AI Crypto Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class Features:
    """Normalised features computed from a price window."""

    momentum: float  # recent return, scaled
    mean_reversion: float  # z-score of price vs its mean (negated)
    volatility: float  # realised volatility of returns
    trend_strength: float  # fraction of up-moves minus 0.5, scaled


def compute_features(prices: np.ndarray) -> Features:
    """Compute features from a 1-D array of recent prices.

    Returns zeros when there is not enough history (fewer than 3 points).
    """
    if prices.size < 3:
        return Features(0.0, 0.0, 0.0, 0.0)

    returns = np.diff(prices) / prices[:-1]
    momentum = float(np.tanh(returns[-min(5, returns.size) :].mean() * 50.0))

    mean = float(prices.mean())
    std = float(prices.std()) or 1.0
    mean_reversion = float(np.tanh(-(prices[-1] - mean) / std))

    volatility = float(np.tanh(returns.std() * 20.0))
    trend_strength = float(np.tanh(((returns > 0).mean() - 0.5) * 4.0))

    return Features(momentum, mean_reversion, volatility, trend_strength)
