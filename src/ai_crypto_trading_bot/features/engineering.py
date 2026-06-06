"""Feature engineering — turn a price window into model-ready features.

These are the inputs an alpha model consumes. Keeping feature extraction separate
means you can swap the model (rule-based, logistic, gradient boosting, an LLM
score) without touching the rest of the pipeline.

All features are squashed into ``[-1, 1]`` (or ``[0, 1]`` where a bounded
oscillator is more natural) so models can combine them without re-scaling.

Part of AI Crypto Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np

# Minimum number of price points before features are meaningful.
MIN_HISTORY = 3


@dataclass(slots=True, frozen=True)
class Features:
    """Normalised features computed from a price window.

    Every field is bounded so an alpha model can weight them directly. The first
    four fields are the original v0.1 features; the remaining ones were added in
    v0.2 to give models a richer view of the tape.
    """

    momentum: float  # recent return, scaled to [-1, 1]
    mean_reversion: float  # z-score of price vs its mean (negated), [-1, 1]
    volatility: float  # realised volatility of returns, [0, 1]
    trend_strength: float  # fraction of up-moves minus 0.5, scaled to [-1, 1]
    rsi: float  # Wilder RSI re-centred to [-1, 1] (>0 = overbought)
    macd: float  # MACD histogram, scaled to [-1, 1]
    bollinger: float  # position within Bollinger band, [-1, 1]
    acceleration: float  # change in momentum (2nd derivative), [-1, 1]

    def as_array(self) -> np.ndarray:
        """Return the features as a 1-D float array in declaration order."""
        return np.asarray(astuple(self), dtype=float)

    @classmethod
    def zeros(cls) -> Features:
        """Return an all-zero feature vector (neutral signal)."""
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _rsi(returns: np.ndarray, period: int = 14) -> float:
    """Wilder RSI on a return series, re-centred to ``[-1, 1]``.

    Returns ``0.0`` (neutral) when there is no movement at all.
    """
    if returns.size == 0:
        return 0.0
    window = returns[-min(period, returns.size) :]
    gains = window[window > 0].sum()
    losses = -window[window < 0].sum()
    if gains + losses == 0:
        return 0.0
    rs_index = 100.0 * gains / (gains + losses)  # 0..100
    return float((rs_index - 50.0) / 50.0)


def _ema(values: np.ndarray, span: int) -> float:
    """Exponential moving average of ``values`` (most recent weighted most)."""
    if values.size == 0:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    weights = (1.0 - alpha) ** np.arange(values.size - 1, -1, -1)
    return float((values * weights).sum() / weights.sum())


def compute_features(prices: np.ndarray) -> Features:
    """Compute features from a 1-D array of recent prices.

    Returns a neutral all-zero vector when there is not enough history (fewer
    than :data:`MIN_HISTORY` points).
    """
    prices = np.asarray(prices, dtype=float)
    if prices.size < MIN_HISTORY:
        return Features.zeros()

    returns = np.diff(prices) / prices[:-1]
    momentum = float(np.tanh(returns[-min(5, returns.size) :].mean() * 50.0))

    mean = float(prices.mean())
    std = float(prices.std()) or 1.0
    mean_reversion = float(np.tanh(-(prices[-1] - mean) / std))

    volatility = float(np.tanh(returns.std() * 20.0))
    trend_strength = float(np.tanh(((returns > 0).mean() - 0.5) * 4.0))

    rsi = _rsi(returns)

    fast = _ema(prices, span=12)
    slow = _ema(prices, span=26)
    signal_scale = std or 1.0
    macd = float(np.tanh((fast - slow) / signal_scale))

    bollinger = float(np.tanh((prices[-1] - mean) / (2.0 * std)))

    if returns.size >= 2:
        recent = returns[-min(5, returns.size) :]
        first_half = recent[: max(1, recent.size // 2)].mean()
        second_half = recent[recent.size // 2 :].mean()
        acceleration = float(np.tanh((second_half - first_half) * 50.0))
    else:
        acceleration = 0.0

    return Features(
        momentum=momentum,
        mean_reversion=mean_reversion,
        volatility=volatility,
        trend_strength=trend_strength,
        rsi=rsi,
        macd=macd,
        bollinger=bollinger,
        acceleration=acceleration,
    )
