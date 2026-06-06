"""Market-regime detection.

Classifies a trailing price window into one of four regimes so the engine can
adapt: scale risk down in turbulent markets, lean into trends, and trade more
cautiously when the tape is ranging. The classifier is rule-based and fully
transparent — no training required — but exposes the raw statistics it used so
you can build your own classifier on top.

Part of AI Crypto Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Regime(str, Enum):
    """Coarse market regimes."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"

    @property
    def risk_multiplier(self) -> float:
        """Suggested position-size multiplier for this regime."""
        return {
            Regime.TRENDING_UP: 1.0,
            Regime.TRENDING_DOWN: 0.5,
            Regime.RANGING: 0.7,
            Regime.VOLATILE: 0.4,
        }[self]


@dataclass(slots=True, frozen=True)
class RegimeSignal:
    """A regime classification with the statistics behind it."""

    regime: Regime
    trend: float  # net drift over the window (slope sign, scaled)
    volatility: float  # realised volatility of returns
    hurst: float  # rough Hurst exponent (>0.5 trending, <0.5 mean-reverting)

    @property
    def risk_multiplier(self) -> float:
        return self.regime.risk_multiplier


def _hurst(returns: np.ndarray) -> float:
    """Estimate a rough Hurst exponent via rescaled-range over the series.

    Values above ~0.5 suggest persistence (trending); below suggest
    mean-reversion. Falls back to 0.5 (random walk) on degenerate input.
    """
    n = returns.size
    if n < 8:
        return 0.5
    mean = returns.mean()
    deviations = np.cumsum(returns - mean)
    spread = float(deviations.max() - deviations.min())
    std = float(returns.std())
    if std == 0 or spread == 0:
        return 0.5
    rescaled_range = spread / std
    hurst = float(np.log(rescaled_range) / np.log(n))
    return float(np.clip(hurst, 0.0, 1.0))


@dataclass(slots=True)
class RegimeDetector:
    """Rule-based market-regime classifier over a trailing price window."""

    vol_threshold: float = 0.02  # daily-vol above this => VOLATILE
    trend_threshold: float = 0.15  # |scaled drift| above this => trending

    def detect(self, prices: np.ndarray) -> RegimeSignal:
        """Classify the supplied price window into a :class:`Regime`."""
        prices = np.asarray(prices, dtype=float)
        if prices.size < 3:
            return RegimeSignal(Regime.RANGING, 0.0, 0.0, 0.5)

        returns = np.diff(prices) / prices[:-1]
        volatility = float(returns.std())

        # Normalised drift: total return scaled by per-step volatility.
        total_return = float(prices[-1] / prices[0] - 1.0)
        scale = volatility * np.sqrt(returns.size) or 1.0
        trend = float(np.tanh(total_return / scale))

        hurst = _hurst(returns)

        if volatility > self.vol_threshold:
            regime = Regime.VOLATILE
        elif trend > self.trend_threshold:
            regime = Regime.TRENDING_UP
        elif trend < -self.trend_threshold:
            regime = Regime.TRENDING_DOWN
        else:
            regime = Regime.RANGING

        return RegimeSignal(regime=regime, trend=trend, volatility=volatility, hurst=hurst)
