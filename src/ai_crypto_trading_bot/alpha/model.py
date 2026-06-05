"""Alpha model — map features to a conviction score in [-1, 1].

``LinearAlpha`` is a transparent weighted model you can hand-tune or fit. Swap in
your own model (sklearn, XGBoost, a neural net, or an LLM sentiment score) by
implementing :class:`AlphaModel.score`.

Part of AI Crypto Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ai_crypto_trading_bot.features.engineering import Features


class AlphaModel(ABC):
    """Maps engineered features to a conviction score in [-1, 1]."""

    @abstractmethod
    def score(self, features: Features) -> float:
        """Return conviction in [-1, 1] (+ = long, - = short/flat)."""


@dataclass(slots=True)
class LinearAlpha(AlphaModel):
    """Weighted-sum alpha over the four engineered features.

    Defaults lean on momentum and trend with a smaller mean-reversion tilt and a
    volatility penalty, which is a sensible transparent baseline.
    """

    w_momentum: float = 1.0
    w_mean_reversion: float = 0.5
    w_volatility: float = -0.3
    w_trend: float = 0.8

    def score(self, features: Features) -> float:
        raw = (
            self.w_momentum * features.momentum
            + self.w_mean_reversion * features.mean_reversion
            + self.w_volatility * features.volatility
            + self.w_trend * features.trend_strength
        )
        return float(np.tanh(raw))
