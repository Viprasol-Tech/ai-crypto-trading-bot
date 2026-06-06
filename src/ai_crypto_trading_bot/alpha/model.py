"""Alpha models — map features to a conviction score in [-1, 1].

Three transparent models ship out of the box:

* :class:`LinearAlpha` — a weighted-sum baseline you can hand-tune or fit.
* :class:`LogisticAlpha` — a logistic / sigmoid score that maps a weighted
  combination of features to a calibrated long/short probability.
* :class:`EnsembleAlpha` — a weighted blend of any number of sub-models, so you
  can combine views (e.g. a momentum model and a mean-reversion model).

Swap in your own model (sklearn, XGBoost, a neural net, or an LLM sentiment
score) by implementing :meth:`AlphaModel.score`.

Part of AI Crypto Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ai_crypto_trading_bot.features.engineering import Features


class AlphaModel(ABC):
    """Maps engineered features to a conviction score in [-1, 1]."""

    @abstractmethod
    def score(self, features: Features) -> float:
        """Return conviction in [-1, 1] (+ = long, - = short/flat)."""

    @property
    def name(self) -> str:
        """Human-readable model name (defaults to the class name)."""
        return type(self).__name__


@dataclass(slots=True)
class LinearAlpha(AlphaModel):
    """Weighted-sum alpha over the engineered features.

    Defaults lean on momentum and trend with a smaller mean-reversion tilt and a
    volatility penalty, which is a sensible transparent baseline. The v0.2
    features (RSI, MACD, Bollinger, acceleration) get smaller default weights.
    """

    w_momentum: float = 1.0
    w_mean_reversion: float = 0.5
    w_volatility: float = -0.3
    w_trend: float = 0.8
    w_rsi: float = -0.2
    w_macd: float = 0.4
    w_bollinger: float = -0.2
    w_acceleration: float = 0.3

    def score(self, features: Features) -> float:
        raw = (
            self.w_momentum * features.momentum
            + self.w_mean_reversion * features.mean_reversion
            + self.w_volatility * features.volatility
            + self.w_trend * features.trend_strength
            + self.w_rsi * features.rsi
            + self.w_macd * features.macd
            + self.w_bollinger * features.bollinger
            + self.w_acceleration * features.acceleration
        )
        return float(np.tanh(raw))


@dataclass(slots=True)
class LogisticAlpha(AlphaModel):
    """Logistic / sigmoid alpha producing a calibrated conviction.

    A weighted combination of features is passed through a logistic function to
    yield a long probability ``p`` in ``(0, 1)``; the conviction is ``2p - 1`` so
    the output lives in ``(-1, 1)`` like every other model. ``temperature``
    controls how sharply probabilities saturate near the extremes.

    This is the natural drop-in for a fitted classifier: set ``weights`` to your
    learned coefficients and ``bias`` to the intercept.
    """

    weights: Sequence[float] = field(
        default_factory=lambda: [1.2, 0.4, -0.4, 1.0, -0.3, 0.6, -0.3, 0.4]
    )
    bias: float = 0.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        self._w = np.asarray(self.weights, dtype=float)

    def probability(self, features: Features) -> float:
        """Return the model's long probability in ``(0, 1)``."""
        x = features.as_array()
        if x.size != self._w.size:
            raise ValueError(
                f"expected {self._w.size} features, got {x.size}; weights and Features must align"
            )
        logit = float(self._w @ x + self.bias) / self.temperature
        return float(1.0 / (1.0 + np.exp(-logit)))

    def score(self, features: Features) -> float:
        return 2.0 * self.probability(features) - 1.0


@dataclass(slots=True)
class EnsembleAlpha(AlphaModel):
    """Weighted blend of sub-models.

    The conviction is the weighted average of the members' scores, clipped to
    ``[-1, 1]``. Equal weights are used when none are supplied. This lets you
    combine, say, a trend model and a mean-reversion model and let the ensemble
    average out their disagreements.
    """

    models: Sequence[AlphaModel]
    weights: Sequence[float] | None = None

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("EnsembleAlpha needs at least one model")
        if self.weights is None:
            self._w = np.full(len(self.models), 1.0 / len(self.models))
        else:
            w = np.asarray(self.weights, dtype=float)
            if w.size != len(self.models):
                raise ValueError("weights length must match number of models")
            total = float(w.sum())
            if total == 0:
                raise ValueError("ensemble weights must not sum to zero")
            self._w = w / total

    def score(self, features: Features) -> float:
        scores = np.array([m.score(features) for m in self.models], dtype=float)
        blended = float(self._w @ scores)
        return float(np.clip(blended, -1.0, 1.0))

    @property
    def name(self) -> str:
        members = ", ".join(m.name for m in self.models)
        return f"Ensemble[{members}]"
