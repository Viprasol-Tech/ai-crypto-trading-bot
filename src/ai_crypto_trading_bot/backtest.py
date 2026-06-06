"""Backtest the AI pipeline: features -> alpha -> regime -> sizing -> equity.

Each bar the engine rebuilds features from a trailing window, scores them,
detects the market regime, computes a target bankroll fraction, applies trading
costs, and rebalances a single long/flat (optionally short) crypto position.

Part of AI Crypto Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ai_crypto_trading_bot.alpha.model import AlphaModel, LinearAlpha
from ai_crypto_trading_bot.config import EngineConfig
from ai_crypto_trading_bot.features.engineering import MIN_HISTORY, compute_features
from ai_crypto_trading_bot.metrics import PerformanceMetrics, compute_metrics
from ai_crypto_trading_bot.regime import Regime, RegimeDetector
from ai_crypto_trading_bot.sizing import target_fraction


@dataclass(slots=True)
class BacktestResult:
    """Summary of an AI backtest, including a full metrics bundle."""

    starting_equity: float
    final_equity: float
    equity_curve: list[float] = field(default_factory=list)
    convictions: list[float] = field(default_factory=list)
    regimes: list[Regime] = field(default_factory=list)
    total_costs: float = 0.0
    num_trades: int = 0
    model_name: str = "LinearAlpha"

    @property
    def total_return_pct(self) -> float:
        if self.starting_equity == 0:
            return 0.0
        return (self.final_equity / self.starting_equity - 1.0) * 100.0

    def metrics(self, periods_per_year: float = 365.0) -> PerformanceMetrics:
        """Compute the full performance-metrics bundle for this run."""
        return compute_metrics(self.equity_curve, periods_per_year=periods_per_year)


def run_backtest(
    prices: Sequence[float],
    model: AlphaModel | None = None,
    config: EngineConfig | None = None,
    window: int | None = None,
    starting_equity: float | None = None,
) -> BacktestResult:
    """Run the AI pipeline over a price series and return the result.

    ``window`` and ``starting_equity`` are convenience overrides kept for
    backward compatibility; prefer passing a full :class:`EngineConfig`.
    """
    cfg = config or EngineConfig()
    if window is not None:
        cfg = cfg.model_copy(update={"window": window})
    if starting_equity is not None:
        cfg = cfg.model_copy(update={"starting_equity": starting_equity})

    alpha = model or LinearAlpha()
    detector = RegimeDetector()
    arr = np.asarray(prices, dtype=float)

    cash = cfg.starting_equity
    units = 0.0
    cost_rate = cfg.costs.cost_rate
    result = BacktestResult(
        starting_equity=cfg.starting_equity,
        final_equity=cfg.starting_equity,
        model_name=alpha.name,
    )

    for i in range(arr.size):
        price = float(arr[i])
        if i >= MIN_HISTORY - 1:
            window_slice = arr[max(0, i - cfg.window + 1) : i + 1]
            features = compute_features(window_slice)
            conviction = alpha.score(features)

            multiplier = 1.0
            regime = Regime.RANGING
            if cfg.use_regime:
                signal = detector.detect(window_slice)
                regime = signal.regime
                multiplier = signal.risk_multiplier

            equity = cash + units * price
            target_value = equity * target_fraction(conviction, cfg.sizing, multiplier)
            target_units = target_value / price if price > 0 else 0.0
            delta = target_units - units

            if abs(delta) > 1e-12:
                traded_notional = abs(delta) * price
                cost = traded_notional * cost_rate
                cash -= delta * price + cost
                units = target_units
                result.total_costs += cost
                result.num_trades += 1

            result.convictions.append(conviction)
            result.regimes.append(regime)

        result.equity_curve.append(cash + units * price)

    result.final_equity = result.equity_curve[-1] if result.equity_curve else cfg.starting_equity
    return result
