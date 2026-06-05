"""Backtest the AI pipeline: features → alpha → sizing → equity curve.

Each step the bot rebuilds features from a trailing window, scores them, computes
a target bankroll fraction, and rebalances a single long/flat crypto position.

Part of AI Crypto Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ai_crypto_trading_bot.alpha.model import AlphaModel, LinearAlpha
from ai_crypto_trading_bot.features.engineering import compute_features
from ai_crypto_trading_bot.sizing import SizingConfig, target_fraction


@dataclass(slots=True)
class BacktestResult:
    """Summary of an AI backtest."""

    starting_equity: float
    final_equity: float
    equity_curve: list[float] = field(default_factory=list)

    @property
    def total_return_pct(self) -> float:
        if self.starting_equity == 0:
            return 0.0
        return (self.final_equity / self.starting_equity - 1.0) * 100.0


def run_backtest(
    prices: Sequence[float],
    model: AlphaModel | None = None,
    window: int = 20,
    starting_equity: float = 10_000.0,
    sizing: SizingConfig | None = None,
) -> BacktestResult:
    """Run the AI pipeline over a price series and return the equity curve."""
    alpha = model or LinearAlpha()
    arr = np.asarray(prices, dtype=float)
    cash = starting_equity
    units = 0.0
    result = BacktestResult(starting_equity=starting_equity, final_equity=starting_equity)

    for i in range(arr.size):
        price = float(arr[i])
        if i >= 2:
            features = compute_features(arr[max(0, i - window + 1) : i + 1])
            conviction = alpha.score(features)
            equity = cash + units * price
            target_value = equity * target_fraction(conviction, sizing)
            target_units = target_value / price if price > 0 else 0.0
            delta = target_units - units
            cash -= delta * price
            units = target_units
        result.equity_curve.append(cash + units * price)

    result.final_equity = result.equity_curve[-1] if result.equity_curve else starting_equity
    return result
