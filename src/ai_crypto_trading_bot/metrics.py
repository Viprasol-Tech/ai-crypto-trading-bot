"""Performance metrics for an equity curve.

Turns a list of equity values into the risk/return statistics a quant actually
looks at: total/annualised return, volatility, Sharpe, Sortino, Calmar, maximum
drawdown, and hit rate. All functions are pure and operate on plain sequences.

Part of AI Crypto Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Trading periods per year. Crypto trades 24/7, so the default assumes daily
# bars across 365 days; override for intraday data.
PERIODS_PER_YEAR = 365.0


@dataclass(slots=True, frozen=True)
class PerformanceMetrics:
    """A bundle of risk/return statistics for an equity curve."""

    total_return_pct: float
    annualised_return_pct: float
    annualised_volatility_pct: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float
    hit_rate_pct: float
    num_periods: int

    def as_dict(self) -> dict[str, float | int]:
        """Return the metrics as a plain dict (handy for JSON/report output)."""
        return {
            "total_return_pct": self.total_return_pct,
            "annualised_return_pct": self.annualised_return_pct,
            "annualised_volatility_pct": self.annualised_volatility_pct,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown_pct": self.max_drawdown_pct,
            "hit_rate_pct": self.hit_rate_pct,
            "num_periods": self.num_periods,
        }


def max_drawdown(equity_curve: Sequence[float] | np.ndarray) -> float:
    """Return the maximum peak-to-trough drawdown as a fraction in ``[0, 1]``."""
    arr = np.asarray(equity_curve, dtype=float)
    if arr.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(arr)
    # Guard against a zero peak (e.g. equity wiped out at the start).
    safe_peak = np.where(running_peak == 0, 1.0, running_peak)
    drawdowns = (running_peak - arr) / safe_peak
    return float(drawdowns.max())


def compute_metrics(
    equity_curve: Sequence[float],
    periods_per_year: float = PERIODS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Compute a full :class:`PerformanceMetrics` bundle from an equity curve."""
    arr = np.asarray(equity_curve, dtype=float)
    if arr.size < 2 or arr[0] == 0:
        return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, int(arr.size))

    returns = np.diff(arr) / arr[:-1]
    total_return = float(arr[-1] / arr[0] - 1.0)

    mean_r = float(returns.mean())
    std_r = float(returns.std())
    downside = returns[returns < 0]
    downside_std = float(downside.std()) if downside.size else 0.0

    ann_factor = np.sqrt(periods_per_year)
    rf_per_period = risk_free_rate / periods_per_year
    excess = mean_r - rf_per_period

    sharpe = float(excess / std_r * ann_factor) if std_r > 0 else 0.0
    sortino = float(excess / downside_std * ann_factor) if downside_std > 0 else 0.0

    annualised_return = float((1.0 + mean_r) ** periods_per_year - 1.0)
    annualised_vol = std_r * ann_factor

    mdd = max_drawdown(arr)
    calmar = float(annualised_return / mdd) if mdd > 0 else 0.0

    nonzero = returns[returns != 0]
    hit_rate = float((nonzero > 0).mean()) if nonzero.size else 0.0

    return PerformanceMetrics(
        total_return_pct=total_return * 100.0,
        annualised_return_pct=annualised_return * 100.0,
        annualised_volatility_pct=annualised_vol * 100.0,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown_pct=mdd * 100.0,
        hit_rate_pct=hit_rate * 100.0,
        num_periods=int(arr.size),
    )
