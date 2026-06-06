"""Tests for performance metrics."""

from __future__ import annotations

import math

import numpy as np

from ai_crypto_trading_bot.metrics import compute_metrics, max_drawdown


def test_max_drawdown_simple() -> None:
    # Peak 120, trough 90 -> drawdown 25%.
    curve = [100, 120, 90, 110]
    assert math.isclose(max_drawdown(curve), 0.25)


def test_max_drawdown_monotonic_is_zero() -> None:
    assert max_drawdown([100, 101, 102, 103]) == 0.0


def test_max_drawdown_empty() -> None:
    assert max_drawdown([]) == 0.0


def test_metrics_positive_return() -> None:
    curve = list(np.linspace(10_000, 12_000, 50))
    m = compute_metrics(curve)
    assert m.total_return_pct > 0
    assert m.num_periods == 50
    assert m.max_drawdown_pct == 0.0


def test_metrics_degenerate_curve() -> None:
    m = compute_metrics([10_000.0])
    assert m.total_return_pct == 0.0
    assert m.sharpe == 0.0


def test_metrics_as_dict_keys() -> None:
    m = compute_metrics(list(np.linspace(100, 110, 20)))
    d = m.as_dict()
    assert "sharpe" in d and "max_drawdown_pct" in d and "calmar" in d


def test_sharpe_zero_when_flat() -> None:
    m = compute_metrics([100.0] * 30)
    assert m.sharpe == 0.0
    assert m.annualised_volatility_pct == 0.0


def test_hit_rate_bounds() -> None:
    rng = np.random.default_rng(0)
    curve = (10_000 * np.cumprod(1 + rng.normal(0.001, 0.01, 200))).tolist()
    m = compute_metrics(curve)
    assert 0.0 <= m.hit_rate_pct <= 100.0
