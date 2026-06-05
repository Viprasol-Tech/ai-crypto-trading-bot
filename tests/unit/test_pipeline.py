"""Tests for features, alpha model, sizing, and the backtest pipeline."""

from __future__ import annotations

import math

import numpy as np

from ai_crypto_trading_bot.alpha.model import LinearAlpha
from ai_crypto_trading_bot.backtest import run_backtest
from ai_crypto_trading_bot.features.engineering import compute_features
from ai_crypto_trading_bot.sizing import SizingConfig, target_fraction


def test_features_zero_without_history() -> None:
    f = compute_features(np.array([100.0, 101.0]))
    assert f == compute_features(np.array([]))


def test_features_in_range() -> None:
    prices = np.array([100, 101, 102, 103, 104, 103, 105, 107, 108, 110], dtype=float)
    f = compute_features(prices)
    for value in (f.momentum, f.mean_reversion, f.volatility, f.trend_strength):
        assert -1.0 <= value <= 1.0


def test_alpha_score_bounded() -> None:
    prices = np.array([100, 102, 104, 106, 108, 110, 112], dtype=float)
    score = LinearAlpha().score(compute_features(prices))
    assert -1.0 <= score <= 1.0


def test_sizing_flat_below_min_conviction() -> None:
    assert target_fraction(0.05, SizingConfig(min_conviction=0.1)) == 0.0


def test_sizing_capped_at_max() -> None:
    frac = target_fraction(1.0, SizingConfig(kelly_fraction=1.0, max_fraction=0.3))
    assert math.isclose(frac, 0.3)


def test_backtest_runs() -> None:
    prices = [100.0 + math.sin(i / 10.0) * 5 for i in range(120)]
    result = run_backtest(prices)
    assert len(result.equity_curve) == len(prices)
    assert isinstance(result.total_return_pct, float)
