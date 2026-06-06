"""Tests for features, alpha models, sizing, and the backtest pipeline."""

from __future__ import annotations

import math

import numpy as np

from ai_crypto_trading_bot.alpha.model import EnsembleAlpha, LinearAlpha, LogisticAlpha
from ai_crypto_trading_bot.backtest import run_backtest
from ai_crypto_trading_bot.config import EngineConfig
from ai_crypto_trading_bot.features.engineering import Features, compute_features
from ai_crypto_trading_bot.sizing import SizingConfig, target_fraction


def test_features_zero_without_history() -> None:
    f = compute_features(np.array([100.0, 101.0]))
    assert f == compute_features(np.array([]))
    assert f == Features.zeros()


def test_features_in_range() -> None:
    prices = np.array([100, 101, 102, 103, 104, 103, 105, 107, 108, 110], dtype=float)
    f = compute_features(prices)
    for value in f.as_array():
        assert -1.0 <= value <= 1.0


def test_features_as_array_length() -> None:
    prices = np.linspace(100, 120, 30)
    assert compute_features(prices).as_array().size == 8


def test_features_rsi_high_on_uptrend() -> None:
    up = np.array([100 + i for i in range(20)], dtype=float)
    assert compute_features(up).rsi > 0.5


def test_alpha_score_bounded() -> None:
    prices = np.array([100, 102, 104, 106, 108, 110, 112], dtype=float)
    for model in (LinearAlpha(), LogisticAlpha()):
        score = model.score(compute_features(prices))
        assert -1.0 <= score <= 1.0


def test_logistic_probability_in_unit_interval() -> None:
    prices = np.linspace(100, 130, 40)
    p = LogisticAlpha().probability(compute_features(prices))
    assert 0.0 < p < 1.0


def test_logistic_score_matches_probability() -> None:
    f = compute_features(np.linspace(100, 130, 40))
    model = LogisticAlpha()
    assert math.isclose(model.score(f), 2.0 * model.probability(f) - 1.0)


def test_ensemble_blends_members() -> None:
    f = compute_features(np.linspace(100, 130, 40))
    members = [LinearAlpha(), LogisticAlpha()]
    ens = EnsembleAlpha(models=members)
    expected = sum(m.score(f) for m in members) / len(members)
    assert math.isclose(ens.score(f), max(-1.0, min(1.0, expected)), rel_tol=1e-9)
    assert "Ensemble" in ens.name


def test_sizing_flat_below_min_conviction() -> None:
    assert target_fraction(0.05, SizingConfig(min_conviction=0.1)) == 0.0


def test_sizing_capped_at_max() -> None:
    frac = target_fraction(1.0, SizingConfig(kelly_fraction=1.0, max_fraction=0.3))
    assert math.isclose(frac, 0.3)


def test_sizing_regime_multiplier_scales_down() -> None:
    # Use a conviction low enough that the max-fraction cap does not bind.
    cfg = SizingConfig(kelly_fraction=0.5, max_fraction=0.3)
    base = target_fraction(0.4, cfg, regime_multiplier=1.0)
    scaled = target_fraction(0.4, cfg, regime_multiplier=0.5)
    assert base < cfg.max_fraction
    assert math.isclose(scaled, base * 0.5)


def test_sizing_short_disabled_by_default() -> None:
    assert target_fraction(-0.8, SizingConfig()) == 0.0


def test_sizing_short_enabled() -> None:
    frac = target_fraction(-0.8, SizingConfig(allow_short=True, max_fraction=0.3))
    assert frac < 0.0
    assert frac >= -0.3


def test_backtest_runs() -> None:
    prices = [100.0 + math.sin(i / 10.0) * 5 for i in range(120)]
    result = run_backtest(prices)
    assert len(result.equity_curve) == len(prices)
    assert isinstance(result.total_return_pct, float)


def test_backtest_records_costs_and_trades() -> None:
    prices = [100.0 + math.sin(i / 8.0) * 6 for i in range(150)]
    result = run_backtest(prices)
    assert result.total_costs >= 0.0
    assert result.num_trades >= 1
    assert len(result.regimes) == len(result.convictions)


def test_backtest_with_config_and_model() -> None:
    prices = [100.0 + i * 0.1 for i in range(80)]
    cfg = EngineConfig(window=15, starting_equity=5_000.0, use_regime=False)
    result = run_backtest(prices, model=LogisticAlpha(), config=cfg)
    assert result.starting_equity == 5_000.0
    assert result.model_name == "LogisticAlpha"
