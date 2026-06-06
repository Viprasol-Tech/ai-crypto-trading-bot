"""Tests for market-regime detection."""

from __future__ import annotations

import numpy as np

from ai_crypto_trading_bot.regime import Regime, RegimeDetector


def test_uptrend_detected() -> None:
    prices = np.array([100 + i * 0.5 for i in range(60)], dtype=float)
    signal = RegimeDetector().detect(prices)
    assert signal.regime is Regime.TRENDING_UP
    assert signal.trend > 0


def test_downtrend_detected() -> None:
    prices = np.array([200 - i * 0.5 for i in range(60)], dtype=float)
    signal = RegimeDetector().detect(prices)
    assert signal.regime is Regime.TRENDING_DOWN
    assert signal.trend < 0


def test_volatile_regime() -> None:
    rng = np.random.default_rng(1)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.05, 80))
    signal = RegimeDetector(vol_threshold=0.02).detect(prices)
    assert signal.regime is Regime.VOLATILE


def test_ranging_regime() -> None:
    prices = np.array([100 + (0.2 if i % 2 else -0.2) for i in range(60)], dtype=float)
    signal = RegimeDetector().detect(prices)
    assert signal.regime is Regime.RANGING


def test_short_window_defaults_to_ranging() -> None:
    signal = RegimeDetector().detect(np.array([100.0, 101.0]))
    assert signal.regime is Regime.RANGING
    assert signal.hurst == 0.5


def test_risk_multiplier_ordering() -> None:
    assert Regime.TRENDING_UP.risk_multiplier > Regime.VOLATILE.risk_multiplier
    assert Regime.RANGING.risk_multiplier <= Regime.TRENDING_UP.risk_multiplier


def test_hurst_in_unit_interval() -> None:
    prices = np.array([100 + i * 0.5 for i in range(60)], dtype=float)
    signal = RegimeDetector().detect(prices)
    assert 0.0 <= signal.hurst <= 1.0
