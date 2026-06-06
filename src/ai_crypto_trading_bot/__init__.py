"""AI Crypto Trading Bot - AI alpha engine for crypto by Viprasol Tech."""

from __future__ import annotations

from ai_crypto_trading_bot.alpha.model import (
    AlphaModel,
    EnsembleAlpha,
    LinearAlpha,
    LogisticAlpha,
)
from ai_crypto_trading_bot.backtest import BacktestResult, run_backtest
from ai_crypto_trading_bot.config import CostModel, EngineConfig
from ai_crypto_trading_bot.features.engineering import Features, compute_features
from ai_crypto_trading_bot.metrics import PerformanceMetrics, compute_metrics
from ai_crypto_trading_bot.regime import Regime, RegimeDetector, RegimeSignal
from ai_crypto_trading_bot.sizing import SizingConfig, target_fraction

__version__ = "0.2.0"
__author__ = "Viprasol Tech Private Limited"
__all__ = [
    "AlphaModel",
    "BacktestResult",
    "CostModel",
    "EngineConfig",
    "EnsembleAlpha",
    "Features",
    "LinearAlpha",
    "LogisticAlpha",
    "PerformanceMetrics",
    "Regime",
    "RegimeDetector",
    "RegimeSignal",
    "SizingConfig",
    "__version__",
    "compute_features",
    "compute_metrics",
    "run_backtest",
    "target_fraction",
]
