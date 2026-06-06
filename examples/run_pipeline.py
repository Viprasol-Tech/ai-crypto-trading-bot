"""End-to-end example: build features, blend two alpha models, backtest, report.

Run from the repo root:

    python examples/run_pipeline.py

It generates a synthetic crypto-like price series, runs an ensemble of the
linear and logistic alpha models with regime-aware Kelly sizing and trading
costs, then prints a full performance report.

Part of AI Crypto Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import math

import numpy as np

from ai_crypto_trading_bot.alpha.model import EnsembleAlpha, LinearAlpha, LogisticAlpha
from ai_crypto_trading_bot.backtest import run_backtest
from ai_crypto_trading_bot.config import CostModel, EngineConfig
from ai_crypto_trading_bot.regime import RegimeDetector
from ai_crypto_trading_bot.report import render_text


def synthetic_prices(n: int = 500, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.7, size=n).cumsum()
    return [100.0 + i * 0.04 + 14.0 * math.sin(i / 18.0) + noise[i] for i in range(n)]


def main() -> None:
    prices = synthetic_prices()

    # Inspect the current market regime.
    signal = RegimeDetector().detect(np.asarray(prices[-60:]))
    print(f"Detected regime: {signal.regime.value} (risk x{signal.risk_multiplier:.2f})")

    # Blend a transparent linear model with a logistic classifier.
    model = EnsembleAlpha(models=[LinearAlpha(), LogisticAlpha()], weights=[0.6, 0.4])

    config = EngineConfig(
        window=24,
        starting_equity=10_000.0,
        costs=CostModel(fee_bps=10.0, slippage_bps=5.0),
        use_regime=True,
    )

    result = run_backtest(prices, model=model, config=config)
    print(render_text(result, periods_per_year=config.periods_per_year))


if __name__ == "__main__":
    main()
