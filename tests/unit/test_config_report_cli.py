"""Tests for config validation, report rendering, and the CLI."""

from __future__ import annotations

import math

import pytest
from typer.testing import CliRunner

from ai_crypto_trading_bot.backtest import run_backtest
from ai_crypto_trading_bot.cli import app
from ai_crypto_trading_bot.config import CostModel, EngineConfig
from ai_crypto_trading_bot.report import render_text

runner = CliRunner()


def test_cost_model_rate() -> None:
    cm = CostModel(fee_bps=10.0, slippage_bps=5.0)
    assert math.isclose(cm.cost_rate, 0.0015)


def test_engine_config_rejects_bad_window() -> None:
    with pytest.raises(ValueError):
        EngineConfig(window=1)


def test_engine_config_rejects_negative_equity() -> None:
    with pytest.raises(ValueError):
        EngineConfig(starting_equity=-100.0)


def test_render_text_is_ascii() -> None:
    prices = [100.0 + math.sin(i / 9.0) * 5 for i in range(120)]
    text = render_text(run_backtest(prices))
    text.encode("ascii")  # raises if any non-ASCII slipped in
    assert "Backtest Report" in text
    assert "Sharpe" in text


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "ai-crypto-trading-bot" in result.stdout


def test_cli_demo() -> None:
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0
    assert "Backtest Report" in result.stdout


def test_cli_models_lists_all() -> None:
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    for name in ("linear", "logistic", "ensemble"):
        assert name in result.stdout


def test_cli_backtest_plain() -> None:
    result = runner.invoke(app, ["backtest", "--model", "ensemble", "--plain", "--bars", "120"])
    assert result.exit_code == 0
    assert "Sharpe" in result.stdout


def test_cli_regime() -> None:
    result = runner.invoke(app, ["regime"])
    assert result.exit_code == 0
    assert "Regime" in result.stdout
