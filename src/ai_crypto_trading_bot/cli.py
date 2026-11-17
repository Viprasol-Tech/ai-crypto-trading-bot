"""CLI for AI Crypto Trading Bot.

Subcommands:

* ``demo``     - run the full AI pipeline on synthetic data and print a report.
* ``backtest`` - run a backtest with a chosen model and print a full report.
* ``regime``   - classify the current synthetic window into a market regime.
* ``models``   - list the bundled alpha models.
* ``version``  - print the installed version.

Part of AI Crypto Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
import typer
from rich.console import Console

from ai_crypto_trading_bot import __version__
from ai_crypto_trading_bot.alpha.model import EnsembleAlpha, LinearAlpha, LogisticAlpha
from ai_crypto_trading_bot.backtest import run_backtest
from ai_crypto_trading_bot.config import EngineConfig
from ai_crypto_trading_bot.regime import RegimeDetector
from ai_crypto_trading_bot.report import print_report, render_text

app = typer.Typer(add_completion=False, help="AI Crypto Trading Bot - by Viprasol Tech.")
console = Console()


class ModelChoice(str, Enum):
    """Selectable bundled alpha models."""

    linear = "linear"
    logistic = "logistic"
    ensemble = "ensemble"


def _build_model(choice: ModelChoice) -> LinearAlpha | LogisticAlpha | EnsembleAlpha:
    if choice is ModelChoice.linear:
        return LinearAlpha()
    if choice is ModelChoice.logistic:
        return LogisticAlpha()
    return EnsembleAlpha(models=[LinearAlpha(), LogisticAlpha()])


def _synthetic_prices(n: int = 400, seed: int = 7) -> list[float]:
    """Trending sine wave plus reproducible noise (crypto-like)."""
    rng = np.random.default_rng(seed)
    noise = np.cumsum(rng.normal(0.0, 0.6, size=n))
    return [100.0 + i * 0.05 + 12.0 * math.sin(i / 15.0) + noise[i] for i in range(n)]


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"ai-crypto-trading-bot [bold cyan]{__version__}[/] - by Viprasol Tech")


@app.command()
def models() -> None:
    """List the bundled alpha models."""
    for choice in ModelChoice:
        console.print(f"- [bold]{choice.value}[/]: {_build_model(choice).name}")


@app.command()
def demo() -> None:
    """Run the AI alpha pipeline backtest on synthetic data."""
    result = run_backtest(_synthetic_prices(), model=LinearAlpha())
    print_report(result, console=console)


@app.command()
def backtest(
    model: ModelChoice = typer.Option(ModelChoice.linear, help="Alpha model to use."),
    bars: int = typer.Option(400, min=10, help="Number of synthetic bars."),
    window: int = typer.Option(20, min=3, help="Trailing feature window."),
    equity: float = typer.Option(10_000.0, min=1.0, help="Starting equity."),
    no_regime: bool = typer.Option(False, help="Disable regime-based sizing."),
    plain: bool = typer.Option(False, help="Print ASCII report instead of a table."),
) -> None:
    """Backtest a chosen model and print a full performance report."""
    cfg = EngineConfig(window=window, starting_equity=equity, use_regime=not no_regime)
    result = run_backtest(_synthetic_prices(bars), model=_build_model(model), config=cfg)
    if plain:
        console.print(render_text(result, periods_per_year=cfg.periods_per_year))
    else:
        print_report(result, console=console, periods_per_year=cfg.periods_per_year)


@app.command()
def regime(bars: int = typer.Option(60, min=3, help="Synthetic bars to classify.")) -> None:
    """Detect the market regime of a synthetic price window."""
    prices = np.asarray(_synthetic_prices(bars))
    signal = RegimeDetector().detect(prices)
    console.print(f"Regime         : [bold magenta]{signal.regime.value}[/]")
    console.print(f"Trend          : {signal.trend:+.3f}")
    console.print(f"Volatility     : {signal.volatility:.4f}")
    console.print(f"Hurst exponent : {signal.hurst:.3f}")
    console.print(f"Risk multiplier: {signal.risk_multiplier:.2f}")


if __name__ == "__main__":
    app()
