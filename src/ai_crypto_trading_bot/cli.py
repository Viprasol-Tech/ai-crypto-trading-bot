"""CLI for AI Crypto Trading Bot.

``ai-crypto-trading-bot demo`` runs the full AI pipeline (features → alpha →
Kelly sizing → backtest) on synthetic crypto-like price data.

Part of AI Crypto Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import math

import typer
from rich.console import Console

from ai_crypto_trading_bot import __version__
from ai_crypto_trading_bot.backtest import run_backtest

app = typer.Typer(add_completion=False, help="AI Crypto Trading Bot — by Viprasol Tech.")
console = Console()


def _synthetic_prices(n: int = 400) -> list[float]:
    return [100.0 + i * 0.05 + 12.0 * math.sin(i / 15.0) for i in range(n)]


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"ai-crypto-trading-bot [bold cyan]{__version__}[/] — by Viprasol Tech")


@app.command()
def demo() -> None:
    """Run the AI alpha pipeline backtest on synthetic data."""
    result = run_backtest(_synthetic_prices())
    console.print(f"Start equity: [bold]${result.starting_equity:,.2f}[/]")
    console.print(f"Final equity: [bold]${result.final_equity:,.2f}[/]")
    console.print(f"Total return: [bold green]{result.total_return_pct:+.2f}%[/]")


if __name__ == "__main__":
    app()
