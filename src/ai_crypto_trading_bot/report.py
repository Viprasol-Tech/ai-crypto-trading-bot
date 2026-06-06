"""Render a backtest result as a rich, human-readable report.

Provides both a plain-text report (ASCII, safe for any terminal or CI log) and a
Rich table for pretty CLI output. Keeping rendering separate from the backtest
keeps the engine import-light and easy to test.

Part of AI Crypto Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections import Counter

from rich.console import Console
from rich.table import Table

from ai_crypto_trading_bot.backtest import BacktestResult


def _regime_breakdown(result: BacktestResult) -> str:
    if not result.regimes:
        return "n/a"
    counts = Counter(r.value for r in result.regimes)
    total = sum(counts.values())
    parts = [f"{name}={100.0 * n / total:.0f}%" for name, n in counts.most_common()]
    return ", ".join(parts)


def render_text(result: BacktestResult, periods_per_year: float = 365.0) -> str:
    """Return an ASCII-only multi-line performance report."""
    m = result.metrics(periods_per_year=periods_per_year)
    lines = [
        "=" * 52,
        " AI Crypto Trading Bot - Backtest Report",
        "=" * 52,
        f" Model              : {result.model_name}",
        f" Bars               : {m.num_periods}",
        f" Trades             : {result.num_trades}",
        f" Start equity       : ${result.starting_equity:,.2f}",
        f" Final equity       : ${result.final_equity:,.2f}",
        f" Total costs        : ${result.total_costs:,.2f}",
        "-" * 52,
        f" Total return       : {m.total_return_pct:+.2f}%",
        f" Annualised return  : {m.annualised_return_pct:+.2f}%",
        f" Annualised vol     : {m.annualised_volatility_pct:.2f}%",
        f" Sharpe ratio       : {m.sharpe:.2f}",
        f" Sortino ratio      : {m.sortino:.2f}",
        f" Calmar ratio       : {m.calmar:.2f}",
        f" Max drawdown       : {m.max_drawdown_pct:.2f}%",
        f" Hit rate           : {m.hit_rate_pct:.1f}%",
        "-" * 52,
        f" Regime mix         : {_regime_breakdown(result)}",
        "=" * 52,
    ]
    return "\n".join(lines)


def render_table(result: BacktestResult, periods_per_year: float = 365.0) -> Table:
    """Return a Rich :class:`~rich.table.Table` summarising the result."""
    m = result.metrics(periods_per_year=periods_per_year)
    table = Table(title=f"Backtest Report - {result.model_name}", title_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Bars", str(m.num_periods))
    table.add_row("Trades", str(result.num_trades))
    table.add_row("Start equity", f"${result.starting_equity:,.2f}")
    table.add_row("Final equity", f"${result.final_equity:,.2f}")
    table.add_row("Total costs", f"${result.total_costs:,.2f}")
    table.add_row("Total return", f"{m.total_return_pct:+.2f}%")
    table.add_row("Annualised return", f"{m.annualised_return_pct:+.2f}%")
    table.add_row("Annualised vol", f"{m.annualised_volatility_pct:.2f}%")
    table.add_row("Sharpe", f"{m.sharpe:.2f}")
    table.add_row("Sortino", f"{m.sortino:.2f}")
    table.add_row("Calmar", f"{m.calmar:.2f}")
    table.add_row("Max drawdown", f"{m.max_drawdown_pct:.2f}%")
    table.add_row("Hit rate", f"{m.hit_rate_pct:.1f}%")
    table.add_row("Regime mix", _regime_breakdown(result))
    return table


def print_report(
    result: BacktestResult,
    console: Console | None = None,
    periods_per_year: float = 365.0,
) -> None:
    """Print the Rich report to the console (creates one if not supplied)."""
    (console or Console()).print(render_table(result, periods_per_year=periods_per_year))
