"""Typed configuration for the trading pipeline (pydantic v2).

A single validated :class:`EngineConfig` wires together the backtest window,
sizing, regime detection, and trading costs. Validation catches bad inputs (a
negative window, fees over 100%) before they silently corrupt a backtest.

Part of AI Crypto Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ai_crypto_trading_bot.sizing import SizingConfig


class CostModel(BaseModel):
    """Trading-cost assumptions applied on every rebalance."""

    model_config = {"frozen": True}

    fee_bps: float = Field(default=10.0, ge=0.0, le=10_000.0)
    """Taker fee in basis points charged on traded notional (10 bps = 0.10%)."""

    slippage_bps: float = Field(default=5.0, ge=0.0, le=10_000.0)
    """Estimated slippage in basis points on traded notional."""

    @property
    def cost_rate(self) -> float:
        """Total round-trip cost rate as a fraction of traded notional."""
        return (self.fee_bps + self.slippage_bps) / 10_000.0


class EngineConfig(BaseModel):
    """Top-level configuration for a backtest / live engine run."""

    model_config = {"arbitrary_types_allowed": True}

    window: int = Field(default=20, ge=3, le=2_000)
    """Trailing window (in bars) used to build features and detect regime."""

    starting_equity: float = Field(default=10_000.0, gt=0.0)
    """Starting bankroll for the backtest."""

    sizing: SizingConfig = Field(default_factory=SizingConfig)
    """Position-sizing parameters."""

    costs: CostModel = Field(default_factory=CostModel)
    """Trading-cost model applied on each rebalance."""

    use_regime: bool = True
    """Whether to scale sizing by the detected market regime."""

    periods_per_year: float = Field(default=365.0, gt=0.0)
    """Annualisation factor for metrics (365 for daily crypto bars)."""
