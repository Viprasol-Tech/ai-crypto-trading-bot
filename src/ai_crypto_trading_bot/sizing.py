"""Kelly-style position sizing from a conviction score.

Treats the conviction as an edge proxy and applies a fractional-Kelly cap so the
bot scales in with confidence but never bets the whole bankroll. An optional
regime multiplier lets the engine scale risk down in unfavourable markets.

Part of AI Crypto Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SizingConfig:
    """Sizing parameters."""

    kelly_fraction: float = 0.5  # fraction of full Kelly to use
    max_fraction: float = 0.3  # hard cap on bankroll committed per position
    min_conviction: float = 0.1  # ignore weak signals below this |score|
    allow_short: bool = False  # if True, negative conviction sizes a short


def target_fraction(
    conviction: float,
    config: SizingConfig | None = None,
    regime_multiplier: float = 1.0,
) -> float:
    """Return the fraction of bankroll to allocate for a conviction score.

    The sign of the result indicates direction: positive = long, negative =
    short (only when ``allow_short`` is set), zero = flat. Weak signals below
    ``min_conviction`` map to flat. ``regime_multiplier`` (typically in
    ``[0, 1]``) scales the final size to reflect the current market regime.
    """
    cfg = config or SizingConfig()

    if abs(conviction) < cfg.min_conviction:
        return 0.0
    if conviction < 0 and not cfg.allow_short:
        return 0.0

    fraction = conviction * cfg.kelly_fraction * regime_multiplier
    cap = cfg.max_fraction
    if fraction >= 0:
        return min(fraction, cap)
    return max(fraction, -cap)
