"""Kelly-style position sizing from a conviction score.

Treats the conviction as an edge proxy and applies a fractional-Kelly cap so the
bot scales in with confidence but never bets the whole bankroll.

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


def target_fraction(conviction: float, config: SizingConfig | None = None) -> float:
    """Return the fraction of bankroll to allocate long for a conviction score.

    Negative or weak convictions map to 0 (flat) — this baseline is long/flat.
    """
    cfg = config or SizingConfig()
    if conviction < cfg.min_conviction:
        return 0.0
    fraction = conviction * cfg.kelly_fraction
    return min(fraction, cfg.max_fraction)
