# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[SemVer](https://semver.org/).

## [0.2.0] - 2025

### Added
- **Richer feature engineering** — added Wilder RSI, MACD histogram, Bollinger-band
  position, and momentum acceleration alongside the original four features, all
  bounded for direct model use. New `Features.as_array()` / `Features.zeros()` helpers.
- **New alpha models** — `LogisticAlpha` (calibrated sigmoid / logistic score with a
  drop-in `weights`/`bias` for fitted classifiers) and `EnsembleAlpha` (weighted blend
  of any number of sub-models).
- **Market-regime detection** — `RegimeDetector` classifies a window into
  trending-up / trending-down / ranging / volatile using drift, realised volatility,
  and a rescaled-range Hurst estimate, each regime carrying a risk multiplier.
- **Regime-aware sizing** — `target_fraction` now accepts a regime multiplier and
  optional short side (`allow_short`).
- **Trading costs** — `CostModel` applies fee + slippage (bps) on every rebalance.
- **Performance metrics** — `compute_metrics` returns total/annualised return,
  volatility, Sharpe, Sortino, Calmar, max drawdown, and hit rate.
- **Backtest report** — `report.py` renders an ASCII report and a Rich table; the
  backtest now tracks convictions, regimes, trade count, and total costs.
- **Typed config** — pydantic `EngineConfig` wires window, sizing, costs, and regime
  toggles together with validation.
- **CLI subcommands** — `backtest`, `regime`, and `models` join `demo` and `version`.
- **Example** — `examples/run_pipeline.py` runs an ensemble end to end.
- Test suite expanded from 6 to 40 tests covering features, models, regime, metrics,
  config, report rendering, and the CLI.

### Changed
- Bumped version to 0.2.0.
- `run_backtest` accepts a full `EngineConfig` (the `window`/`starting_equity`
  arguments remain as backward-compatible overrides).

## [0.1.0] - 2025

### Added
- Initial release of ai-crypto-trading-bot: AI-driven crypto trading bot: feature engineering, alpha scoring, Kelly sizing.
