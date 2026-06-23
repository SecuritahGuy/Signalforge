# Changelog

All notable changes to SignalForge are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- ROADMAP.md with phased plan (foundation, gaps, hardening, research depth, live-adjacent)
- SQLite metadata store (`src/signalforge/db.py`) with paper ledger, run history, and account snapshots
- `scripts/migrate_to_db.py` — one-time migration from CSV artifacts to SQLite
- CHANGELOG.md
- model_type support for `lgbm` and `xgboost` in modeling pipeline
- Rebalance exit rule in paper.py (replaces prior placeholder)
- Inverse-volatility position sizing method in portfolio_backtest.py
- GitHub Actions CI workflow (`.github/workflows/ci.yml`)
- `RebalanceConfig` fields `min_days_held` and `exit_below_score`
- `learning_rate`, `subsample`, `colsample_bytree`, `num_leaves`, `min_child_samples`, `min_child_weight` hyperparameters to `BaselineModelConfig`
- Tests for LightGBM, XGBoost, rebalance exit rule, and inverse-volatility sizing
- Model ensemble module (`src/signalforge/ensemble.py`): average/weighted/meta blending with walk-forward validation
- **Multi-timeframe features** (`src/signalforge/features.py`): extended windows from (5, 20, 60) to (5, 10, 20, 40, 60, 120) for returns/volatility/momentum; added SMA (10, 50, 200), range (10, 60, 120), volume/dollar-volume (10, 40, 120) windows; beta/correlation at 60d and 120d; all sector-relative ranks, sector returns, and market-relative features extended automatically; all window tuples are now configurable function parameters
- **Time-based Decay exit rule** (`src/signalforge/paper.py`): exits when `days_held >= half_life_days * log2(entry_score / min_score_for_decay)`, giving higher-conviction entries longer runway before mandatory exit
- `TimeDecayConfig` dataclass with `half_life_days`, `min_days_hold`, `min_score_for_decay`
- **Trailing Volatility Stop exit rule** (`src/signalforge/paper.py`): adaptive trail distance based on rolling daily volatility, clamped by configurable tightest/widest bounds, plumbed through full exit lifecycle
- `TrailingVolatilityStopConfig` dataclass with `tightest_trail_pct`, `widest_trail_pct`, `volatility_lookback`, `volatility_multiple`
- `_volatility_lookup` helper: precomputes rolling daily std per symbol
- `EXIT_RULE_VERSION` bumped to `exit_rules.v2`
- **Sector-stop exit rule** (`src/signalforge/paper.py`): exits when rolling mean return of the position's sector falls below `sector_decline_pct`
- `SectorStopConfig` dataclass with `sector_decline_pct`, `lookback_days`, `min_sector_records`
- `_sector_lookup` helper: precomputes rolling mean sector return per sector/date with `min_records` guard against thin-sector false triggers
- `config/paper.yaml` — added `sector_stop` section (disabled by default)
- `scripts/update_paper_ledger.py` — parses `sector_stop` fields from YAML

### Changed

- `paper.py: reconcile_exits` — accepts optional `universe: pd.DataFrame | None = None` for symbol→sector mapping; passes `min_records` to `_sector_lookup`
- `paper.py: _first_exit_decision` — accepts optional `sector_lookup` dict and looks up sector return for each position/date
- `paper.py: _exit_decision_for_date` — accepts `current_sector_return` float and evaluates sector-stop before score deterioration
- `features.py`: all three feature functions accept configurable `return_windows`, `sma_windows`, `range_windows`, `volume_windows`, `beta_windows` parameters; `_rolling_beta_60d` / `_rolling_market_corr_60d` renamed to `_rolling_beta` / `_rolling_market_corr` with configurable window
- `paper.py: _first_exit_decision` — accepts optional `vol_lookup` dict for vol-based trailing stop activation and exit
- `paper.py: _open_position_state` — handles trailing volatility stop activation alongside fixed trailing stop
- `portfolio_backtest.py: build_portfolio_targets` — accepts optional `returns` parameter for inverse-volatility weights
- `portfolio_backtest.py: _target_holdings_for_date` and `run_portfolio_backtest` — flow returns data for sizing
- `config/paper.yaml` — rebalance section updated with min_days_held and exit_below_score; added trailing_volatility_stop section (disabled by default)
- `scripts/update_paper_ledger.py: _load_exit_rules_config` — parses new rebalance and trailing_volatility_stop fields from YAML
- `scripts/run_symbol_discovery_rd.py` — removed dead `build_multi_lane_candidates` call and unused import

### Fixed

- Unused variable `multi_lane` in scripts/run_symbol_discovery_rd.py
- `ensemble.py: _summarize_ensemble_split` — wrap numpy array in pd.Series before passing to `information_coefficient` (expects Series)
- `ensemble.py: _compute_weights` — `abs(corr)` → `max(corr, 0.0)` to avoid flipping negative correlations; fallback to equal weights when all correlations ≤ 0
