# Changelog

All notable changes to SignalForge are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- ROADMAP.md with phased plan (foundation, gaps, hardening, research depth, live-adjacent)
- CHANGELOG.md
- model_type support for `lgbm` and `xgboost` in modeling pipeline
- Rebalance exit rule in paper.py (replaces prior placeholder)
- Inverse-volatility position sizing method in portfolio_backtest.py
- GitHub Actions CI workflow (`.github/workflows/ci.yml`)
- `RebalanceConfig` fields `min_days_held` and `exit_below_score`
- `learning_rate`, `subsample`, `colsample_bytree`, `num_leaves`, `min_child_samples`, `min_child_weight` hyperparameters to `BaselineModelConfig`
- Tests for LightGBM, XGBoost, rebalance exit rule, and inverse-volatility sizing

### Changed

- `paper.py: _exit_decision_for_date` — rebalance rule now checks min_days_held and exit_below_score
- `portfolio_backtest.py: build_portfolio_targets` — accepts optional `returns` parameter for inverse-volatility weights
- `portfolio_backtest.py: _target_holdings_for_date` and `run_portfolio_backtest` — flow returns data for sizing
- `config/paper.yaml` — rebalance section updated with min_days_held and exit_below_score
- `scripts/update_paper_ledger.py: _load_exit_rules_config` — parses new rebalance fields from YAML
- `scripts/run_symbol_discovery_rd.py` — removed dead `build_multi_lane_candidates` call and unused import

### Fixed

- Unused variable `multi_lane` in scripts/run_symbol_discovery_rd.py
