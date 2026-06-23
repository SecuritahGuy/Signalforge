# SignalForge — Co-pilot Context

## Completed
- SQLite metadata store (`db.py`): full CRUD, paper_orders/run_history/account_snapshots tables, WAL journal, foreign keys, CSV export
- LightGBM & XGBoost wired into modeling pipeline (optional deps with try/except)
- Rebalance exit rule (replaces prior placeholder in paper.py)
- Inverse-volatility position sizing in portfolio_backtest.py
- Model ensemble module (`ensemble.py`): average/weighted/meta blending, walk-forward, all 9 tests passing
- GitHub Actions CI workflow
- **Trailing Volatility Stop exit rule** (`paper.py`): adaptive trail based on rolling daily vol, clamped by tightest/widest bounds, plumbed through full exit lifecycle
- **Time-based Decay exit rule** (`paper.py`): exits when days_held reaches `half_life_days * log2(entry_score / min_score_for_decay)`, giving higher-conviction entries longer runway
- **Multi-timeframe features** (`features.py`): extended windows from (5, 20, 60) → (5, 10, 20, 40, 60, 120) for returns/volatility/momentum; added SMA (10, 50, 200), range (10, 60, 120), volume (10, 40, 120) windows; beta/correlation at 60d and 120d; all sector/market relatives extended; configurable via function parameters
- **Sector-stop exit rule** (`paper.py`): exits when rolling mean return of the position's sector falls below `sector_decline_pct`; accepts optional `universe` DataFrame in `reconcile_exits` for symbol→sector mapping; enforces `min_sector_records` to avoid thin-sector false triggers
- 178 tests, all passing

## Blocked
(none)

## Key Files
- `src/signalforge/paper.py`: exit rules, paper lifecycle
- `src/signalforge/ensemble.py`: model ensembling
- `src/signalforge/db.py`: SQLite store
- `src/signalforge/modeling.py`: model training pipeline
- `src/signalforge/backtest.py`: backtesting
- `src/signalforge/validation.py`: walk-forward splits
- `src/signalforge/intraday.py`: intraday risk monitor
- `tests/test_ensemble.py`: 9 ensemble tests
- `tests/test_paper.py`: paper lifecycle tests
- `tests/test_db.py`: 17 DB tests
- `config/paper.yaml`: active configuration

## Code Conventions
- Type hints on all public functions
- Dataclasses for configs
- No breaking changes to CSV pipeline
- `from __future__ import annotations` at top of all modules
- `ddof=1` for sample std, `np.nan` for missing metrics
