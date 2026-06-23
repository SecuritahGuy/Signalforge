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
- **234 tests originally** — now **297 tests, all passing**
- **Removed `config/research.yaml`** — dead code, never loaded at runtime; research.py uses explicit function params and BacktestConfig dataclass
- **Server deps declared** — `pip install -e ".[server]"` for fastapi, uvicorn, sse-starlette, pydantic
- **Feature Engineering Expansion** (`features.py`, `modeling.py`): 6 new feature groups (lagged, calendar, cross-sectional z-scores, technical indicators, interaction, factor proxies) behind flags defaulting to off; `DEFAULT_FEATURE_COLUMNS` updated. 19 tests.
- **Hyperparameter Optimization** (`hyperopt.py`): Optuna with 5 model types, 3 search methods, median pruner, walk-forward CV objective. 16 tests.
- **Model Registry & Persistence** (`model_registry.py`, `db.py`): joblib serialization to `data/models/`, SQLite-backed metadata in `model_registry` table. 13 tests.
- **Logging & Error Handling** (`logging_config.py`, `exceptions.py`): 11 custom exceptions, `setup_logging()` with module-level overrides, integrated into 5 core modules. All bare `raise ValueError/KeyError` replaced.
- **Production Paper Trading** (`paper.py`, `db.py`, `test_paper_production.py`): `FillConfig` with partial fills (via `fill_pct`, `fill_window_days`, `filling` status), `BorrowCostConfig` with `hard_to_borrow_symbols` and `hard_to_borrow_rate`, `DividendConfig` with assumed-yield and DataFrame modes. `reconcile_borrow_costs()`, `reconcile_dividends()` functions. `_current_cash` and `_dedupe_active_orders` handle `filling` status. 20 tests.
- **Bug fixes**: `_dedupe_active_orders` includes `filling` status; `active_symbols` check in `reconcile_fills` scoped to `planned` orders only (was blocking `filling` re-fills); `_current_cash` includes `filling` orders in committed cash. 302 tests passing.<｜end▁of▁thinking｜>

<｜｜DSML｜｜parameter name="oldString" string="true">## Completed
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
- 234 tests, all passing (+58 new backtest tests, 57 direct + 1 existing migrated; fixed cooldown re-trigger bug in `backtest.py`)
- **Removed `config/research.yaml`** — dead code, never loaded at runtime; research.py uses explicit function params and BacktestConfig dataclass
- **Server deps declared** — `pip install -e ".[server]"` for fastapi, uvicorn, sse-starlette, pydantic

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
