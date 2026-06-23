# SignalForge Roadmap

## Phase 0 — Foundation (current)

- [x] Core data pipeline: price/universe ingestion, features, labels
- [x] Walk-forward validation with purge/embargo
- [x] Modeling pipeline: Ridge, ElasticNet, RandomForest
- [x] Full long/short and long-only backtesting with risk controls
- [x] Paper trading lifecycle: order planning, fills, exits, audit
- [x] Five exit rules (stop-loss, trailing-stop, score-deterioration, horizon, rebalance placeholder)
- [x] Discovery engine with 5 watchlist lanes + multi-lane scoring
- [x] Experiment grid (6 feature sets × 3 model families)
- [x] Fundamentals enrichment
- [x] Intraday risk monitor
- [x] Paper-style backtest (simulates live lifecycle date-by-date)
- [x] Web dashboard (React/TypeScript) with 12 views
- [x] 30+ CLI scripts and 30 test files

## Phase 1 — Close the gaps

- [x] **Implement rebalance exit rule** — replaces the prior placeholder in `paper.py`.
- [x] **Wire up LightGBM and XGBoost** — optional deps in `pyproject.toml`, integrated into `modeling.py` with try/except.
- [x] **Add risk-parity and volatility-target position sizing** — inverse-volatility sizing in `portfolio_backtest.py`.
- [ ] **Make YAML configs authoritative or remove them** — `config/research.yaml` is never loaded at runtime, creating a documentation/code drift risk.
- [x] **Set up CI (GitHub Actions)** — runs tests on every push.
- [x] **Initial git commit** — baselines the project.

## Phase 2 — Hardening

- [x] **Persistent metadata store** — SQLite paper ledger, run history, account snapshots in `db.py`.
- [x] **Trailing Volatility Stop** — adaptive stop distance based on rolling daily volatility, clamped by tightest/widest bounds.
- [x] **Time-based Decay** — exits when `days_held >= half_life_days * log2(entry_score / min_score_for_decay)`; higher-conviction entries get longer runway.
- [ ] **Add more exit rule variants** — sector-stop.
- [x] **Model ensemble and stacking** — average/weighted/meta blending in `ensemble.py` with walk-forward.
- [x] **Multi-timeframe features** — extended windows (5, 10, 20, 40, 60, 120) for returns/vol/momentum; SMA (10, 50, 200); range (10, 60, 120); volume/dollar-volume at (10, 40, 120); beta/corr at 60d/120d; all sector/market relatives extended; configurable via function params.
- [ ] **Feature importance monitoring** — drift detection for feature distributions and model SHAP values over time.
- [ ] **Dashboard auth and multi-session** — if the web UI moves beyond localhost.

## Phase 3 — Research depth

- [ ] **Alternative data sources** — add provider interface (FRED, Alpha Vantage, Intrinio) alongside yfinance.
- [ ] **Options-flow features** — put/call ratios, open interest changes (if data source permits).
- [ ] **Macro regime detection** — label regimes (risk-on/risk-off, volatility, trend) and condition model selection.
- [ ] **Cross-sectional PCA features** — dimensionality reduction across the universe for latent factor exposure.
- [ ] **Attention/transformer baseline** — experiment with a small transformer for sequential return prediction.

## Phase 4 — Live-adjacent

- [ ] **Broker adapter interface** — abstract order placement, position reconciliation, P&L sync. Intentionally not a live integration itself, but a defined contract so one could be written.
- [ ] **Real-time paper monitor** — push notifications for triggered exits (email or Telegram).
- [ ] **Calendar-aware scheduling** — handle holidays, early closes, and corporate actions in the daily workflow.
- [ ] **Multi-currency and ADR support** — extend the universe model beyond USD-only.

## Non-goals (explicit)

- Live broker integration, automated trading, or investment recommendations — see README.
