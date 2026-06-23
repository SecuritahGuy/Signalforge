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

## Phase 1 — Close the gaps (next)

- [ ] **Implement rebalance exit rule** — the only explicit `# Placeholder` in the codebase (`paper.py:813`). Portfolio-level risk-weighted exits to complement per-position rules.
- [ ] **Wire up LightGBM and XGBoost** — optional deps exist in `pyproject.toml` but neither is integrated into `modeling.py`.
- [ ] **Add risk-parity and volatility-target position sizing** — currently only `equal_weight` is supported in `portfolio_backtest.py`.
- [ ] **Make YAML configs authoritative or remove them** — `config/research.yaml` is never loaded at runtime, creating a documentation/code drift risk.
- [ ] **Set up CI (GitHub Actions)** — run tests and lint on every push.
- [ ] **Initial git commit** — baseline the project.

## Phase 2 — Hardening

- [ ] **Persistent metadata store** — migrate paper ledger and run history from CSV to SQLite for queryability and integrity.
- [ ] **Add more exit rule variants** — trailing volatility stop, time-based decay, sector-stop.
- [ ] **Model ensemble and stacking** — combine Ridge/ElasticNet/RF/LGBM/XGBoost into blended predictions.
- [ ] **Multi-timeframe features** — add weekly/monthly rolling windows beyond the current 5/20/60.
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
