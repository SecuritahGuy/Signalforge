# SignalForge

SignalForge is a stock ML research project built around leakage-aware forecasting,
walk-forward validation, realistic backtesting, and explicit risk controls.

The first target is a disciplined daily-equity research loop:

1. Ingest point-in-time market and reference data.
2. Build timestamped features that only use information known at prediction time.
3. Generate forward-return and excess-return labels.
4. Validate models with walk-forward splits, purging, and embargo periods.
5. Backtest predictions with costs, turnover, liquidity, and benchmark comparison.
6. Promote only model versions that survive out-of-sample and trading-style metrics.

## Initial Modeling Contract

- Prediction timestamp: after market close on day `t`.
- Primary labels: 5-day and 20-day forward excess return.
- First benchmark model: linear baseline, then LightGBM or XGBoost.
- Required validation: walk-forward validation with purging for overlapping labels.
- Required evaluation: Sharpe, max drawdown, turnover, hit rate, and cost-adjusted return.

## Project Layout

```text
src/signalforge/
  data.py         CSV ingestion and normalized price/universe contracts
  features.py     Leakage-safe price, volume, and sector-relative features
  labels.py       Forward-return and excess-return label generation
  research.py     End-to-end research-frame and smoke-backtest helpers
  validation.py   Walk-forward split generation with purge and embargo support
  metrics.py      Trading and prediction-quality metrics
  backtest.py     Long/short portfolio simulation from model scores
data/reference/
  tracked_universe.csv  Initial symbols: mega-cap tech, selected fun names, SPY
config/
  paper.yaml      Active exit-rules and paper-trading config
tests/
  Unit tests for leakage-sensitive behavior
```

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest
```

For Yahoo Finance data downloads:

```bash
pip install -e ".[dev,data,apple]"
python scripts/check_mlx_acceleration.py
python scripts/download_yahoo_prices.py --start 2018-01-01 --output data/raw/yahoo_prices.csv
python scripts/build_research_frame.py --prices data/raw/yahoo_prices.csv \
  --output data/processed/research_frame.csv
python scripts/train_baseline.py --research-frame data/processed/research_frame.csv
python scripts/train_baseline.py --research-frame data/processed/research_frame.csv \
  --model-type random_forest \
  --predictions-output data/processed/random_forest_predictions.csv \
  --summary-output reports/random_forest_walkforward_summary.csv \
  --metadata-output reports/random_forest_model_metadata.json \
  --feature-importance-output reports/random_forest_feature_importance.csv
python scripts/compare_model_summaries.py \
  --summary ridge=reports/baseline_walkforward_summary.csv \
  --summary random_forest=reports/random_forest_walkforward_summary.csv \
  --output reports/model_comparison.csv
python scripts/build_research_frame.py --prices data/raw/yahoo_prices.csv \
  --horizons 5,20 \
  --output data/processed/research_frame.csv
python scripts/run_experiments.py --research-frame data/processed/research_frame.csv \
  --leaderboard-output reports/experiment_leaderboard.csv \
  --split-summary-output reports/experiment_split_summaries.csv
python scripts/run_experiments.py --research-frame data/processed/research_frame.csv \
  --fast \
  --target-volatility 0.12 \
  --max-drawdown-stop 0.12 \
  --cooldown-days 20 \
  --max-symbol-trades 25 \
  --leaderboard-output reports/risk_experiment_leaderboard.csv \
  --split-summary-output reports/risk_experiment_split_summaries.csv
python scripts/audit_top_experiment.py \
  --leaderboard reports/risk_experiment_leaderboard.csv \
  --initial-capital 2000 \
  --output-prefix reports/top_experiment
python scripts/run_concentration_sensitivity.py \
  --predictions reports/top_experiment_predictions.csv \
  --realized-return fwd_20d_return \
  --output reports/concentration_sensitivity.csv
python scripts/run_bankroll_sensitivity.py \
  --predictions reports/top_experiment_predictions.csv \
  --output reports/bankroll_sensitivity.csv
```

## Daily Paper Workflow

Use the daily paper workflow runner to choose the right routine for the current
time of day:

```bash
.venv/bin/python scripts/run_daily_paper_workflow.py
```

Before market open, `auto` mode refreshes prices, rebuilds the research frame,
and reconciles existing paper orders/fills/exits without adding new planned
orders.

During regular market hours, `auto` mode also writes a monitoring-only report
that marks open paper positions to the latest available daily price and labels
positions as `hold`, `exit_pending`, or `waiting_for_fill`. It does not generate
new buy or sell decisions. It also runs the realism audit so negative cash,
duplicate active symbols, stale prices, and exposure issues are visible.

After the cutoff, `auto` mode also generates the latest paper portfolio and
appends new planned orders to the persistent paper ledger. The after-close
bundle also refreshes the monitor report, paper-style historical backtest, and
model visibility report unless those heavier reports are skipped.

Useful overrides:

```bash
.venv/bin/python scripts/run_daily_paper_workflow.py --mode reconcile
.venv/bin/python scripts/run_daily_paper_workflow.py --mode intraday-monitor
.venv/bin/python scripts/run_daily_paper_workflow.py --mode after-close
.venv/bin/python scripts/run_daily_paper_workflow.py --dry-run
```

For a hands-off runner that keeps refreshing data, reconciling the paper ledger,
writing monitor reports, and choosing the right routine from the local clock, run:

```bash
.venv/bin/python scripts/run_daily_paper_workflow.py --loop --interval-minutes 30
```

To run the paper loop plus the research-only S&P 500 symbol-discovery monitor from
one terminal, use the daily operations runner:

```bash
.venv/bin/python scripts/run_daily_operations.py --loop --interval-minutes 30
```

It runs the paper workflow every cycle. During market hours, it also runs the
lightweight intraday risk monitor for open paper positions. After close, it runs
the broad symbol discovery refresh once per latest price date and writes a
combined review to `reports/daily_ops_review.md`. At the end of every cycle it
also refreshes `web/public/data/dashboard.json`, so the local dashboard can show
the latest files after pressing its reload button.

That is the "just run it" command. Use `Ctrl-C` to stop it. Each completed cycle
also writes a timestamped snapshot under `reports/daily_runs/` and appends a row
to `reports/daily_runs/history.csv` so equity, cash, positions, monitor state,
and backtest health can be tracked over time. To run the after-close plan without
the heavier backtest/visibility reports:

```bash
.venv/bin/python scripts/run_daily_paper_workflow.py --mode after-close \
  --skip-backtest \
  --skip-visibility
```

To run the operations loop without refreshing the local UI bundle:

```bash
.venv/bin/python scripts/run_daily_operations.py \
  --loop \
  --interval-minutes 30 \
  --skip-dashboard-sync
```

The default daily paper loop uses `--paper-min-score 0.02`. The R&D portfolio
rules showed this higher threshold improves closed-trade win rate versus the
older `0.01` threshold, at the cost of fewer trades and lower total return. To
favor more activity over hit rate, lower it explicitly:

```bash
.venv/bin/python scripts/run_daily_operations.py --loop --paper-min-score 0.01
```

The paper loop also enables fractional shares by default. With a small paper
account, this keeps high-priced, high-scoring symbols from being skipped only
because a whole share is larger than the target position size. To force
whole-share-only realism:

```bash
.venv/bin/python scripts/run_daily_operations.py \
  --loop \
  --no-paper-allow-fractional-shares
```

The persistent paper account lives in:

```text
data/paper/paper_trading_ledger.csv
data/paper/paper_trading_skipped_archive.csv
reports/paper_account_summary.json
reports/paper_monitor_report.md
reports/paper_realism_audit_report.md
reports/paper_style_backtest_summary.json
reports/model_visibility_summary.md
reports/daily_runs/history.csv
```

The persistent ledger is intended to track lifecycle rows: planned, open, and
closed positions. Daily order artifacts keep the full skipped-order audit trail.
If older skipped rows make the persistent ledger noisy, compact it with:

```bash
.venv/bin/python scripts/compact_paper_ledger.py
```

By default, daily snapshots avoid duplicating very large generated data files.
Use `--include-large-artifacts` only when you specifically want a full archived
copy of the price file, research frame, and paper-style backtest ledger.

### Paper Exit Rules

Paper positions are still opened from the daily paper portfolio plan, but exits
are now evaluated from `config/paper.yaml` whenever the paper ledger is
reconciled. The default rules are:

1. `stop_loss`: close an open position if its marked return is at or below
   `-8%`.
2. `trailing_stop`: after a position has gained at least `12%`, close it if the
   latest adjusted close falls `6%` or more from the highest adjusted close seen
   since entry.
3. `score_deterioration`: after 5 business days, close it if the current score
   is at or below `0.005` or has declined by at least `60%` from entry score.
4. `rebalance`: reserved for future portfolio-level exits; it is disabled and
   does not generate turnover today.
5. `horizon`: close on or after the configured 20-business-day target exit date
   if no earlier rule has triggered.

The evaluator uses the first matching rule in that order and records
`exit_reason`, `exit_signal_value`, `actual_exit_trigger_date`,
`highest_close_since_entry`, `trailing_stop_activated`, and
`exit_rule_version` in the persistent ledger.

To dry-run the rules against the current ledger without mutating it:

```bash
.venv/bin/python scripts/evaluate_exit_rules.py \
  --ledger data/paper/paper_trading_ledger.csv \
  --prices data/raw/yahoo_prices.csv \
  --score-data reports/paper_portfolio_watchlist.csv \
  --exit-rules-config config/paper.yaml
```

To reconcile and persist fills/exits, use the normal workflow or
`scripts/update_paper_ledger.py`. The monitor report and local dashboard expose
the exit reason, score state, trailing-stop state, and next scheduled horizon
date for open positions.

### Intraday Risk Monitor

The daily workflow uses daily Yahoo OHLCV bars. The daily operations runner runs
the lightweight intraday monitor by default during market hours:

```bash
.venv/bin/python scripts/run_daily_operations.py --loop --interval-minutes 30
```

The intraday monitor downloads recent marks for currently open paper symbols,
evaluates only price-risk exits (`intraday_stop_loss` and
`intraday_trailing_stop`), and writes:

```text
data/paper/intraday_marks.csv
reports/paper_intraday_risk_decisions.csv
reports/paper_intraday_risk_summary.json
reports/paper_intraday_risk_report.md
```

By default it is dry-run for ledger persistence: it reports triggered exits but
does not mutate `data/paper/paper_trading_ledger.csv`. Add
`--intraday-risk-write-ledger` to the daily operations runner only when you want
triggered intraday paper exits to be persisted.

To disable intraday risk checks for a run:

```bash
.venv/bin/python scripts/run_daily_operations.py \
  --loop \
  --interval-minutes 30 \
  --skip-intraday-risk
```

You can also run the intraday monitor standalone:

```bash
.venv/bin/python scripts/run_intraday_risk_monitor.py \
  --loop \
  --interval-seconds 300
```

## R&D Experiments

While the paper account runs, use the R&D runner to compare feature families and
portfolio rules against the current walk-forward/paper-style setup:

```bash
.venv/bin/python scripts/run_rd_experiments.py
```

It writes:

```text
reports/rd_feature_ablation.csv
reports/rd_feature_ablation_splits.csv
reports/rd_portfolio_rules.csv
reports/rd_summary.md
```

The portfolio-rule test uses executable-style entries from `next_open` and exits
from `exit_close_20d`, so it is closer to the paper ledger mechanics than a raw
close-to-close return comparison.

## Symbol Discovery R&D

Use the symbol-discovery R&D feed to monitor potential additions without adding
them to the traded paper universe:

```bash
.venv/bin/python scripts/build_broad_universe.py \
  --source sp500 \
  --output data/reference/sp500_universe.csv
.venv/bin/python scripts/download_yahoo_prices.py \
  --universe data/reference/sp500_universe.csv \
  --start 2020-01-01 \
  --output data/raw/sp500_yahoo_prices.csv
.venv/bin/python scripts/build_research_frame.py \
  --prices data/raw/sp500_yahoo_prices.csv \
  --universe data/reference/sp500_universe.csv \
  --horizons 5,20 \
  --output data/processed/sp500_research_frame.csv
.venv/bin/python scripts/run_symbol_discovery_rd.py \
  --research-frame data/processed/sp500_research_frame.csv \
  --universe data/reference/sp500_universe.csv \
  --existing-watchlist data/reference/tracked_universe.csv
```

It monitors the top third of eligible discovery candidates by default and writes:

```text
reports/symbol_discovery_rd/candidates.csv
reports/symbol_discovery_rd/monitoring_state.csv
reports/symbol_discovery_rd/promotion_candidates.csv
reports/symbol_discovery_rd/report.md
reports/symbol_discovery_rd/summary.json
```

Promotion candidates are still review-only. A symbol must survive the monitoring
period, appear repeatedly, clear liquidity/price filters, and not already be
tracked or active in paper trading before it is marked `eligible_for_review`.

To convert review-eligible discovery names into a promotion plan:

```bash
.venv/bin/python scripts/promote_discovery_candidates.py
```

That command is dry-run by default and writes:

```text
reports/symbol_discovery_promotion_plan_candidates.csv
reports/symbol_discovery_promotion_plan_report.md
reports/symbol_discovery_promotion_plan_summary.json
```

The promotion plan defaults to a five-symbol batch. Eligible rows beyond
`--max-symbols` are kept visible in the blocked section with a
`max_symbols_limit_*` blocker, so `--approve` only appends the reviewed batch.
Additional gates can require a stronger score, multiple discovery lanes,
repeated appearances, a minimum monitoring age, and a maximum number of
same-sector additions:

```bash
.venv/bin/python scripts/promote_discovery_candidates.py \
  --min-discovery-score 70 \
  --min-lane-count 2 \
  --min-appearances 3 \
  --min-monitoring-age-days 5 \
  --max-sector-symbols 2
```

Only after reviewing the plan should you append names to the traded paper
universe:

```bash
.venv/bin/python scripts/promote_discovery_candidates.py --approve
```

Promotion updates `data/reference/tracked_universe.csv`; it does not create
orders by itself. The next paper workflow cycle must score the expanded universe
before any promoted symbol can become a paper candidate.

To let the daily operations loop auto-approve discovery promotions after those
same gates pass, opt in explicitly:

```bash
.venv/bin/python scripts/run_daily_operations.py \
  --loop \
  --auto-approve-discovery-promotions \
  --promotion-min-discovery-score 70 \
  --promotion-min-lane-count 2 \
  --promotion-min-appearances 3 \
  --promotion-min-monitoring-age-days 5 \
  --promotion-max-sector-symbols 2
```

This only automates promotion into the paper universe. Paper order generation,
cash use, position sizing, exits, audits, and actionability checks still run
through the existing paper workflow gates.

## Paper Tracking Review

Use the daily review bundle to refresh audit, rebuild history, and compare live
paper history against the backtest reference:

```bash
.venv/bin/python scripts/run_daily_review_bundle.py
```

It writes:

```text
reports/paper_tracking_summary.json
reports/paper_tracking_report.md
```

For a heavier review that also refreshes the R&D ablation and portfolio-rule
tables:

```bash
.venv/bin/python scripts/run_daily_review_bundle.py --include-rd
```

`yfinance` is a third-party client for Yahoo Finance data. Treat it as a convenient
research data source, not a point-in-time institutional feed.

## Initial Tracked Universe

The starter universe lives in `data/reference/tracked_universe.csv`.

It includes mega-cap technology names such as AAPL, MSFT, NVDA, GOOGL, AMZN,
META, TSLA, and AMD, plus a few more interesting names for research variety:
PLTR, RIVN, DIS, COST, SBUX, and NKE. SPY is included as the default benchmark
for excess-return labels.

## Current Status

This repo is scaffolded for local research. It intentionally does not include a live
broker integration, automated trading, or investment recommendations.
