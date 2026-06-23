from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from signalforge.data import load_price_csv
from signalforge.paper import PaperTradingConfig, mark_paper_positions, reconcile_exits

try:
    from scripts.update_paper_ledger import _load_exit_rules_config, _load_scores
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from update_paper_ledger import _load_exit_rules_config, _load_scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate configured paper exit rules against open positions."
    )
    parser.add_argument("--ledger", default="data/paper/paper_trading_ledger.csv")
    parser.add_argument("--prices", default="data/raw/yahoo_prices.csv")
    parser.add_argument("--score-data", default="reports/paper_portfolio_watchlist.csv")
    parser.add_argument("--exit-rules-config", default="config/paper.yaml")
    parser.add_argument("--output-ledger", default=None)
    parser.add_argument("--summary-output", default="reports/paper_exit_rules_summary.json")
    parser.add_argument("--write-ledger", action="store_true")
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    ledger = pd.read_csv(ledger_path)
    prices = load_price_csv(args.prices)
    scores = _load_scores(Path(args.score_data))
    exit_rules = _load_exit_rules_config(args.exit_rules_config, horizon_days=20)
    config = PaperTradingConfig(exit_rules=exit_rules)

    evaluated = reconcile_exits(ledger, prices, scores=scores, config=config)
    marks = mark_paper_positions(evaluated, prices, scores, config=config)
    summary = _summary(ledger, evaluated, marks)

    output_ledger = ledger_path if args.write_ledger else Path(args.output_ledger or args.ledger)
    if args.write_ledger or args.output_ledger:
        output_ledger.parent.mkdir(parents=True, exist_ok=True)
        evaluated.to_csv(output_ledger, index=False)

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"evaluated exit rules for {summary['open_before']} open positions")
    print(f"exit-triggered positions: {summary['newly_closed']}")
    print(f"wrote exit-rule summary to {summary_path}")


def _summary(before: pd.DataFrame, after: pd.DataFrame, marks: pd.DataFrame) -> dict:
    before_open = before.loc[before["status"] == "open"] if "status" in before else before.iloc[0:0]
    after_closed = after.loc[after["status"] == "closed"] if "status" in after else after.iloc[0:0]
    before_closed_ids = (
        set(before.loc[before["status"] == "closed", "order_id"])
        if {"status", "order_id"}.issubset(before.columns)
        else set()
    )
    newly_closed = after_closed.loc[~after_closed["order_id"].isin(before_closed_ids)]
    closed_by_reason = (
        newly_closed.groupby("exit_reason", dropna=False).size().astype(int).to_dict()
        if "exit_reason" in newly_closed
        else {}
    )
    exit_pending = (
        int((marks["action"] == "exit_pending").sum())
        if not marks.empty and "action" in marks
        else 0
    )
    return {
        "open_before": int(len(before_open)),
        "newly_closed": int(len(newly_closed)),
        "exit_pending_positions": exit_pending,
        "closed_by_reason": {str(key): int(value) for key, value in closed_by_reason.items()},
    }


if __name__ == "__main__":
    main()
