from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from signalforge.data import load_price_csv
from signalforge.paper import PaperTradingConfig, mark_paper_positions, summarize_paper_account

try:
    from scripts.update_paper_ledger import _load_exit_rules_config, _load_scores
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from update_paper_ledger import _load_exit_rules_config, _load_scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mark open paper positions and write a monitoring-only report."
    )
    parser.add_argument("--ledger", default="data/paper/paper_trading_ledger.csv")
    parser.add_argument("--prices", default="data/raw/yahoo_prices.csv")
    parser.add_argument("--score-data", default="reports/paper_portfolio_watchlist.csv")
    parser.add_argument("--exit-rules-config", default="config/paper.yaml")
    parser.add_argument("--output-prefix", default="reports/paper_monitor")
    parser.add_argument("--initial-capital", type=float, default=2_000.0)
    args = parser.parse_args()

    ledger = pd.read_csv(args.ledger)
    prices = load_price_csv(args.prices)
    scores = _load_scores(Path(args.score_data))
    exit_rules = _load_exit_rules_config(args.exit_rules_config, horizon_days=20)
    config = PaperTradingConfig(initial_capital=args.initial_capital, exit_rules=exit_rules)
    marks = mark_paper_positions(ledger, prices, scores, config=config)
    summary = _monitor_summary(
        marks,
        ledger,
        summarize_paper_account(ledger, prices, initial_capital=args.initial_capital),
        prices,
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    marks.to_csv(output_prefix.with_name(output_prefix.name + "_positions.csv"), index=False)
    output_prefix.with_name(output_prefix.name + "_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    output_prefix.with_name(output_prefix.name + "_report.md").write_text(
        _markdown_report(summary, marks)
    )
    print(f"wrote paper monitor artifacts with prefix {output_prefix}")


def _monitor_summary(
    marks: pd.DataFrame,
    ledger: pd.DataFrame,
    account_summary: dict,
    prices: pd.DataFrame,
) -> dict:
    latest_price_timestamp = prices["date"].max().normalize() if not prices.empty else pd.NaT
    latest_price_date = (
        latest_price_timestamp.date().isoformat() if not pd.isna(latest_price_timestamp) else None
    )
    open_marks = marks.loc[marks["status"] == "open"] if not marks.empty else marks
    planned = marks.loc[marks["status"] == "planned"] if not marks.empty else marks
    normalized_ledger = ledger.copy()
    closed_today = 0
    closed_by_reason: dict[str, int] = {}
    if "exit_date" in normalized_ledger:
        closed = normalized_ledger.loc[normalized_ledger["status"] == "closed"]
        if not closed.empty and not pd.isna(latest_price_timestamp):
            exit_dates = pd.to_datetime(closed["exit_date"], errors="coerce")
            closed_today = int((exit_dates == latest_price_timestamp).sum())
            if "exit_reason" not in closed:
                closed = closed.assign(exit_reason="")
            closed_by_reason = (
                closed.loc[exit_dates == latest_price_timestamp]
                .groupby("exit_reason", dropna=False)
                .size()
                .to_dict()
            )
    sector_exposure = (
        open_marks.groupby("sector", dropna=False)["mark_value"].sum().to_dict()
        if not open_marks.empty
        else {}
    )
    return {
        "mode": "monitor_only_no_new_trades",
        "latest_price_date": latest_price_date,
        **account_summary,
        "exit_pending_positions": int((open_marks["action"] == "exit_pending").sum())
        if not open_marks.empty
        else 0,
        "hold_positions": int((open_marks["action"] == "hold").sum())
        if not open_marks.empty
        else 0,
        "waiting_for_fill": int((planned["action"] == "waiting_for_fill").sum())
        if not planned.empty
        else 0,
        "closed_today": closed_today,
        "closed_by_reason": {str(key): int(value) for key, value in closed_by_reason.items()},
        "next_scheduled_horizon_exits": _next_horizon_exits(open_marks),
        "sector_exposure": {str(key): float(value) for key, value in sector_exposure.items()},
    }


def _markdown_report(summary: dict, marks: pd.DataFrame) -> str:
    open_marks = marks.loc[marks["status"] == "open"] if not marks.empty else marks
    planned = marks.loc[marks["status"] == "planned"] if not marks.empty else marks
    return "\n".join(
        [
            "# SignalForge Paper Monitor",
            "",
            f"Latest price date: `{summary['latest_price_date']}`",
            "",
            "## Account",
            "",
            f"- Equity: `${summary['equity']:.2f}`",
            f"- Cash: `${summary['cash']:.2f}`",
            f"- Unrealized PnL: `${summary['unrealized_pnl']:.2f}`",
            f"- Realized PnL: `${summary['realized_pnl']:.2f}`",
            f"- Open positions: `{summary['open_positions']}`",
            f"- Waiting for fill: `{summary['waiting_for_fill']}`",
            f"- Exit pending: `{summary['exit_pending_positions']}`",
            f"- Closed today: `{summary['closed_today']}`",
            f"- Closed by reason: `{summary['closed_by_reason']}`",
            "",
            "## Open Positions",
            "",
            _markdown_table(
                open_marks[
                    [
                        "symbol",
                        "action",
                        "shares",
                        "entry_price",
                        "latest_price",
                        "mark_value",
                        "unrealized_pnl",
                        "unrealized_return",
                        "days_open",
                        "current_score",
                        "entry_score",
                        "highest_close_since_entry",
                        "trailing_stop_activated",
                        "target_exit_date",
                        "actual_exit_trigger_date",
                        "exit_reason",
                        "exit_signal_value",
                    ]
                ]
                if not open_marks.empty
                else open_marks
            ),
            "",
            "## Planned Orders Waiting For Fill",
            "",
            _markdown_table(
                planned[
                    [
                        "symbol",
                        "action",
                        "shares",
                        "reference_price",
                        "target_exit_date",
                    ]
                ]
                if not planned.empty
                else planned
            ),
            "",
            "No buy/sell decisions are generated in monitor mode.",
            "",
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    rows = [list(display.columns), ["---"] * len(display.columns)]
    rows.extend(display.astype(str).values.tolist())
    return "\n".join("| " + " | ".join(str(value) for value in row) + " |" for row in rows)


def _next_horizon_exits(open_marks: pd.DataFrame, limit: int = 5) -> list[dict]:
    if open_marks.empty:
        return []
    rows = (
        open_marks.loc[:, ["symbol", "target_exit_date"]]
        .dropna()
        .sort_values(["target_exit_date", "symbol"])
        .head(limit)
    )
    return rows.astype(str).to_dict(orient="records")


if __name__ == "__main__":
    main()
