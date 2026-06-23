from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from signalforge.data import load_price_csv
from signalforge.intraday import (
    evaluate_intraday_risk_exits,
    latest_intraday_marks,
    normalize_intraday_marks,
    open_symbols,
)
from signalforge.paper import PaperTradingConfig
from signalforge.providers.yahoo import download_yahoo_intraday_marks

try:
    from scripts.update_paper_ledger import _load_exit_rules_config
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from update_paper_ledger import _load_exit_rules_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor open paper positions with lightweight intraday risk marks."
    )
    parser.add_argument("--ledger", default="data/paper/paper_trading_ledger.csv")
    parser.add_argument("--daily-prices", default="data/raw/yahoo_prices.csv")
    parser.add_argument("--marks-input", default=None)
    parser.add_argument("--marks-output", default="data/paper/intraday_marks.csv")
    parser.add_argument("--output-prefix", default="reports/paper_intraday_risk")
    parser.add_argument("--exit-rules-config", default="config/paper.yaml")
    parser.add_argument("--initial-capital", type=float, default=2_000.0)
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--period", default="1d")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument("--write-ledger", action="store_true")
    args = parser.parse_args()

    while True:
        run_once(args)
        if not args.loop:
            break
        time.sleep(max(5.0, args.interval_seconds))


def run_once(args: argparse.Namespace) -> None:
    ledger_path = Path(args.ledger)
    ledger = pd.read_csv(ledger_path)
    symbols = open_symbols(ledger)
    if not symbols:
        marks = pd.DataFrame()
        evaluated = ledger
        decisions = pd.DataFrame()
    elif args.marks_input:
        marks = normalize_intraday_marks(pd.read_csv(args.marks_input), source="csv")
    else:
        marks = download_yahoo_intraday_marks(
            symbols,
            interval=args.interval,
            period=args.period,
        )

    if symbols:
        daily_prices = load_price_csv(args.daily_prices) if Path(args.daily_prices).exists() else None
        exit_rules = _load_exit_rules_config(args.exit_rules_config, horizon_days=20)
        config = PaperTradingConfig(initial_capital=args.initial_capital, exit_rules=exit_rules)
        evaluated, decisions = evaluate_intraday_risk_exits(
            ledger,
            marks,
            daily_prices=daily_prices,
            config=config,
        )
    latest_marks = latest_intraday_marks(marks) if not marks.empty else marks
    summary = _summary(symbols, latest_marks, decisions)

    marks_path = Path(args.marks_output)
    marks_path.parent.mkdir(parents=True, exist_ok=True)
    if not latest_marks.empty:
        _append_marks(marks_path, latest_marks)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(output_prefix.with_name(output_prefix.name + "_decisions.csv"), index=False)
    output_prefix.with_name(output_prefix.name + "_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    output_prefix.with_name(output_prefix.name + "_report.md").write_text(
        _markdown_report(summary, decisions)
    )

    if args.write_ledger and summary["triggered_exits"] > 0:
        evaluated.to_csv(ledger_path, index=False)

    print(
        "intraday risk monitor: "
        f"{summary['open_symbols']} open symbols, "
        f"{summary['marked_symbols']} marked, "
        f"{summary['triggered_exits']} triggered exits"
    )
    if args.write_ledger and summary["triggered_exits"] > 0:
        print(f"wrote updated paper ledger to {ledger_path}")
    print(f"wrote intraday risk artifacts with prefix {output_prefix}")


def _append_marks(path: Path, marks: pd.DataFrame) -> None:
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, marks], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    else:
        combined = marks
    combined.to_csv(path, index=False)


def _summary(symbols: list[str], marks: pd.DataFrame, decisions: pd.DataFrame) -> dict:
    triggered = (
        decisions.loc[decisions["triggered"]]
        if not decisions.empty and "triggered" in decisions
        else decisions.iloc[0:0]
    )
    by_reason = (
        triggered.groupby("exit_reason", dropna=False).size().astype(int).to_dict()
        if not triggered.empty
        else {}
    )
    latest_timestamp = None
    if not marks.empty and "timestamp" in marks:
        latest_timestamp = pd.to_datetime(marks["timestamp"], errors="coerce").max()
    return {
        "mode": "intraday_risk_monitor",
        "latest_mark_timestamp": latest_timestamp,
        "open_symbols": len(symbols),
        "marked_symbols": int(marks["symbol"].nunique()) if not marks.empty else 0,
        "decision_rows": int(len(decisions)),
        "triggered_exits": int(len(triggered)),
        "triggered_by_reason": {str(key): int(value) for key, value in by_reason.items()},
        "write_ledger_required_for_persistence": True,
    }


def _markdown_report(summary: dict, decisions: pd.DataFrame) -> str:
    triggered = (
        decisions.loc[decisions["triggered"]]
        if not decisions.empty and "triggered" in decisions
        else decisions.iloc[0:0]
    )
    watch = (
        decisions.sort_values("current_return", ascending=True).head(15)
        if not decisions.empty and "current_return" in decisions
        else decisions
    )
    return "\n".join(
        [
            "# SignalForge Intraday Risk Monitor",
            "",
            f"Latest mark timestamp: `{summary['latest_mark_timestamp']}`",
            f"Open symbols: `{summary['open_symbols']}`",
            f"Marked symbols: `{summary['marked_symbols']}`",
            f"Triggered exits: `{summary['triggered_exits']}`",
            f"Triggered by reason: `{summary['triggered_by_reason']}`",
            "",
            "## Triggered Exits",
            "",
            _markdown_table(
                triggered[
                    [
                        "timestamp",
                        "symbol",
                        "price",
                        "current_return",
                        "highest_price_since_entry",
                        "exit_reason",
                        "exit_signal_value",
                    ]
                ]
                if not triggered.empty
                else triggered
            ),
            "",
            "## Watch List",
            "",
            _markdown_table(
                watch[
                    [
                        "timestamp",
                        "symbol",
                        "action",
                        "price",
                        "current_return",
                        "highest_price_since_entry",
                        "trailing_stop_activated",
                    ]
                ]
                if not watch.empty
                else watch
            ),
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


if __name__ == "__main__":
    main()
