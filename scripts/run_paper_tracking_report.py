from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize paper-trading history against the research backtest."
    )
    parser.add_argument("--history", default="reports/daily_runs/history.csv")
    parser.add_argument("--ledger", default="data/paper/paper_trading_ledger.csv")
    parser.add_argument("--backtest-summary", default="reports/paper_style_backtest_summary.json")
    parser.add_argument("--output-prefix", default="reports/paper_tracking")
    parser.add_argument("--initial-capital", type=float, default=2_000.0)
    args = parser.parse_args()

    history = pd.read_csv(args.history)
    ledger = pd.read_csv(args.ledger)
    backtest = _load_json(Path(args.backtest_summary))
    summary = build_tracking_summary(
        history,
        ledger,
        backtest,
        initial_capital=args.initial_capital,
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_name(output_prefix.name + "_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    output_prefix.with_name(output_prefix.name + "_report.md").write_text(
        render_tracking_report(summary)
    )
    print(f"wrote paper tracking report with prefix {output_prefix}")


def build_tracking_summary(
    history: pd.DataFrame,
    ledger: pd.DataFrame,
    backtest: dict,
    *,
    initial_capital: float,
) -> dict:
    normalized = history.copy()
    if not normalized.empty:
        normalized["local_time"] = pd.to_datetime(normalized["local_time"], errors="coerce")
        for column in (
            "account_equity",
            "account_cash",
            "account_realized_pnl",
            "account_unrealized_pnl",
            "audit_error_count",
            "audit_warning_count",
        ):
            if column in normalized:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        normalized = normalized.sort_values("local_time")

    latest = normalized.iloc[-1].to_dict() if not normalized.empty else {}
    equity = float(latest.get("account_equity", initial_capital))
    peak_equity = (
        float(normalized["account_equity"].max())
        if "account_equity" in normalized and normalized["account_equity"].notna().any()
        else equity
    )
    current_drawdown = equity / peak_equity - 1.0 if peak_equity else 0.0
    audit_errors = (
        int(normalized["audit_error_count"].fillna(0).sum())
        if "audit_error_count" in normalized
        else 0
    )
    audit_warnings = (
        int(normalized["audit_warning_count"].fillna(0).sum())
        if "audit_warning_count" in normalized
        else 0
    )
    closed = ledger.loc[ledger["status"] == "closed"] if not ledger.empty else ledger
    open_positions = ledger.loc[ledger["status"] == "open"] if not ledger.empty else ledger
    return {
        "history_rows": int(len(normalized)),
        "first_snapshot": _iso_or_none(normalized["local_time"].iloc[0])
        if not normalized.empty
        else None,
        "latest_snapshot": _iso_or_none(latest.get("local_time")),
        "latest_price_date": latest.get("latest_price_date"),
        "paper_equity": equity,
        "paper_cash": float(latest.get("account_cash", 0.0)),
        "paper_total_return": equity / initial_capital - 1.0,
        "paper_peak_equity": peak_equity,
        "paper_current_drawdown": current_drawdown,
        "paper_realized_pnl": float(latest.get("account_realized_pnl", 0.0)),
        "paper_unrealized_pnl": float(latest.get("account_unrealized_pnl", 0.0)),
        "open_positions": int(len(open_positions)),
        "closed_positions": int(len(closed)),
        "closed_win_rate": float((closed["net_pnl"] > 0).mean()) if not closed.empty else 0.0,
        "audit_error_count_total": audit_errors,
        "audit_warning_count_total": audit_warnings,
        "latest_audit_status": latest.get("audit_status", ""),
        "backtest_ending_equity": float(backtest.get("ending_equity", initial_capital)),
        "backtest_total_return": float(backtest.get("total_return", 0.0)),
        "backtest_sharpe": float(backtest.get("sharpe", 0.0)),
        "backtest_max_drawdown": float(backtest.get("max_drawdown", 0.0)),
        "backtest_closed_win_rate": float(backtest.get("closed_win_rate", 0.0)),
        "assessment": _assessment(len(closed), audit_errors),
    }


def render_tracking_report(summary: dict) -> str:
    return "\n".join(
        [
            "# SignalForge Paper Tracking Report",
            "",
            f"Assessment: `{summary['assessment']}`",
            f"History rows: `{summary['history_rows']}`",
            f"Latest snapshot: `{summary['latest_snapshot']}`",
            f"Latest price date: `{summary['latest_price_date']}`",
            "",
            "## Paper Account",
            "",
            f"- Equity: `${summary['paper_equity']:.2f}`",
            f"- Cash: `${summary['paper_cash']:.2f}`",
            f"- Total return: `{summary['paper_total_return']:.2%}`",
            f"- Current drawdown from observed peak: `{summary['paper_current_drawdown']:.2%}`",
            f"- Realized PnL: `${summary['paper_realized_pnl']:.2f}`",
            f"- Unrealized PnL: `${summary['paper_unrealized_pnl']:.2f}`",
            f"- Open positions: `{summary['open_positions']}`",
            f"- Closed positions: `{summary['closed_positions']}`",
            "",
            "## Realism Audit",
            "",
            f"- Latest status: `{summary['latest_audit_status']}`",
            f"- Total audit errors in history: `{summary['audit_error_count_total']}`",
            f"- Total audit warnings in history: `{summary['audit_warning_count_total']}`",
            "",
            "## Backtest Reference",
            "",
            f"- Ending equity: `${summary['backtest_ending_equity']:.2f}`",
            f"- Total return: `{summary['backtest_total_return']:.2%}`",
            f"- Sharpe: `{summary['backtest_sharpe']:.3f}`",
            f"- Max drawdown: `{summary['backtest_max_drawdown']:.2%}`",
            f"- Closed win rate: `{summary['backtest_closed_win_rate']:.2%}`",
            "",
        ]
    )


def _assessment(closed_positions: int, audit_errors: int) -> str:
    if audit_errors:
        return "fix realism issues before interpreting performance"
    if closed_positions < 10:
        return "too early for performance judgment"
    return "ready for paper-vs-backtest comparison"


def _iso_or_none(value) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
