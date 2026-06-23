from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain which daily paper candidates are actionable after live constraints."
    )
    parser.add_argument("--daily-orders", default="reports/paper_portfolio_order_ledger.csv")
    parser.add_argument("--ledger", default="data/paper/paper_trading_ledger.csv")
    parser.add_argument("--account-summary", default="reports/paper_account_summary.json")
    parser.add_argument("--prices", default="data/raw/yahoo_prices.csv")
    parser.add_argument("--output-prefix", default="reports/paper_actionability")
    args = parser.parse_args()

    daily_orders = pd.read_csv(args.daily_orders)
    ledger = pd.read_csv(args.ledger) if Path(args.ledger).exists() else pd.DataFrame()
    account = _load_account_summary(Path(args.account_summary))
    latest_price_date = _latest_price_date(Path(args.prices))

    candidates = build_actionability_candidates(
        daily_orders,
        ledger,
        account_summary=account,
        latest_price_date=latest_price_date,
    )
    summary = build_actionability_summary(candidates, ledger, account_summary=account)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_prefix.with_name(output_prefix.name + "_candidates.csv"), index=False)
    output_prefix.with_name(output_prefix.name + "_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    output_prefix.with_name(output_prefix.name + "_report.md").write_text(
        render_actionability_report(summary, candidates)
    )
    print(f"wrote paper actionability report with prefix {output_prefix}")


def build_actionability_candidates(
    daily_orders: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    account_summary: dict,
    latest_price_date: str | None = None,
) -> pd.DataFrame:
    required = {"status", "symbol", "score", "skip_reason"}
    missing = required.difference(daily_orders.columns)
    if missing:
        raise KeyError(f"daily_orders is missing required columns: {sorted(missing)}")

    active_symbols = _active_symbols(ledger)
    cash = float(account_summary.get("cash", 0.0))
    output = daily_orders.copy()
    output["symbol"] = output["symbol"].astype(str).str.upper()
    output["status"] = output["status"].fillna("").astype(str)
    output["skip_reason"] = output["skip_reason"].fillna("").astype(str)
    output["score"] = pd.to_numeric(output["score"], errors="coerce").fillna(0.0)
    plan_date = _first_present(output, "date")
    output["plan_date"] = plan_date
    output["latest_price_date"] = latest_price_date
    output["plan_is_latest_price_date"] = bool(
        latest_price_date is None
        or plan_date is None
        or plan_date == latest_price_date
    )

    if "estimated_required_cash" not in output:
        output["estimated_required_cash"] = 0.0
    output["estimated_required_cash"] = pd.to_numeric(
        output["estimated_required_cash"], errors="coerce"
    ).fillna(0.0)
    if "estimated_entry_value" not in output:
        output["estimated_entry_value"] = 0.0
    output["estimated_entry_value"] = pd.to_numeric(
        output["estimated_entry_value"], errors="coerce"
    ).fillna(0.0)

    output["already_active"] = output["symbol"].isin(active_symbols)
    output["cash_available"] = cash
    output["effective_action"] = output.apply(_effective_action, axis=1)
    output["actionable_new_order"] = output["effective_action"].eq("actionable_new_order")
    output["cash_after_order"] = output.apply(
        lambda row: cash - row["estimated_required_cash"]
        if row["actionable_new_order"]
        else pd.NA,
        axis=1,
    )
    return output.sort_values("score", ascending=False).reset_index(drop=True)


def build_actionability_summary(
    candidates: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    account_summary: dict,
) -> dict:
    action_counts = candidates["effective_action"].value_counts().sort_index().to_dict()
    skip_counts = (
        candidates.loc[candidates["status"].eq("skipped"), "skip_reason"]
        .fillna("")
        .replace("", "unspecified")
        .value_counts()
        .to_dict()
    )
    actionable = candidates.loc[candidates["actionable_new_order"]]
    active = ledger.loc[ledger["status"].isin(["planned", "open"])] if not ledger.empty else ledger
    as_of_date = _first_present(candidates, "date")
    latest_price_date = _first_present(candidates, "latest_price_date")
    plan_is_latest_price_date = (
        bool(candidates["plan_is_latest_price_date"].all())
        if "plan_is_latest_price_date" in candidates
        else True
    )
    return {
        "as_of_date": as_of_date,
        "latest_price_date": latest_price_date,
        "plan_is_latest_price_date": plan_is_latest_price_date,
        "buy_generation_mode": "after_close_only",
        "cash": float(account_summary.get("cash", 0.0)),
        "equity": float(account_summary.get("equity", 0.0)),
        "open_positions": int(account_summary.get("open_positions", 0)),
        "planned_orders": int(account_summary.get("planned_orders", 0)),
        "active_symbols": int(len(_active_symbols(ledger))),
        "candidate_count": int(len(candidates)),
        "model_planned_count": int(candidates["status"].eq("planned").sum()),
        "actionable_new_order_count": int(len(actionable)),
        "actionable_estimated_required_cash": float(actionable["estimated_required_cash"].sum()),
        "blocked_by_active_symbol_count": int(
            candidates["effective_action"].eq("blocked_active_symbol").sum()
        ),
        "active_ledger_rows": int(len(active)),
        "effective_action_counts": action_counts,
        "skip_reason_counts": skip_counts,
    }


def render_actionability_report(summary: dict, candidates: pd.DataFrame) -> str:
    actionable = candidates.loc[candidates["actionable_new_order"]].head(15)
    blocked = candidates.loc[candidates["effective_action"].eq("blocked_active_symbol")].head(15)
    skipped = candidates.loc[candidates["status"].eq("skipped")].head(15)
    return "\n".join(
        [
            "# SignalForge Paper Actionability",
            "",
            f"Plan date: `{summary['as_of_date']}`",
            f"Latest price date: `{summary.get('latest_price_date')}`",
            f"Plan matches latest price date: `{summary.get('plan_is_latest_price_date')}`",
            f"Buy generation mode: `{summary.get('buy_generation_mode')}`",
            "",
            (
                "> This is the last after-close buy plan. New buys are not generated during "
                "market-hours monitor cycles."
                if not summary.get("plan_is_latest_price_date", True)
                else "> This plan matches the latest synced daily price date."
            ),
            "",
            "## Account Constraints",
            "",
            f"- Equity: `${summary['equity']:.2f}`",
            f"- Cash: `${summary['cash']:.2f}`",
            f"- Open positions: `{summary['open_positions']}`",
            f"- Planned orders: `{summary['planned_orders']}`",
            f"- Active symbols: `{summary['active_symbols']}`",
            "",
            "## Candidate Summary",
            "",
            f"- Candidates reviewed: `{summary['candidate_count']}`",
            f"- Model-planned rows before live constraints: `{summary['model_planned_count']}`",
            f"- Actionable new orders after live constraints: `{summary['actionable_new_order_count']}`",
            f"- Estimated required cash for actionable orders: "
            f"`${summary['actionable_estimated_required_cash']:.2f}`",
            f"- Blocked by already-active symbols: `{summary['blocked_by_active_symbol_count']}`",
            "",
            "## Effective Actions",
            "",
            _dict_table(summary["effective_action_counts"], "action", "rows"),
            "",
            "## Skip Reasons",
            "",
            _dict_table(summary["skip_reason_counts"], "skip_reason", "rows"),
            "",
            "## Actionable New Orders",
            "",
            _candidate_table(actionable),
            "",
            "## Top Active-Symbol Blocks",
            "",
            _candidate_table(blocked),
            "",
            "## Top Skipped Candidates",
            "",
            _candidate_table(skipped),
            "",
        ]
    )


def _effective_action(row: pd.Series) -> str:
    if not bool(row.get("plan_is_latest_price_date", True)):
        return "stale_plan_wait_for_after_close"
    if bool(row["already_active"]):
        return "blocked_active_symbol"
    if row["status"] == "planned":
        if row["estimated_required_cash"] > row["cash_available"]:
            return "blocked_insufficient_cash"
        return "actionable_new_order"
    reason = str(row.get("skip_reason", "") or "skipped")
    return f"skipped_{reason}"


def _load_account_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _latest_price_date(path: Path) -> str | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, usecols=["date"])
    if frame.empty:
        return None
    return pd.to_datetime(frame["date"], errors="coerce").max().date().isoformat()


def _active_symbols(ledger: pd.DataFrame) -> set[str]:
    if ledger.empty or not {"status", "symbol"}.issubset(ledger.columns):
        return set()
    active = ledger.loc[ledger["status"].isin(["planned", "open"]), "symbol"]
    return set(active.astype(str).str.upper())


def _first_present(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame or frame.empty:
        return None
    value = frame[column].dropna().iloc[0] if not frame[column].dropna().empty else None
    if value is None:
        return None
    return pd.Timestamp(value).date().isoformat()


def _dict_table(values: dict, key_name: str, value_name: str) -> str:
    if not values:
        return "_No rows._"
    rows = [[key_name, value_name], ["---", "---"]]
    rows.extend([[str(key), str(value)] for key, value in values.items()])
    return "\n".join("| " + " | ".join(str(value) for value in row) + " |" for row in rows)


def _candidate_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [
        "symbol",
        "status",
        "effective_action",
        "score",
        "reference_price",
        "shares",
        "estimated_entry_value",
        "estimated_required_cash",
        "skip_reason",
    ]
    display = frame.loc[:, [column for column in columns if column in frame]].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    rows = [list(display.columns), ["---"] * len(display.columns)]
    rows.extend(display.astype(str).values.tolist())
    return "\n".join("| " + " | ".join(str(value) for value in row) + " |" for row in rows)


if __name__ == "__main__":
    main()
