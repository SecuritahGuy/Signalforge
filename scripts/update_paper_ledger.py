from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from signalforge.data import load_price_csv
from signalforge.paper import (
    ExitRulesConfig,
    PaperTradingConfig,
    RebalanceConfig,
    ScoreDeteriorationConfig,
    StopLossConfig,
    TrailingStopConfig,
    build_planned_orders,
    reconcile_exits,
    reconcile_fills,
    summarize_paper_account,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or reconcile the persistent SignalForge paper-trading ledger."
    )
    parser.add_argument(
        "--ledger",
        default="data/paper/paper_trading_ledger.csv",
        help="Persistent paper ledger path.",
    )
    parser.add_argument("--prices", default="data/raw/yahoo_prices.csv")
    parser.add_argument(
        "--planned-orders",
        default="reports/paper_portfolio_order_ledger.csv",
        help="Daily planned-order artifact from run_paper_portfolio.py.",
    )
    parser.add_argument("--summary-output", default="reports/paper_account_summary.json")
    parser.add_argument(
        "--score-data",
        default="reports/paper_portfolio_watchlist.csv",
        help="Optional daily score file for score-deterioration exits.",
    )
    parser.add_argument(
        "--exit-rules-config",
        default=None,
        help="Optional YAML-like file with an exit_rules section.",
    )
    parser.add_argument("--initial-capital", type=float, default=2_000.0)
    parser.add_argument("--position-weight", type=float, default=0.10)
    parser.add_argument("--long-fraction", type=float, default=0.10)
    parser.add_argument("--min-score", type=float, default=0.01)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--allow-fractional-shares", action="store_true")
    parser.add_argument(
        "--skip-add-plans",
        action="store_true",
        help="Only reconcile existing ledger rows; do not add today's planned orders.",
    )
    args = parser.parse_args()

    exit_rules = _load_exit_rules_config(args.exit_rules_config, horizon_days=args.horizon)
    config = PaperTradingConfig(
        initial_capital=args.initial_capital,
        position_weight=args.position_weight,
        long_fraction=args.long_fraction,
        min_score=args.min_score,
        horizon=args.horizon,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        allow_fractional_shares=args.allow_fractional_shares,
        exit_rules=exit_rules,
    )
    prices = load_price_csv(args.prices)
    scores = _load_scores(Path(args.score_data))
    ledger_path = Path(args.ledger)
    ledger = _load_or_empty_ledger(ledger_path)

    if not args.skip_add_plans:
        planned_artifact = pd.read_csv(args.planned_orders)
        planned_artifact["date"] = pd.to_datetime(planned_artifact["date"])
        existing_summary = summarize_paper_account(
            ledger,
            prices,
            initial_capital=args.initial_capital,
        )
        open_symbols = set()
        if not ledger.empty and "status" in ledger and "symbol" in ledger:
            active = ledger["status"].isin(["open", "planned"])
            open_symbols = set(ledger.loc[active, "symbol"].str.upper())
        planned = build_planned_orders(
            planned_artifact.rename(columns={"reference_price": "adj_close"}),
            config=config,
            available_cash=existing_summary["cash"],
            excluded_symbols=open_symbols,
        )
        ledger = _append_new_orders(ledger, planned)

    ledger = reconcile_fills(ledger, prices, config=config)
    ledger = reconcile_exits(ledger, prices, scores=scores, config=config)
    summary = summarize_paper_account(ledger, prices, initial_capital=args.initial_capital)

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(ledger_path, index=False)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"wrote {len(ledger):,} paper ledger rows to {ledger_path}")
    print(f"wrote paper account summary to {summary_path}")


def _load_or_empty_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_scores(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "score" not in frame or "symbol" not in frame:
        return None
    return frame


def _load_exit_rules_config(path: str | None, *, horizon_days: int) -> ExitRulesConfig:
    if path is None:
        return ExitRulesConfig(horizon_days=horizon_days)
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"exit rules config does not exist: {config_path}")
    data = _parse_simple_yaml(config_path.read_text()).get("exit_rules", {})
    return ExitRulesConfig(
        horizon_days=int(data.get("horizon_days", horizon_days)),
        stop_loss=StopLossConfig(
            enabled=bool(data.get("stop_loss", {}).get("enabled", False)),
            pct=float(data.get("stop_loss", {}).get("pct", -0.08)),
        ),
        trailing_stop=TrailingStopConfig(
            enabled=bool(data.get("trailing_stop", {}).get("enabled", False)),
            activate_at_return=float(
                data.get("trailing_stop", {}).get("activate_at_return", 0.12)
            ),
            trail_from_high_pct=float(
                data.get("trailing_stop", {}).get("trail_from_high_pct", -0.06)
            ),
        ),
        score_deterioration=ScoreDeteriorationConfig(
            enabled=bool(data.get("score_deterioration", {}).get("enabled", False)),
            min_days_held=int(data.get("score_deterioration", {}).get("min_days_held", 5)),
            exit_below_score=float(
                data.get("score_deterioration", {}).get("exit_below_score", 0.005)
            ),
            exit_if_score_declines_pct=float(
                data.get("score_deterioration", {}).get("exit_if_score_declines_pct", 0.60)
            ),
        ),
        rebalance=RebalanceConfig(
            enabled=bool(data.get("rebalance", {}).get("enabled", False)),
            min_days_held=int(data.get("rebalance", {}).get("min_days_held", 10)),
            exit_below_score=float(
                data.get("rebalance", {}).get("exit_below_score", 0.01)
            ),
        ),
    )


def _parse_simple_yaml(text: str) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, raw_value = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value.strip() == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(raw_value.strip())
    return root


def _parse_scalar(value: str) -> object:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def _append_new_orders(existing: pd.DataFrame, planned: pd.DataFrame) -> pd.DataFrame:
    planned = planned.loc[planned["status"] == "planned"].copy()
    if planned.empty:
        return existing
    if existing.empty:
        return planned
    known_ids = set(existing["order_id"])
    active_symbols = set(
        existing.loc[existing["status"].isin(["open", "planned"]), "symbol"].str.upper()
    )
    known_symbol_dates = {
        (pd.Timestamp(row.planned_date).date().isoformat(), row.symbol.upper())
        for row in existing.itertuples(index=False)
    }
    new_orders = planned.loc[
        ~planned["order_id"].isin(known_ids)
        & ~planned["symbol"].str.upper().isin(active_symbols)
        & ~planned.apply(
            lambda row: (
                pd.Timestamp(row["planned_date"]).date().isoformat(),
                row["symbol"].upper(),
            )
            in known_symbol_dates,
            axis=1,
        )
    ]
    if new_orders.empty:
        return existing
    return pd.concat([existing, new_orders], ignore_index=True)


if __name__ == "__main__":
    main()
