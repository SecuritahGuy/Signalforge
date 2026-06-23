#!/usr/bin/env python3
"""One-time migration: import existing CSV artifacts into SQLite.

Usage:
    python scripts/migrate_to_db.py
    python scripts/migrate_to_db.py --db-path data/paper/signalforge.db
    python scripts/migrate_to_db.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from signalforge.db import SignalForgeDB, DatabaseConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import existing CSV artifacts into SQLite metadata store."
    )
    parser.add_argument("--db-path", default="data/paper/signalforge.db")
    parser.add_argument(
        "--ledger", default="data/paper/paper_trading_ledger.csv"
    )
    parser.add_argument("--history", default="reports/daily_runs/history.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = DatabaseConfig(path=args.db_path)
    summary: dict[str, int] = {}

    if args.dry_run:
        print("=== DRY RUN — no writes ===")

    # --- Paper ledger ---
    ledger_path = Path(args.ledger)
    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
        expected = {
            "order_id", "status", "planned_date", "symbol",
        }
        missing = expected.difference(ledger.columns)
        if missing:
            print(f"  Skipping ledger: missing columns {missing}")
        else:
            summary["ledger_rows"] = len(ledger)
            if not args.dry_run:
                db = SignalForgeDB(config)
                try:
                    replaced = db.replace_ledger(ledger)
                    print(f"  Paper ledger: {replaced} rows imported")
                finally:
                    db.close()
    else:
        print(f"  Ledger not found: {ledger_path}")

    # --- Run history ---
    history_path = Path(args.history)
    if history_path.exists():
        history = pd.read_csv(history_path)
        required = {"run_id", "local_time"}
        if required.issubset(history.columns):
            summary["history_rows"] = len(history)
            if not args.dry_run:
                db = SignalForgeDB(config)
                try:
                    replaced = db.replace_history(history)
                    print(f"  Run history: {replaced} rows imported")
                finally:
                    db.close()
    else:
        print(f"  History not found: {history_path}")

    total = sum(summary.values())
    if args.dry_run:
        print(f"\nWould import {total} total rows from {len(summary)} sources.")
    else:
        print(f"\nImported {total} rows from {len(summary)} sources into {args.db_path}")

    # --- Account snapshots from history CSV rows ---
    if not args.dry_run and "history_rows" in summary:
        db = SignalForgeDB(config)
        try:
            for _, row in history.iterrows():
                try:
                    snapshot_date = str(row.get("latest_price_date", ""))
                    if not snapshot_date or pd.isna(snapshot_date):
                        snapshot_date = str(row.get("run_id", ""))[:10]
                    equity = float(row.get("account_equity", 0))
                    if equity > 0:
                        db.add_snapshot(
                            snapshot_date=snapshot_date,
                            equity=equity,
                            cash=_safe_float(row.get("account_cash")),
                            realized_pnl=_safe_float(row.get("account_realized_pnl")),
                            unrealized_pnl=_safe_float(row.get("account_unrealized_pnl")),
                            open_positions=int(row.get("account_open_positions", 0)),
                            closed_positions=int(row.get("account_closed_positions", 0)),
                        )
                except (ValueError, TypeError):
                    continue
            print(f"  Account snapshots: derived from run history")
        finally:
            db.close()

    print("Done.")


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return f
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    main()
