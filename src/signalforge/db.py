from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from signalforge.paper import PAPER_LEDGER_COLUMNS

DEFAULT_DB_PATH = "data/paper/signalforge.db"

_BOOL_COLUMNS = {"trailing_stop_activated"}
_DATE_COLUMNS = {
    "planned_date",
    "target_exit_date",
    "actual_exit_trigger_date",
    "fill_date",
    "exit_date",
    "snapshot_date",
    "run_date",
}


@dataclass(frozen=True)
class DatabaseConfig:
    path: str = DEFAULT_DB_PATH
    journal_mode: str = "WAL"


class SignalForgeDB:
    """SQLite metadata store for paper ledger, run history, and account snapshots."""

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        cfg = config or DatabaseConfig()
        db_path = Path(cfg.path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(f"PRAGMA journal_mode={cfg.journal_mode}")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._path = str(db_path)
        self._init_schema()

    @property
    def path(self) -> str:
        return self._path

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS paper_orders (
        order_id          TEXT PRIMARY KEY,
        status            TEXT NOT NULL DEFAULT 'planned',
        planned_date      TEXT NOT NULL,
        symbol            TEXT NOT NULL,
        sector            TEXT DEFAULT '',
        score             REAL DEFAULT 0.0,
        shares            REAL DEFAULT 0.0,
        reference_price   REAL DEFAULT 0.0,
        estimated_entry_value REAL DEFAULT 0.0,
        estimated_cost    REAL DEFAULT 0.0,
        target_exit_date  TEXT,
        actual_exit_trigger_date TEXT,
        fill_date         TEXT,
        entry_price       REAL,
        entry_value       REAL DEFAULT 0.0,
        entry_cost        REAL DEFAULT 0.0,
        exit_date         TEXT,
        exit_price        REAL,
        exit_value        REAL DEFAULT 0.0,
        exit_cost         REAL DEFAULT 0.0,
        gross_pnl         REAL DEFAULT 0.0,
        net_pnl           REAL DEFAULT 0.0,
        return_val        REAL DEFAULT 0.0,
        exit_reason       TEXT DEFAULT '',
        exit_signal_value REAL,
        exit_rule_version TEXT DEFAULT '',
        highest_close_since_entry REAL,
        trailing_stop_activated INTEGER DEFAULT 0,
        skip_reason       TEXT DEFAULT '',
        created_at        TEXT DEFAULT (datetime('now')),
        updated_at        TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS run_history (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        run_date          TEXT NOT NULL,
        run_type          TEXT NOT NULL,
        mode              TEXT,
        latest_price_date TEXT,
        equity            REAL,
        cash              REAL,
        realized_pnl      REAL,
        unrealized_pnl    REAL,
        open_positions    INTEGER DEFAULT 0,
        closed_positions  INTEGER DEFAULT 0,
        planned_orders    INTEGER DEFAULT 0,
        audit_status      TEXT,
        audit_error_count INTEGER DEFAULT 0,
        audit_warning_count INTEGER DEFAULT 0,
        backtest_equity   REAL,
        backtest_return   REAL,
        backtest_sharpe   REAL,
        backtest_max_drawdown REAL,
        backtest_win_rate REAL,
        summary_json      TEXT,
        created_at        TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS account_snapshots (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_date     TEXT NOT NULL,
        equity            REAL NOT NULL,
        cash              REAL,
        realized_pnl      REAL,
        unrealized_pnl    REAL,
        open_positions    INTEGER DEFAULT 0,
        closed_positions  INTEGER DEFAULT 0,
        created_at        TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_orders_status ON paper_orders(status);
    CREATE INDEX IF NOT EXISTS idx_orders_symbol ON paper_orders(symbol);
    CREATE INDEX IF NOT EXISTS idx_orders_planned_date ON paper_orders(planned_date);
    CREATE INDEX IF NOT EXISTS idx_run_history_run_date ON run_history(run_date);
    CREATE INDEX IF NOT EXISTS idx_snapshots_date ON account_snapshots(snapshot_date);
    """

    # ------------------------------------------------------------------
    # Paper ledger
    # ------------------------------------------------------------------

    def get_ledger(self) -> pd.DataFrame:
        """Return full paper ledger as a DataFrame."""
        rows = self._conn.execute(
            "SELECT * FROM paper_orders ORDER BY planned_date, order_id"
        ).fetchall()
        return self._rows_to_df(rows, PAPER_LEDGER_COLUMNS)

    def get_open_positions(self) -> pd.DataFrame:
        """Return only open positions."""
        rows = self._conn.execute(
            "SELECT * FROM paper_orders WHERE status = 'open' ORDER BY fill_date"
        ).fetchall()
        return self._rows_to_df(rows, PAPER_LEDGER_COLUMNS)

    def get_order(self, order_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM paper_orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        if "trailing_stop_activated" in result:
            result["trailing_stop_activated"] = bool(result["trailing_stop_activated"])
        return result

    def upsert_order(self, order: dict[str, Any]) -> str:
        """Insert or replace a single order row. Returns order_id."""
        row = _normalize_order_dict(order)
        columns = ", ".join(row)
        placeholders = ", ".join(":" + col for col in row)
        conflict_cols = ", ".join(
            col
            for col in row
            if col
            not in ("order_id", "created_at")
        )
        self._conn.execute(
            f"""
            INSERT INTO paper_orders ({columns})
            VALUES ({placeholders})
            ON CONFLICT(order_id) DO UPDATE SET
              {", ".join(f"{col} = excluded.{col}" for col in row if col not in ("order_id", "created_at"))},
              updated_at = datetime('now')
            """,
            row,
        )
        self._conn.commit()
        return str(row["order_id"])

    def replace_ledger(self, ledger: pd.DataFrame) -> int:
        """Replace entire ledger with DataFrame rows. Returns row count."""
        self._conn.execute("DELETE FROM paper_orders")
        count = self._append_ledger_rows(ledger)
        self._conn.commit()
        return count

    def append_orders(self, orders: pd.DataFrame) -> int:
        """Append new planned orders, skipping existing order_ids. Returns added count."""
        existing = {
            row["order_id"]
            for row in self._conn.execute("SELECT order_id FROM paper_orders").fetchall()
        }
        new = orders.loc[~orders["order_id"].isin(existing)]
        if new.empty:
            return 0
        count = self._append_ledger_rows(new)
        self._conn.commit()
        return count

    def _append_ledger_rows(self, frame: pd.DataFrame) -> int:
        db_cols = tuple(
            "return_val" if col == "return" else col for col in PAPER_LEDGER_COLUMNS
        )
        count = 0
        for _, row in frame.iterrows():
            self._conn.execute(
                f"""
                INSERT OR IGNORE INTO paper_orders ({", ".join(db_cols)})
                VALUES ({", ".join("?" for _ in db_cols)})
                """,
                _order_row_values(row),
            )
            count += 1
        return count

    def export_ledger_csv(self, path: str | Path) -> Path:
        """Export paper ledger to CSV (for dashboard backward compat)."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.get_ledger().to_csv(output, index=False)
        return output

    # ------------------------------------------------------------------
    # Run history
    # ------------------------------------------------------------------

    def get_history(self) -> pd.DataFrame:
        rows = self._conn.execute(
            "SELECT * FROM run_history ORDER BY run_date DESC"
        ).fetchall()
        return self._rows_to_df(rows)

    def append_run(self, run_data: dict[str, Any]) -> int:
        norm = {
            "run_date": str(run_data.get("run_date", "")),
            "run_type": str(run_data.get("run_type", "")),
            "mode": str(run_data.get("mode", "")),
            "latest_price_date": str(run_data.get("latest_price_date") or ""),
            "equity": _float_or_none(run_data.get("equity")),
            "cash": _float_or_none(run_data.get("cash")),
            "realized_pnl": _float_or_none(run_data.get("realized_pnl")),
            "unrealized_pnl": _float_or_none(run_data.get("unrealized_pnl")),
            "open_positions": int(run_data.get("open_positions", 0)),
            "closed_positions": int(run_data.get("closed_positions", 0)),
            "planned_orders": int(run_data.get("planned_orders", 0)),
            "audit_status": str(run_data.get("audit_status", "")),
            "audit_error_count": int(run_data.get("audit_error_count", 0)),
            "audit_warning_count": int(run_data.get("audit_warning_count", 0)),
            "backtest_equity": _float_or_none(run_data.get("backtest_equity")),
            "backtest_return": _float_or_none(run_data.get("backtest_return")),
            "backtest_sharpe": _float_or_none(run_data.get("backtest_sharpe")),
            "backtest_max_drawdown": _float_or_none(run_data.get("backtest_max_drawdown")),
            "backtest_win_rate": _float_or_none(run_data.get("backtest_win_rate")),
            "summary_json": json.dumps(run_data.get("summary", {}), default=str),
        }
        cursor = self._conn.execute(
            """
            INSERT INTO run_history (
                run_date, run_type, mode, latest_price_date,
                equity, cash, realized_pnl, unrealized_pnl,
                open_positions, closed_positions, planned_orders,
                audit_status, audit_error_count, audit_warning_count,
                backtest_equity, backtest_return, backtest_sharpe,
                backtest_max_drawdown, backtest_win_rate, summary_json
            ) VALUES (
                :run_date, :run_type, :mode, :latest_price_date,
                :equity, :cash, :realized_pnl, :unrealized_pnl,
                :open_positions, :closed_positions, :planned_orders,
                :audit_status, :audit_error_count, :audit_warning_count,
                :backtest_equity, :backtest_return, :backtest_sharpe,
                :backtest_max_drawdown, :backtest_win_rate, :summary_json
            )
            """,
            norm,
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def replace_history(self, history: pd.DataFrame) -> int:
        self._conn.execute("DELETE FROM run_history")
        count = 0
        for _, row in history.iterrows():
            self.append_run(row.to_dict())
            count += 1
        return count

    def export_history_csv(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.get_history().to_csv(output, index=False)
        return output

    # ------------------------------------------------------------------
    # Account snapshots
    # ------------------------------------------------------------------

    def add_snapshot(
        self,
        *,
        snapshot_date: str,
        equity: float,
        cash: float | None = None,
        realized_pnl: float | None = None,
        unrealized_pnl: float | None = None,
        open_positions: int = 0,
        closed_positions: int = 0,
    ) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO account_snapshots
                (snapshot_date, equity, cash, realized_pnl, unrealized_pnl,
                 open_positions, closed_positions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_date,
                equity,
                cash,
                realized_pnl,
                unrealized_pnl,
                open_positions,
                closed_positions,
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def get_snapshots(self) -> pd.DataFrame:
        rows = self._conn.execute(
            "SELECT * FROM account_snapshots ORDER BY snapshot_date"
        ).fetchall()
        return self._rows_to_df(rows)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SignalForgeDB:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def vacuum(self) -> None:
        self._conn.execute("VACUUM")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rows_to_df(
        rows: list[sqlite3.Row],
        columns: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=list(columns) if columns else None)
        frame = pd.DataFrame([dict(r) for r in rows])
        if columns:
            for col in columns:
                if col not in frame:
                    frame[col] = pd.NA
            frame = frame.loc[:, columns]
        for col in frame.columns:
            if col in _BOOL_COLUMNS:
                frame[col] = frame[col].fillna(False).astype(bool)
            elif col in _DATE_COLUMNS:
                frame[col] = pd.to_datetime(frame[col], errors="coerce")
        if "return_val" in frame.columns:
            frame = frame.rename(columns={"return_val": "return"})
        return frame


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _normalize_order_dict(order: dict[str, Any]) -> dict[str, Any]:
    row = dict(order)
    if "return" in row:
        row["return_val"] = row.pop("return")
    for key in _BOOL_COLUMNS:
        if key in row:
            row[key] = bool(row[key])
    for key in ("created_at", "updated_at"):
        row.pop(key, None)
    return row


def _order_row_values(row: pd.Series) -> list[Any]:
    values = []
    for col in PAPER_LEDGER_COLUMNS:
        raw = row.get(col)
        if col == "return":
            raw = row.get("return", row.get("return_val", 0.0))
        if col in _BOOL_COLUMNS:
            values.append(1 if raw else 0)
        elif col in _DATE_COLUMNS:
            try:
                ts = pd.Timestamp(raw)
                values.append(str(ts) if pd.notna(ts) else None)
            except (ValueError, TypeError):
                values.append(None)
        elif isinstance(raw, (np.generic,)):
            values.append(raw.item() if pd.notna(raw) else None)
        elif pd.isna(raw) if hasattr(pd, "isna") else raw is None:
            values.append(None)
        else:
            values.append(raw)
    return values


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return None if np.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
