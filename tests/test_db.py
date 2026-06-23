from __future__ import annotations

import pandas as pd
import pytest

from signalforge.db import SignalForgeDB, DatabaseConfig


@pytest.fixture
def db(tmp_path: str) -> SignalForgeDB:
    config = DatabaseConfig(path=str(tmp_path / "test.db"))
    return SignalForgeDB(config)


def test_schema_creates_tables(db: SignalForgeDB) -> None:
    tables = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = [r["name"] for r in tables]
    assert "account_snapshots" in names
    assert "paper_orders" in names
    assert "run_history" in names


def test_empty_ledger_returns_empty_dataframe(db: SignalForgeDB) -> None:
    ledger = db.get_ledger()
    assert ledger.empty
    assert "order_id" in ledger.columns


def test_upsert_order_returns_order_id(db: SignalForgeDB) -> None:
    order_id = db.upsert_order(
        {
            "order_id": "2024-01-01-AAPL-001",
            "status": "planned",
            "planned_date": "2024-01-01",
            "symbol": "AAPL",
            "score": 0.05,
            "shares": 10,
            "reference_price": 150.0,
        }
    )
    assert order_id == "2024-01-01-AAPL-001"

    ledger = db.get_ledger()
    assert len(ledger) == 1
    assert ledger.loc[0, "order_id"] == "2024-01-01-AAPL-001"
    assert ledger.loc[0, "status"] == "planned"


def test_upsert_order_updates_existing_row(db: SignalForgeDB) -> None:
    db.upsert_order(
        {
            "order_id": "2024-01-01-AAPL-001",
            "status": "planned",
            "planned_date": "2024-01-01",
            "symbol": "AAPL",
        }
    )
    db.upsert_order(
        {
            "order_id": "2024-01-01-AAPL-001",
            "status": "closed",
            "planned_date": "2024-01-01",
            "symbol": "AAPL",
            "net_pnl": 50.0,
        }
    )

    row = db.get_order("2024-01-01-AAPL-001")
    assert row is not None
    assert row["status"] == "closed"
    assert row["net_pnl"] == 50.0


def test_get_order_returns_none_for_missing(db: SignalForgeDB) -> None:
    assert db.get_order("nonexistent") is None


def test_replace_ledger_replaces_all_rows(db: SignalForgeDB) -> None:
    ledger = pd.DataFrame(
        {
            "order_id": ["id1", "id2"],
            "status": ["planned", "planned"],
            "planned_date": ["2024-01-01", "2024-01-02"],
            "symbol": ["AAPL", "MSFT"],
        }
    )
    count = db.replace_ledger(ledger)
    assert count == 2
    assert len(db.get_ledger()) == 2

    ledger2 = pd.DataFrame(
        {
            "order_id": ["id3"],
            "status": ["planned"],
            "planned_date": ["2024-01-01"],
            "symbol": ["GOOGL"],
        }
    )
    db.replace_ledger(ledger2)
    assert len(db.get_ledger()) == 1
    assert db.get_ledger().loc[0, "symbol"] == "GOOGL"


def test_append_orders_skips_duplicates(db: SignalForgeDB) -> None:
    first = pd.DataFrame(
        {
            "order_id": ["id1", "id2"],
            "status": ["planned", "planned"],
            "planned_date": ["2024-01-01", "2024-01-02"],
            "symbol": ["AAPL", "MSFT"],
        }
    )
    db.replace_ledger(first)

    second = pd.DataFrame(
        {
            "order_id": ["id2", "id3"],
            "status": ["planned", "planned"],
            "planned_date": ["2024-01-02", "2024-01-03"],
            "symbol": ["MSFT", "GOOGL"],
        }
    )
    added = db.append_orders(second)
    assert added == 1
    assert len(db.get_ledger()) == 3


def test_open_positions_filters_by_status(db: SignalForgeDB) -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["id1", "id2", "id3"],
            "status": ["open", "closed", "planned"],
            "planned_date": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "symbol": ["AAPL", "MSFT", "GOOGL"],
        }
    )
    db.replace_ledger(orders)
    open_positions = db.get_open_positions()
    assert len(open_positions) == 1
    assert open_positions.loc[0, "symbol"] == "AAPL"


def test_export_ledger_csv_writes_file(db: SignalForgeDB, tmp_path: str) -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["id1"],
            "status": ["planned"],
            "planned_date": ["2024-01-01"],
            "symbol": ["AAPL"],
        }
    )
    db.replace_ledger(orders)

    out = db.export_ledger_csv(str(tmp_path / "ledger.csv"))
    assert out.exists()
    reloaded = pd.read_csv(out)
    assert len(reloaded) == 1
    assert reloaded.loc[0, "symbol"] == "AAPL"


def test_run_history_append_and_retrieve(db: SignalForgeDB) -> None:
    run_id = db.append_run(
        {
            "run_date": "2024-01-01",
            "run_type": "paper_workflow",
            "mode": "after-close",
            "equity": 1000.0,
            "cash": 500.0,
        }
    )
    assert run_id > 0

    history = db.get_history()
    assert len(history) == 1
    assert history.loc[0, "equity"] == 1000.0


def test_run_history_replace(db: SignalForgeDB) -> None:
    db.append_run({"run_date": "2024-01-01", "run_type": "a", "mode": "x"})
    db.append_run({"run_date": "2024-01-02", "run_type": "b", "mode": "y"})
    assert len(db.get_history()) == 2

    new = pd.DataFrame(
        {
            "run_date": ["2024-02-01"],
            "run_type": ["c"],
            "mode": [""],
        }
    )
    db.replace_history(new)
    assert len(db.get_history()) == 1
    assert db.get_history().loc[0, "run_type"] == "c"


def test_export_history_csv_writes_file(db: SignalForgeDB, tmp_path: str) -> None:
    db.append_run({"run_date": "2024-01-01", "run_type": "test", "mode": ""})
    out = db.export_history_csv(str(tmp_path / "history.csv"))
    assert out.exists()
    reloaded = pd.read_csv(out)
    assert len(reloaded) == 1


def test_account_snapshots_add_and_retrieve(db: SignalForgeDB) -> None:
    db.add_snapshot(
        snapshot_date="2024-01-01",
        equity=1000.0,
        cash=800.0,
        open_positions=2,
    )
    db.add_snapshot(
        snapshot_date="2024-01-02",
        equity=1100.0,
        cash=900.0,
        open_positions=1,
    )

    snapshots = db.get_snapshots()
    assert len(snapshots) == 2
    assert snapshots.loc[0, "equity"] == 1000.0
    assert snapshots.loc[1, "equity"] == 1100.0


def test_context_manager_closes_connection(db: SignalForgeDB) -> None:
    path = db.path
    with SignalForgeDB(DatabaseConfig(path=path)) as db2:
        db2.add_snapshot(snapshot_date="2024-01-01", equity=100.0)
    # Should not raise
    with SignalForgeDB(DatabaseConfig(path=path)) as db3:
        assert len(db3.get_snapshots()) == 1


def test_trailing_stop_activated_round_trips_as_bool(db: SignalForgeDB) -> None:
    db.upsert_order(
        {
            "order_id": "id1",
            "status": "open",
            "planned_date": "2024-01-01",
            "symbol": "AAPL",
            "trailing_stop_activated": True,
        }
    )
    row = db.get_order("id1")
    assert row is not None
    assert row["trailing_stop_activated"] is True


def test_replace_ledger_handles_all_paper_columns(db: SignalForgeDB) -> None:
    from signalforge.paper import PAPER_LEDGER_COLUMNS

    data = {col: [f"val_{col}"] for col in PAPER_LEDGER_COLUMNS}
    data["order_id"] = ["test_full_cols"]
    data["status"] = ["planned"]
    data["planned_date"] = ["2024-01-01"]
    data["symbol"] = ["AAPL"]
    data["trailing_stop_activated"] = [True]
    data["shares"] = [10.0]

    frame = pd.DataFrame(data)
    db.replace_ledger(frame)

    ledger = db.get_ledger()
    assert len(ledger) == 1
    assert ledger.loc[0, "symbol"] == "AAPL"
    assert ledger.loc[0, "trailing_stop_activated"] == True


def test_vacuum_does_not_raise(db: SignalForgeDB) -> None:
    db.add_snapshot(snapshot_date="2024-01-01", equity=100.0)
    db.vacuum()  # should not raise
