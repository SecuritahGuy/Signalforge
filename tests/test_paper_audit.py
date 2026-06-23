import pandas as pd

from signalforge.paper_audit import PaperAuditConfig, run_paper_realism_audit


def test_paper_realism_audit_passes_clean_ledger():
    ledger = _ledger()
    prices = _prices()

    audit = run_paper_realism_audit(
        ledger,
        prices,
        as_of_date="2024-01-02",
        config=PaperAuditConfig(initial_capital=1_000),
    )

    assert audit["status"] == "pass"
    assert audit["finding_count"] == 0


def test_paper_realism_audit_flags_negative_cash_and_duplicate_active_symbols():
    ledger = pd.concat([_ledger(), _ledger()], ignore_index=True)
    ledger.loc[1, "order_id"] = "duplicate"
    account_summary = {
        "cash": -5.0,
        "equity": 995.0,
        "open_positions": 2,
        "planned_orders": 0,
    }

    audit = run_paper_realism_audit(
        ledger,
        _prices(),
        account_summary=account_summary,
        as_of_date="2024-01-02",
        config=PaperAuditConfig(initial_capital=1_000),
    )

    codes = {finding["code"] for finding in audit["findings"]}
    assert audit["status"] == "fail"
    assert "negative_cash" in codes
    assert "duplicate_active_symbols" in codes
    assert "account_cash_mismatch" in codes


def test_paper_realism_audit_flags_stale_price_file_and_sector_exposure():
    ledger = _ledger()
    prices = _prices()

    audit = run_paper_realism_audit(
        ledger,
        prices,
        as_of_date="2024-01-10",
        config=PaperAuditConfig(
            initial_capital=1_000,
            max_price_staleness_days=3,
            max_sector_exposure=0.05,
        ),
    )

    codes = {finding["code"] for finding in audit["findings"]}
    assert audit["status"] == "pass"
    assert audit["warning_count"] == 2
    assert "stale_price_file" in codes
    assert "sector_exposure_high" in codes


def _ledger() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["one"],
            "status": ["open"],
            "planned_date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["A"],
            "sector": ["Tech"],
            "score": [0.03],
            "shares": [2.0],
            "reference_price": [100.0],
            "estimated_entry_value": [200.0],
            "estimated_cost": [0.0],
            "target_exit_date": pd.to_datetime(["2024-01-29"]),
            "fill_date": pd.to_datetime(["2024-01-02"]),
            "entry_price": [100.0],
            "entry_value": [200.0],
            "entry_cost": [0.0],
            "exit_date": [pd.NaT],
            "exit_price": [0.0],
            "exit_value": [0.0],
            "exit_cost": [0.0],
            "gross_pnl": [0.0],
            "net_pnl": [0.0],
            "return": [0.0],
            "skip_reason": [""],
        }
    )


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["A", "A"],
            "open": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 100.0],
            "close": [100.0, 100.0],
            "adj_close": [100.0, 100.0],
            "volume": [1_000_000, 1_000_000],
        }
    )
