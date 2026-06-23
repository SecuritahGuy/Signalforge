import pandas as pd

from scripts.run_paper_actionability_report import (
    build_actionability_candidates,
    build_actionability_summary,
    render_actionability_report,
)


def test_actionability_report_separates_active_blocks_from_new_orders():
    daily_orders = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01"] * 4),
            "status": ["planned", "planned", "skipped", "planned"],
            "symbol": ["A", "B", "C", "D"],
            "score": [0.04, 0.03, 0.02, 0.01],
            "reference_price": [100.0, 50.0, 20.0, 200.0],
            "shares": [1.0, 2.0, 0.0, 2.0],
            "estimated_entry_value": [100.0, 100.0, 0.0, 400.0],
            "estimated_required_cash": [100.1, 100.1, 0.0, 400.4],
            "skip_reason": ["", "", "score_below_threshold", ""],
        }
    )
    ledger = pd.DataFrame({"status": ["open"], "symbol": ["A"]})
    account = {"cash": 150.0, "equity": 1_000.0, "open_positions": 1, "planned_orders": 0}

    candidates = build_actionability_candidates(daily_orders, ledger, account_summary=account)
    summary = build_actionability_summary(candidates, ledger, account_summary=account)
    report = render_actionability_report(summary, candidates)

    assert candidates.loc[candidates["symbol"].eq("A"), "effective_action"].item() == (
        "blocked_active_symbol"
    )
    assert candidates.loc[candidates["symbol"].eq("B"), "effective_action"].item() == (
        "actionable_new_order"
    )
    assert candidates.loc[candidates["symbol"].eq("D"), "effective_action"].item() == (
        "blocked_insufficient_cash"
    )
    assert summary["actionable_new_order_count"] == 1
    assert summary["blocked_by_active_symbol_count"] == 1
    assert "Actionable New Orders" in report


def test_actionability_report_marks_stale_plan_as_not_live_buyable():
    daily_orders = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01"]),
            "status": ["planned"],
            "symbol": ["A"],
            "score": [0.04],
            "reference_price": [100.0],
            "shares": [1.0],
            "estimated_entry_value": [100.0],
            "estimated_required_cash": [100.1],
            "skip_reason": [""],
        }
    )
    ledger = pd.DataFrame(columns=["status", "symbol"])
    account = {"cash": 500.0, "equity": 1_000.0, "open_positions": 0, "planned_orders": 0}

    candidates = build_actionability_candidates(
        daily_orders,
        ledger,
        account_summary=account,
        latest_price_date="2026-06-02",
    )
    summary = build_actionability_summary(candidates, ledger, account_summary=account)
    report = render_actionability_report(summary, candidates)

    assert candidates.loc[0, "effective_action"] == "stale_plan_wait_for_after_close"
    assert not summary["plan_is_latest_price_date"]
    assert summary["actionable_new_order_count"] == 0
    assert "last after-close buy plan" in report
