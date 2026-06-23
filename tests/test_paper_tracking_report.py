import pandas as pd
import pytest

from scripts.run_paper_tracking_report import build_tracking_summary, render_tracking_report


def test_build_tracking_summary_marks_early_paper_history():
    history = pd.DataFrame(
        {
            "local_time": ["2026-05-27T09:00:00-05:00", "2026-05-27T10:00:00-05:00"],
            "latest_price_date": ["2026-05-27", "2026-05-27"],
            "account_equity": [2_010.0, 2_020.0],
            "account_cash": [100.0, 100.0],
            "account_realized_pnl": [0.0, 0.0],
            "account_unrealized_pnl": [10.0, 20.0],
            "audit_status": ["pass", "pass"],
            "audit_error_count": [0, 0],
            "audit_warning_count": [0, 1],
        }
    )
    ledger = pd.DataFrame({"status": ["open", "open"], "net_pnl": [0.0, 0.0]})
    backtest = {
        "ending_equity": 4_200.0,
        "total_return": 1.1,
        "sharpe": 0.8,
        "max_drawdown": -0.2,
        "closed_win_rate": 0.57,
    }

    summary = build_tracking_summary(history, ledger, backtest, initial_capital=2_000)

    assert summary["paper_total_return"] == pytest.approx(0.01)
    assert summary["audit_warning_count_total"] == 1
    assert summary["assessment"] == "too early for performance judgment"


def test_render_tracking_report_includes_backtest_reference():
    report = render_tracking_report(
        {
            "assessment": "too early",
            "history_rows": 1,
            "latest_snapshot": "2026-05-27T10:00:00-05:00",
            "latest_price_date": "2026-05-27",
            "paper_equity": 2_020.0,
            "paper_cash": 100.0,
            "paper_total_return": 0.01,
            "paper_current_drawdown": 0.0,
            "paper_realized_pnl": 0.0,
            "paper_unrealized_pnl": 20.0,
            "open_positions": 2,
            "closed_positions": 0,
            "latest_audit_status": "pass",
            "audit_error_count_total": 0,
            "audit_warning_count_total": 0,
            "backtest_ending_equity": 4_200.0,
            "backtest_total_return": 1.1,
            "backtest_sharpe": 0.8,
            "backtest_max_drawdown": -0.2,
            "backtest_closed_win_rate": 0.57,
        }
    )

    assert "Paper Tracking Report" in report
    assert "Backtest Reference" in report
