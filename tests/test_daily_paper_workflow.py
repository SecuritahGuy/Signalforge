import csv
import json
from argparse import Namespace
from datetime import datetime, time
from zoneinfo import ZoneInfo

from scripts.run_daily_paper_workflow import (
    HISTORY_COLUMNS,
    after_close_heavy_already_ran,
    append_history,
    build_steps,
    collect_tracking_metrics,
    latest_price_date_from_prices,
    parse_clock_time,
    resolve_mode,
    snapshot_run,
)


def test_resolve_mode_uses_after_close_cutoff():
    before_open = datetime(2026, 5, 26, 7, 0, tzinfo=ZoneInfo("America/Chicago"))
    before_close = datetime(2026, 5, 26, 10, 0, tzinfo=ZoneInfo("America/Chicago"))
    after_close = datetime(2026, 5, 26, 16, 0, tzinfo=ZoneInfo("America/Chicago"))

    assert (
        resolve_mode(
            requested_mode="auto",
            now=before_open,
            market_open_time=time(8, 30),
            after_close_time=time(15, 45),
        )
        == "reconcile"
    )
    assert (
        resolve_mode(
            requested_mode="auto",
            now=before_close,
            market_open_time=time(8, 30),
            after_close_time=time(15, 45),
        )
        == "intraday-monitor"
    )
    assert (
        resolve_mode(
            requested_mode="auto",
            now=after_close,
            market_open_time=time(8, 30),
            after_close_time=time(15, 45),
        )
        == "after-close"
    )
    assert (
        resolve_mode(
            requested_mode="after-close",
            now=before_close,
            market_open_time=time(8, 30),
            after_close_time=time(15, 45),
        )
        == "after-close"
    )


def test_build_steps_for_reconcile_only_skips_new_plan_generation():
    names = [step.name for step in build_steps(_args(), mode="reconcile")]

    assert names == [
        "refresh-yahoo-prices",
        "rebuild-research-frame",
        "reconcile-existing-paper-ledger",
    ]


def test_build_steps_for_after_close_appends_new_plans():
    steps = build_steps(_args(), mode="after-close")
    names = [step.name for step in steps]

    assert names == [
        "refresh-yahoo-prices",
        "rebuild-research-frame",
        "reconcile-existing-paper-ledger",
        "generate-paper-portfolio",
        "write-paper-actionability-report",
        "append-new-paper-plans",
        "write-paper-monitor-report",
        "run-paper-realism-audit",
        "run-paper-style-backtest",
        "run-model-visibility-report",
    ]
    commands = {step.name: step.command for step in steps}
    assert _arg_value(commands["generate-paper-portfolio"], "--min-score") == "0.02"
    assert _arg_value(commands["append-new-paper-plans"], "--min-score") == "0.02"
    assert _arg_value(commands["run-paper-style-backtest"], "--min-score") == "0.02"
    assert _arg_value(commands["run-model-visibility-report"], "--min-score") == "0.02"
    assert "--allow-fractional-shares" in commands["generate-paper-portfolio"]
    assert "--allow-fractional-shares" in commands["append-new-paper-plans"]
    assert "--allow-fractional-shares" in commands["run-paper-style-backtest"]


def test_build_steps_for_after_close_can_skip_heavy_reports():
    args = _args()
    args.skip_backtest = True
    args.skip_visibility = True

    names = [step.name for step in build_steps(args, mode="after-close")]

    assert "run-paper-style-backtest" not in names
    assert "run-model-visibility-report" not in names
    assert names[-1] == "run-paper-realism-audit"


def test_build_steps_for_after_close_can_skip_heavy_after_first_run():
    names = [
        step.name
        for step in build_steps(_args(), mode="after-close", include_after_close_heavy=False)
    ]

    assert names == [
        "refresh-yahoo-prices",
        "rebuild-research-frame",
        "reconcile-existing-paper-ledger",
        "write-paper-monitor-report",
        "write-paper-actionability-report",
        "run-paper-realism-audit",
    ]


def test_build_steps_for_after_close_can_skip_audit():
    args = _args()
    args.skip_audit = True

    names = [step.name for step in build_steps(args, mode="after-close")]

    assert "run-paper-realism-audit" not in names


def test_build_steps_for_intraday_monitor_writes_monitor_report():
    names = [step.name for step in build_steps(_args(), mode="intraday-monitor")]

    assert names == [
        "refresh-yahoo-prices",
        "rebuild-research-frame",
        "reconcile-existing-paper-ledger",
        "write-paper-monitor-report",
        "write-paper-actionability-report",
        "run-paper-realism-audit",
    ]


def test_parse_clock_time():
    assert parse_clock_time("15:45") == time(15, 45)


def test_after_close_guard_detects_prior_heavy_run_for_latest_price_date(tmp_path):
    prices = tmp_path / "prices.csv"
    prices.write_text("date,symbol\n2026-05-31,A\n2026-06-01,A\n")
    history = tmp_path / "history.csv"
    history.write_text(
        "run_id,mode,latest_price_date,portfolio_planned_order_count\n"
        "one,after-close,2026-06-01,8\n"
    )

    latest_price_date = latest_price_date_from_prices(prices)

    assert latest_price_date == "2026-06-01"
    assert after_close_heavy_already_ran(history, latest_price_date=latest_price_date)
    assert not after_close_heavy_already_ran(history, latest_price_date="2026-06-02")


def test_collect_tracking_metrics_flattens_scalar_summary_fields(tmp_path):
    args = _args(tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/paper_account_summary.json").write_text(
        json.dumps(
            {
                "equity": 2021.5,
                "cash": 321.7,
                "sector_exposure": {"Technology": 100.0},
            }
        )
    )
    (tmp_path / "reports/paper_monitor_summary.json").write_text(
        json.dumps({"latest_price_date": "2026-05-26", "waiting_for_fill": 2})
    )
    (tmp_path / "reports/paper_style_backtest_summary.json").write_text(
        json.dumps({"ending_equity": 4205.2, "max_drawdown": -0.19})
    )

    metrics = collect_tracking_metrics(args)

    assert metrics["account_equity"] == 2021.5
    assert metrics["monitor_latest_price_date"] == "2026-05-26"
    assert metrics["backtest_max_drawdown"] == -0.19
    assert "account_sector_exposure" not in metrics


def test_snapshot_run_copies_artifacts_and_appends_history(tmp_path):
    args = _args(tmp_path)
    reports = tmp_path / "reports"
    paper = tmp_path / "data/paper"
    reports.mkdir(parents=True)
    paper.mkdir(parents=True)
    (paper / "paper_trading_ledger.csv").write_text("symbol,status\nAAPL,open\n")
    (reports / "paper_account_summary.json").write_text(
        json.dumps(
            {
                "equity": 2021.5,
                "cash": 321.7,
                "realized_pnl": 1.0,
                "unrealized_pnl": 20.5,
                "open_positions": 9,
                "planned_orders": 2,
            }
        )
    )
    (reports / "paper_monitor_summary.json").write_text(
        json.dumps(
            {
                "latest_price_date": "2026-05-26",
                "waiting_for_fill": 2,
                "exit_pending_positions": 0,
                "hold_positions": 9,
            }
        )
    )
    (reports / "paper_style_backtest_summary.json").write_text(
        json.dumps(
            {
                "ending_equity": 4205.2,
                "total_return": 1.1,
                "sharpe": 0.82,
                "max_drawdown": -0.19,
                "closed_win_rate": 0.57,
            }
        )
    )
    (reports / "paper_realism_audit_summary.json").write_text(
        json.dumps({"status": "pass", "error_count": 0, "warning_count": 1})
    )

    run_dir = snapshot_run(
        args,
        resolved_mode="intraday-monitor",
        steps=build_steps(args, mode="intraday-monitor"),
        now=datetime(2026, 5, 26, 11, 0, tzinfo=ZoneInfo("America/Chicago")),
    )

    assert (run_dir / "run_summary.json").exists()
    assert (run_dir / "paper_account_summary.json").exists()
    assert (run_dir / "paper_trading_ledger.csv").exists()

    with (tmp_path / "reports/daily_runs/history.csv").open() as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["mode"] == "intraday-monitor"
    assert rows[0]["latest_price_date"] == "2026-05-26"
    assert rows[0]["account_equity"] == "2021.5"
    assert rows[0]["audit_status"] == "pass"
    assert rows[0]["audit_warning_count"] == "1"
    assert rows[0]["backtest_max_drawdown"] == "-0.19"


def test_append_history_rewrites_old_schema_with_audit_columns(tmp_path):
    history = tmp_path / "history.csv"
    history.write_text(
        "run_id,local_time,mode,latest_price_date,account_equity\n"
        "old,2026-05-27T08:00:00-05:00,intraday-monitor,2026-05-27,2000\n"
    )
    summary = {
        "run_id": "new",
        "local_time": "2026-05-27T09:00:00-05:00",
        "mode": "intraday-monitor",
        "metrics": {
            "monitor_latest_price_date": "2026-05-27",
            "account_equity": 2010,
            "audit_status": "pass",
            "audit_error_count": 0,
            "audit_warning_count": 0,
        },
    }

    append_history(history, summary)

    with history.open() as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == HISTORY_COLUMNS
    assert rows[0]["run_id"] == "old"
    assert rows[0]["audit_status"] == ""
    assert rows[1]["run_id"] == "new"
    assert rows[1]["audit_status"] == "pass"


def _arg_value(command: tuple[str, ...], flag: str) -> str:
    return command[command.index(flag) + 1]


def _args(root=None) -> Namespace:
    return _args_for_root(root)


def _args_for_root(root) -> Namespace:
    def path(value: str) -> str:
        if root is None:
            return value
        return str(root / value)

    return Namespace(
        universe=path("data/reference/tracked_universe.csv"),
        prices=path("data/raw/yahoo_prices.csv"),
        research_frame=path("data/processed/research_frame.csv"),
        paper_prefix=path("reports/paper_portfolio"),
        paper_ledger=path("data/paper/paper_trading_ledger.csv"),
        paper_summary=path("reports/paper_account_summary.json"),
        monitor_prefix=path("reports/paper_monitor"),
        audit_prefix=path("reports/paper_realism_audit"),
        paper_backtest_prefix=path("reports/paper_style_backtest"),
        visibility_prefix=path("reports/model_visibility"),
        actionability_prefix=path("reports/paper_actionability"),
        tracking_root=path("reports/daily_runs"),
        history=path("reports/daily_runs/history.csv"),
        start="2020-01-01",
        horizons="5,20",
        paper_min_score=0.02,
        paper_allow_fractional_shares=True,
        skip_backtest=False,
        skip_visibility=False,
        skip_audit=False,
        no_snapshot=False,
        include_large_artifacts=False,
        rerun_after_close_heavy=False,
    )
