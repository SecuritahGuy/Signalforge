import json
from argparse import Namespace

from scripts.run_daily_operations import (
    dashboard_sync_steps,
    intraday_risk_steps,
    paper_workflow_steps,
    promotion_plan_steps,
    should_run_symbol_discovery,
    symbol_discovery_steps,
    write_combined_review,
)


def test_should_run_symbol_discovery_skips_matching_as_of_date(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"as_of_date": "2026-06-01"}))

    assert not should_run_symbol_discovery(
        summary,
        latest_price_date="2026-06-01",
        rerun=False,
    )
    assert should_run_symbol_discovery(
        summary,
        latest_price_date="2026-06-02",
        rerun=False,
    )
    assert should_run_symbol_discovery(
        summary,
        latest_price_date="2026-06-01",
        rerun=True,
    )


def test_operations_steps_use_separate_paper_and_broad_artifacts():
    args = _args()

    paper = paper_workflow_steps(args)[0].command
    discovery = [step.command for step in symbol_discovery_steps(args)]

    assert "--universe" in paper
    assert "data/reference/tracked_universe.csv" in paper
    assert _arg_value(paper, "--paper-min-score") == "0.02"
    assert "--paper-allow-fractional-shares" in paper
    assert "data/reference/sp500_universe.csv" in discovery[0]
    assert "data/raw/sp500_yahoo_prices.csv" in discovery[1]
    assert "data/processed/sp500_research_frame.csv" in discovery[2]
    assert "reports/symbol_discovery_rd" in discovery[3]
    promotion = promotion_plan_steps(args)[0].command
    assert "scripts/promote_discovery_candidates.py" in promotion
    assert "reports/symbol_discovery_rd/promotion_candidates.csv" in promotion
    assert "--approve" not in promotion


def test_promotion_step_can_auto_approve_with_thresholds():
    args = _args(
        auto_approve_discovery_promotions=True,
        promotion_max_symbols=3,
        promotion_min_discovery_score=75.0,
        promotion_min_lane_count=2,
        promotion_min_appearances=4,
        promotion_min_monitoring_age_days=7,
        promotion_max_sector_symbols=1,
    )

    command = promotion_plan_steps(args)[0].command

    assert "--approve" in command
    assert _arg_value(command, "--max-symbols") == "3"
    assert _arg_value(command, "--min-discovery-score") == "75.0"
    assert _arg_value(command, "--min-lane-count") == "2"
    assert _arg_value(command, "--min-appearances") == "4"
    assert _arg_value(command, "--min-monitoring-age-days") == "7"
    assert _arg_value(command, "--max-sector-symbols") == "1"


def test_intraday_risk_step_is_available_for_default_operations_loop():
    args = _args()

    command = intraday_risk_steps(args)[0].command

    assert "scripts/run_intraday_risk_monitor.py" in command
    assert "--ledger" in command
    assert "data/paper/paper_trading_ledger.csv" in command
    assert "--exit-rules-config" in command
    assert "config/paper.yaml" in command


def test_dashboard_sync_step_refreshes_local_ui_bundle():
    command = dashboard_sync_steps(_args())[0].command

    assert command == ("npm", "--prefix", "web", "run", "sync")


def test_write_combined_review_uses_paper_and_discovery_summaries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reports = tmp_path / "reports"
    discovery_dir = reports / "symbol_discovery_rd"
    reports.mkdir()
    discovery_dir.mkdir()
    (reports / "paper_tracking_summary.json").write_text(
        json.dumps(
            {
                "latest_snapshot": "2026-06-01T16:00:00-05:00",
                "latest_price_date": "2026-06-01",
                "paper_equity": 2184.0,
                "paper_total_return": 0.092,
                "paper_current_drawdown": -0.001,
                "open_positions": 11,
                "closed_positions": 0,
                "latest_audit_status": "pass",
                "assessment": "too early",
            }
        )
    )
    (reports / "paper_actionability_summary.json").write_text(
        json.dumps(
            {
                "candidate_count": 94,
                "model_planned_count": 8,
                "actionable_new_order_count": 0,
                "blocked_by_active_symbol_count": 11,
            }
        )
    )
    (discovery_dir / "summary.json").write_text(
        json.dumps(
            {
                "as_of_date": "2026-06-01",
                "source_universe_count": 504,
                "eligible_after_filters": 410,
                "monitored_candidate_count": 137,
                "promotion_candidate_count": 0,
            }
        )
    )
    (discovery_dir / "candidates.csv").write_text(
        "symbol,name,sector,discovery_score,lane_count,return_20d,return_60d,promotion_blockers\n"
        "DDOG,Datadog,Information Technology,79.7,4,0.1,0.2,monitoring\n"
    )
    (reports / "symbol_discovery_promotion_plan_summary.json").write_text(
        json.dumps({"ready_to_promote_count": 2})
    )

    output = reports / "daily_ops_review.md"
    write_combined_review(output, discovery_output_dir=discovery_dir)

    text = output.read_text()
    assert "Paper Trading" in text
    assert "Symbol Discovery R&D" in text
    assert "Ready to promote: `2`" in text
    assert "DDOG" in text


def _arg_value(command: tuple[str, ...], flag: str) -> str:
    return command[command.index(flag) + 1]


def _args(**overrides) -> Namespace:
    values = dict(
        timezone="America/Chicago",
        market_open_time="08:30",
        after_close_time="15:45",
        paper_universe="data/reference/tracked_universe.csv",
        paper_prices="data/raw/yahoo_prices.csv",
        paper_research_frame="data/processed/research_frame.csv",
        paper_min_score=0.02,
        paper_allow_fractional_shares=True,
        broad_universe="data/reference/sp500_universe.csv",
        broad_prices="data/raw/sp500_yahoo_prices.csv",
        broad_research_frame="data/processed/sp500_research_frame.csv",
        discovery_output_dir="reports/symbol_discovery_rd",
        combined_review="reports/daily_ops_review.md",
        start="2020-01-01",
        horizons="5,20",
        auto_approve_discovery_promotions=False,
        promotion_max_symbols=5,
        promotion_min_discovery_score=60.0,
        promotion_min_lane_count=0,
        promotion_min_appearances=0,
        promotion_min_monitoring_age_days=0,
        promotion_max_sector_symbols=None,
        rerun_after_close_heavy=False,
        skip_dashboard_sync=False,
        dry_run=False,
    )
    values.update(overrides)
    return Namespace(
        **values,
    )
