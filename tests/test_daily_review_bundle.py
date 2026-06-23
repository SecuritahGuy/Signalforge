from argparse import Namespace

from scripts.run_daily_review_bundle import build_review_steps


def test_build_review_steps_runs_lightweight_bundle_by_default():
    names = [step.name for step in build_review_steps(_args())]

    assert names == [
        "run-paper-realism-audit",
        "rebuild-daily-history",
        "run-paper-tracking-report",
    ]


def test_build_review_steps_can_include_rd_experiments():
    args = _args()
    args.include_rd = True

    names = [step.name for step in build_review_steps(args)]

    assert names[-1] == "run-rd-experiments"


def test_build_review_steps_can_fail_on_audit_errors():
    args = _args()
    args.fail_on_audit_errors = True

    audit_step = build_review_steps(args)[0]

    assert "--fail-on-errors" in audit_step.command


def _args() -> Namespace:
    return Namespace(
        ledger="data/paper/paper_trading_ledger.csv",
        prices="data/raw/yahoo_prices.csv",
        account_summary="reports/paper_account_summary.json",
        tracking_root="reports/daily_runs",
        history="reports/daily_runs/history.csv",
        backtest_summary="reports/paper_style_backtest_summary.json",
        audit_prefix="reports/paper_realism_audit",
        tracking_prefix="reports/paper_tracking",
        rd_prefix="reports/rd",
        research_frame="data/processed/research_frame.csv",
        predictions="reports/exec_top_experiment_min_score_001_predictions.csv",
        include_rd=False,
        fail_on_audit_errors=False,
    )
