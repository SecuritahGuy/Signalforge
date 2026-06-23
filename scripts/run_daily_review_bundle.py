from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReviewStep:
    name: str
    command: tuple[str, ...]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the daily SignalForge review bundle from existing artifacts."
    )
    parser.add_argument("--ledger", default="data/paper/paper_trading_ledger.csv")
    parser.add_argument("--prices", default="data/raw/yahoo_prices.csv")
    parser.add_argument("--account-summary", default="reports/paper_account_summary.json")
    parser.add_argument("--tracking-root", default="reports/daily_runs")
    parser.add_argument("--history", default="reports/daily_runs/history.csv")
    parser.add_argument("--backtest-summary", default="reports/paper_style_backtest_summary.json")
    parser.add_argument("--audit-prefix", default="reports/paper_realism_audit")
    parser.add_argument("--tracking-prefix", default="reports/paper_tracking")
    parser.add_argument("--rd-prefix", default="reports/rd")
    parser.add_argument("--research-frame", default="data/processed/research_frame.csv")
    parser.add_argument(
        "--predictions",
        default="reports/exec_top_experiment_min_score_001_predictions.csv",
    )
    parser.add_argument("--include-rd", action="store_true")
    parser.add_argument("--fail-on-audit-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for step in build_review_steps(args):
        command_text = " ".join(step.command)
        print(f"\n[{step.name}] {command_text}")
        if not args.dry_run:
            subprocess.run(step.command, check=True, cwd=Path(__file__).parents[1])


def build_review_steps(args: argparse.Namespace) -> list[ReviewStep]:
    audit_command = [
        sys.executable,
        "scripts/run_paper_realism_audit.py",
        "--ledger",
        args.ledger,
        "--prices",
        args.prices,
        "--account-summary",
        args.account_summary,
        "--output-prefix",
        args.audit_prefix,
    ]
    if args.fail_on_audit_errors:
        audit_command.append("--fail-on-errors")

    steps = [
        ReviewStep("run-paper-realism-audit", tuple(audit_command)),
        ReviewStep(
            "rebuild-daily-history",
            (
                sys.executable,
                "scripts/rebuild_daily_history.py",
                "--tracking-root",
                args.tracking_root,
                "--history",
                args.history,
            ),
        ),
        ReviewStep(
            "run-paper-tracking-report",
            (
                sys.executable,
                "scripts/run_paper_tracking_report.py",
                "--history",
                args.history,
                "--ledger",
                args.ledger,
                "--backtest-summary",
                args.backtest_summary,
                "--output-prefix",
                args.tracking_prefix,
            ),
        ),
    ]
    if args.include_rd:
        steps.append(
            ReviewStep(
                "run-rd-experiments",
                (
                    sys.executable,
                    "scripts/run_rd_experiments.py",
                    "--research-frame",
                    args.research_frame,
                    "--predictions",
                    args.predictions,
                    "--output-prefix",
                    args.rd_prefix,
                ),
            )
        )
    return steps


if __name__ == "__main__":
    main()
