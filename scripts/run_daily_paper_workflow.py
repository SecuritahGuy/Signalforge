from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time as time_module
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    command: tuple[str, ...]


HISTORY_COLUMNS = [
    "run_id",
    "local_time",
    "mode",
    "latest_price_date",
    "account_equity",
    "account_cash",
    "account_realized_pnl",
    "account_unrealized_pnl",
    "account_open_positions",
    "account_planned_orders",
    "monitor_waiting_for_fill",
    "monitor_exit_pending_positions",
    "monitor_hold_positions",
    "audit_status",
    "audit_error_count",
    "audit_warning_count",
    "portfolio_planned_order_count",
    "portfolio_planned_estimated_entry_value",
    "backtest_ending_equity",
    "backtest_total_return",
    "backtest_sharpe",
    "backtest_max_drawdown",
    "backtest_closed_win_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the right SignalForge paper workflow for the current time of day."
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "reconcile", "intraday-monitor", "after-close"),
        default="auto",
        help=(
            "auto chooses reconcile before market open, intraday-monitor during regular "
            "hours, and after-close afterward."
        ),
    )
    parser.add_argument("--timezone", default="America/Chicago")
    parser.add_argument("--market-open-time", default="08:30")
    parser.add_argument("--after-close-time", default="15:45")
    parser.add_argument("--universe", default="data/reference/tracked_universe.csv")
    parser.add_argument("--prices", default="data/raw/yahoo_prices.csv")
    parser.add_argument("--research-frame", default="data/processed/research_frame.csv")
    parser.add_argument("--paper-prefix", default="reports/paper_portfolio")
    parser.add_argument("--paper-ledger", default="data/paper/paper_trading_ledger.csv")
    parser.add_argument("--paper-summary", default="reports/paper_account_summary.json")
    parser.add_argument("--paper-exit-rules-config", default="config/paper.yaml")
    parser.add_argument("--monitor-prefix", default="reports/paper_monitor")
    parser.add_argument("--audit-prefix", default="reports/paper_realism_audit")
    parser.add_argument("--paper-backtest-prefix", default="reports/paper_style_backtest")
    parser.add_argument("--visibility-prefix", default="reports/model_visibility")
    parser.add_argument("--actionability-prefix", default="reports/paper_actionability")
    parser.add_argument("--tracking-root", default="reports/daily_runs")
    parser.add_argument("--history", default="reports/daily_runs/history.csv")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--horizons", default="5,20")
    parser.add_argument(
        "--paper-min-score",
        type=float,
        default=0.02,
        help=(
            "Minimum model score used for daily paper order generation, ledger appends, "
            "paper-style backtest, and model visibility."
        ),
    )
    parser.add_argument(
        "--paper-allow-fractional-shares",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow fractional paper shares so small accounts can express high-confidence, "
            "high-priced signals. Use --no-paper-allow-fractional-shares for whole shares."
        ),
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-minutes", type=float, default=30.0)
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--skip-visibility", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--no-snapshot", action="store_true")
    parser.add_argument("--include-large-artifacts", action="store_true")
    parser.add_argument(
        "--rerun-after-close-heavy",
        action="store_true",
        help="Rerun after-close order generation, backtest, and visibility even if already run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    while True:
        run_once(args)
        if not args.loop or args.dry_run:
            break
        sleep_seconds = max(args.interval_minutes, 1.0) * 60.0
        print(f"\nSleeping {sleep_seconds / 60.0:.1f} minutes before next workflow cycle.")
        time_module.sleep(sleep_seconds)


def run_once(args: argparse.Namespace) -> str:
    now = datetime.now(ZoneInfo(args.timezone))
    resolved_mode = resolve_mode(
        requested_mode=args.mode,
        now=now,
        market_open_time=parse_clock_time(args.market_open_time),
        after_close_time=parse_clock_time(args.after_close_time),
    )
    print(f"SignalForge daily paper workflow mode: {resolved_mode}")
    print(f"Local time: {now.isoformat(timespec='seconds')}")

    if resolved_mode == "after-close" and not args.rerun_after_close_heavy:
        common_steps = build_steps(args, mode="reconcile")
        for step in common_steps:
            run_step(step, dry_run=args.dry_run)
        latest_price_date = latest_price_date_from_prices(Path(args.prices))
        include_after_close_heavy = not after_close_heavy_already_ran(
            Path(args.history),
            latest_price_date=latest_price_date,
        )
        if not include_after_close_heavy:
            print(
                "\nAfter-close heavy workflow already ran for "
                f"{latest_price_date}; running monitor/audit only."
            )
        steps = build_steps(
            args,
            mode=resolved_mode,
            include_after_close_heavy=include_after_close_heavy,
        )
        remaining_steps = steps[len(common_steps) :]
    else:
        steps = build_steps(args, mode=resolved_mode)
        remaining_steps = steps

    for step in remaining_steps:
        run_step(step, dry_run=args.dry_run)
    if not args.dry_run and not args.no_snapshot:
        snapshot_path = snapshot_run(args, resolved_mode=resolved_mode, steps=steps, now=now)
        print(f"\nTracked run snapshot: {snapshot_path}")
    return resolved_mode


def run_step(step: WorkflowStep, *, dry_run: bool) -> None:
    command_text = " ".join(step.command)
    print(f"\n[{step.name}] {command_text}")
    if not dry_run:
        subprocess.run(step.command, check=True, cwd=Path(__file__).parents[1])


def resolve_mode(
    *,
    requested_mode: str,
    now: datetime,
    market_open_time: time,
    after_close_time: time,
) -> str:
    if requested_mode != "auto":
        return requested_mode
    if market_open_time <= now.time() < after_close_time:
        return "intraday-monitor"
    if now.time() >= after_close_time:
        return "after-close"
    return "reconcile"


def parse_clock_time(raw: str) -> time:
    hour, minute = raw.split(":", 1)
    return time(hour=int(hour), minute=int(minute))


def build_steps(
    args: argparse.Namespace,
    *,
    mode: str,
    include_after_close_heavy: bool = True,
) -> list[WorkflowStep]:
    exit_rules_config = getattr(args, "paper_exit_rules_config", "config/paper.yaml")
    steps = [
        WorkflowStep(
            "refresh-yahoo-prices",
            (
                sys.executable,
                "scripts/download_yahoo_prices.py",
                "--universe",
                args.universe,
                "--start",
                args.start,
                "--output",
                args.prices,
            ),
        ),
        WorkflowStep(
            "rebuild-research-frame",
            (
                sys.executable,
                "scripts/build_research_frame.py",
                "--prices",
                args.prices,
                "--universe",
                args.universe,
                "--horizons",
                args.horizons,
                "--output",
                args.research_frame,
            ),
        ),
        WorkflowStep(
            "reconcile-existing-paper-ledger",
            (
                sys.executable,
                "scripts/update_paper_ledger.py",
                "--ledger",
                args.paper_ledger,
                "--prices",
                args.prices,
                "--summary-output",
                args.paper_summary,
                "--exit-rules-config",
                exit_rules_config,
                "--skip-add-plans",
            ),
        ),
    ]
    if mode == "intraday-monitor":
        steps.append(_monitor_step(args))
        steps.append(_actionability_step(args))
        if not args.skip_audit:
            steps.append(_audit_step(args))
    if mode == "after-close" and include_after_close_heavy:
        steps.extend(
            [
                WorkflowStep(
                    "generate-paper-portfolio",
                    (
                        sys.executable,
                        "scripts/run_paper_portfolio.py",
                        "--research-frame",
                        args.research_frame,
                        "--output-prefix",
                        args.paper_prefix,
                        "--min-score",
                        str(args.paper_min_score),
                        *(_fractional_share_args(args)),
                    ),
                ),
                _actionability_step(args),
                WorkflowStep(
                    "append-new-paper-plans",
                    (
                        sys.executable,
                        "scripts/update_paper_ledger.py",
                        "--ledger",
                        args.paper_ledger,
                        "--prices",
                        args.prices,
                        "--planned-orders",
                        f"{args.paper_prefix}_order_ledger.csv",
                        "--summary-output",
                        args.paper_summary,
                        "--exit-rules-config",
                        exit_rules_config,
                        "--min-score",
                        str(args.paper_min_score),
                        *(_fractional_share_args(args)),
                    ),
                ),
                _monitor_step(args),
            ]
        )
    if mode == "after-close" and not include_after_close_heavy:
        steps.append(_monitor_step(args))
        steps.append(_actionability_step(args))
    if mode == "after-close":
        if not args.skip_audit:
            steps.append(_audit_step(args))
        if include_after_close_heavy and not args.skip_backtest:
            steps.append(
                WorkflowStep(
                    "run-paper-style-backtest",
                    (
                        sys.executable,
                        "scripts/run_paper_style_backtest.py",
                        "--prices",
                        args.prices,
                        "--output-prefix",
                        args.paper_backtest_prefix,
                        "--min-score",
                        str(args.paper_min_score),
                        *(_fractional_share_args(args)),
                    ),
                )
            )
        if include_after_close_heavy and not args.skip_visibility:
            steps.append(
                WorkflowStep(
                    "run-model-visibility-report",
                    (
                        sys.executable,
                        "scripts/run_visibility_report.py",
                        "--output-prefix",
                        args.visibility_prefix,
                        "--min-score",
                        str(args.paper_min_score),
                    ),
                )
            )
    return steps


def _fractional_share_args(args: argparse.Namespace) -> tuple[str, ...]:
    if getattr(args, "paper_allow_fractional_shares", False):
        return ("--allow-fractional-shares",)
    return ()


def _monitor_step(args: argparse.Namespace) -> WorkflowStep:
    exit_rules_config = getattr(args, "paper_exit_rules_config", "config/paper.yaml")
    return WorkflowStep(
        "write-paper-monitor-report",
        (
            sys.executable,
            "scripts/run_paper_monitor.py",
            "--ledger",
            args.paper_ledger,
            "--prices",
            args.prices,
            "--output-prefix",
            args.monitor_prefix,
            "--exit-rules-config",
            exit_rules_config,
        ),
    )


def _actionability_step(args: argparse.Namespace) -> WorkflowStep:
    return WorkflowStep(
        "write-paper-actionability-report",
        (
            sys.executable,
            "scripts/run_paper_actionability_report.py",
            "--daily-orders",
            f"{args.paper_prefix}_order_ledger.csv",
            "--ledger",
            args.paper_ledger,
            "--account-summary",
            args.paper_summary,
            "--prices",
            args.prices,
            "--output-prefix",
            args.actionability_prefix,
        ),
    )


def _audit_step(args: argparse.Namespace) -> WorkflowStep:
    return WorkflowStep(
        "run-paper-realism-audit",
        (
            sys.executable,
            "scripts/run_paper_realism_audit.py",
            "--ledger",
            args.paper_ledger,
            "--prices",
            args.prices,
            "--account-summary",
            args.paper_summary,
            "--output-prefix",
            args.audit_prefix,
        ),
    )


def snapshot_run(
    args: argparse.Namespace,
    *,
    resolved_mode: str,
    steps: list[WorkflowStep],
    now: datetime,
) -> Path:
    run_id = now.strftime("%Y%m%dT%H%M%S%z")
    run_dir = Path(args.tracking_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    artifacts = copy_tracking_artifacts(args, run_dir=run_dir)
    metrics = collect_tracking_metrics(args)
    summary = {
        "run_id": run_id,
        "mode": resolved_mode,
        "local_time": now.isoformat(timespec="seconds"),
        "steps": [step.name for step in steps],
        "artifacts": artifacts,
        "metrics": metrics,
    }
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    append_history(Path(args.history), summary)
    return run_dir


def copy_tracking_artifacts(args: argparse.Namespace, *, run_dir: Path) -> dict[str, str]:
    copied = {}
    for artifact_name, source in tracking_artifacts(args):
        source_path = Path(source)
        if not source_path.exists():
            continue
        destination = run_dir / source_path.name
        shutil.copy2(source_path, destination)
        copied[artifact_name] = str(destination)
    return copied


def tracking_artifacts(args: argparse.Namespace) -> list[tuple[str, str]]:
    artifacts = [
        ("paper_ledger", args.paper_ledger),
        ("paper_account_summary", args.paper_summary),
        ("paper_portfolio_summary", f"{args.paper_prefix}_summary.json"),
        ("paper_portfolio_orders", f"{args.paper_prefix}_order_ledger.csv"),
        ("paper_portfolio_watchlist", f"{args.paper_prefix}_watchlist.csv"),
        ("paper_actionability_summary", f"{args.actionability_prefix}_summary.json"),
        ("paper_actionability_candidates", f"{args.actionability_prefix}_candidates.csv"),
        ("paper_actionability_report", f"{args.actionability_prefix}_report.md"),
        ("paper_monitor_summary", f"{args.monitor_prefix}_summary.json"),
        ("paper_monitor_positions", f"{args.monitor_prefix}_positions.csv"),
        ("paper_monitor_report", f"{args.monitor_prefix}_report.md"),
        ("paper_realism_audit_summary", f"{args.audit_prefix}_summary.json"),
        ("paper_realism_audit_report", f"{args.audit_prefix}_report.md"),
        ("paper_backtest_summary", f"{args.paper_backtest_prefix}_summary.json"),
        ("paper_backtest_report", f"{args.paper_backtest_prefix}_report.md"),
        ("paper_backtest_daily_equity", f"{args.paper_backtest_prefix}_daily_equity.csv"),
        ("model_visibility_summary", f"{args.visibility_prefix}_summary.md"),
        ("model_visibility_score_buckets", f"{args.visibility_prefix}_score_buckets.csv"),
        ("model_visibility_prediction_drift", f"{args.visibility_prefix}_prediction_drift.csv"),
        (
            "model_visibility_pick_explanations",
            f"{args.visibility_prefix}_paper_pick_explanations.csv",
        ),
    ]
    if args.include_large_artifacts:
        artifacts.extend(
            [
                ("prices", args.prices),
                ("research_frame", args.research_frame),
                ("paper_backtest_ledger", f"{args.paper_backtest_prefix}_ledger.csv"),
            ]
        )
    return artifacts


def collect_tracking_metrics(args: argparse.Namespace) -> dict[str, object]:
    metrics: dict[str, object] = {}
    merge_metric_file(metrics, Path(args.paper_summary), prefix="account")
    merge_metric_file(metrics, Path(f"{args.monitor_prefix}_summary.json"), prefix="monitor")
    merge_metric_file(metrics, Path(f"{args.audit_prefix}_summary.json"), prefix="audit")
    merge_metric_file(metrics, Path(f"{args.paper_prefix}_summary.json"), prefix="portfolio")
    merge_metric_file(
        metrics, Path(f"{args.actionability_prefix}_summary.json"), prefix="actionability"
    )
    merge_metric_file(
        metrics, Path(f"{args.paper_backtest_prefix}_summary.json"), prefix="backtest"
    )
    return metrics


def merge_metric_file(metrics: dict[str, object], path: Path, *, prefix: str) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text())
    for key, value in data.items():
        if isinstance(value, dict | list):
            continue
        metrics[f"{prefix}_{key}"] = value


def append_history(history_path: Path, summary: dict[str, object]) -> None:
    row = build_history_row(summary)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if not history_path.exists():
        _write_history_rows(history_path, [row])
        return

    with history_path.open(newline="") as file:
        reader = csv.DictReader(file)
        existing_rows = [
            {column: existing.get(column, "") for column in HISTORY_COLUMNS}
            for existing in reader
        ]
        existing_fields = reader.fieldnames or []

    if existing_fields != HISTORY_COLUMNS:
        _write_history_rows(history_path, [*existing_rows, row])
        return

    with history_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS)
        writer.writerow(row)


def build_history_row(summary: dict[str, object]) -> dict[str, object]:
    metrics = summary["metrics"]
    assert isinstance(metrics, dict)
    return {
        "run_id": summary["run_id"],
        "local_time": summary["local_time"],
        "mode": summary["mode"],
        "latest_price_date": metrics.get("monitor_latest_price_date")
        or metrics.get("portfolio_as_of_date"),
        "account_equity": metrics.get("account_equity"),
        "account_cash": metrics.get("account_cash"),
        "account_realized_pnl": metrics.get("account_realized_pnl"),
        "account_unrealized_pnl": metrics.get("account_unrealized_pnl"),
        "account_open_positions": metrics.get("account_open_positions"),
        "account_planned_orders": metrics.get("account_planned_orders"),
        "monitor_waiting_for_fill": metrics.get("monitor_waiting_for_fill"),
        "monitor_exit_pending_positions": metrics.get("monitor_exit_pending_positions"),
        "monitor_hold_positions": metrics.get("monitor_hold_positions"),
        "audit_status": metrics.get("audit_status"),
        "audit_error_count": metrics.get("audit_error_count"),
        "audit_warning_count": metrics.get("audit_warning_count"),
        "portfolio_planned_order_count": metrics.get("portfolio_planned_order_count"),
        "portfolio_planned_estimated_entry_value": metrics.get(
            "portfolio_planned_estimated_entry_value"
        ),
        "backtest_ending_equity": metrics.get("backtest_ending_equity"),
        "backtest_total_return": metrics.get("backtest_total_return"),
        "backtest_sharpe": metrics.get("backtest_sharpe"),
        "backtest_max_drawdown": metrics.get("backtest_max_drawdown"),
        "backtest_closed_win_rate": metrics.get("backtest_closed_win_rate"),
    }


def _write_history_rows(history_path: Path, rows: list[dict[str, object]]) -> None:
    with history_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def latest_price_date_from_prices(prices_path: Path) -> str | None:
    if not prices_path.exists():
        return None
    with prices_path.open(newline="") as file:
        reader = csv.DictReader(file)
        dates = [row.get("date", "") for row in reader if row.get("date")]
    if not dates:
        return None
    return max(date[:10] for date in dates)


def after_close_heavy_already_ran(
    history_path: Path,
    *,
    latest_price_date: str | None,
) -> bool:
    if latest_price_date is None or not history_path.exists():
        return False
    with history_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("mode") != "after-close":
                continue
            if row.get("latest_price_date") != latest_price_date:
                continue
            if row.get("portfolio_planned_order_count") not in (None, ""):
                return True
    return False


if __name__ == "__main__":
    main()
