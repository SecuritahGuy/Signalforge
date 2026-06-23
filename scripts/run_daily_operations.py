from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time as time_module
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.run_daily_paper_workflow import (
    latest_price_date_from_prices,
    parse_clock_time,
    resolve_mode,
)


@dataclass(frozen=True)
class OperationStep:
    name: str
    command: tuple[str, ...]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the all-day SignalForge paper and symbol-discovery operations loop."
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-minutes", type=float, default=30.0)
    parser.add_argument("--timezone", default="America/Chicago")
    parser.add_argument("--market-open-time", default="08:30")
    parser.add_argument("--after-close-time", default="15:45")
    parser.add_argument("--paper-universe", default="data/reference/tracked_universe.csv")
    parser.add_argument("--paper-ledger", default="data/paper/paper_trading_ledger.csv")
    parser.add_argument("--paper-prices", default="data/raw/yahoo_prices.csv")
    parser.add_argument("--paper-research-frame", default="data/processed/research_frame.csv")
    parser.add_argument("--paper-exit-rules-config", default="config/paper.yaml")
    parser.add_argument("--paper-min-score", type=float, default=0.02)
    parser.add_argument(
        "--paper-allow-fractional-shares",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--broad-universe", default="data/reference/sp500_universe.csv")
    parser.add_argument("--broad-prices", default="data/raw/sp500_yahoo_prices.csv")
    parser.add_argument("--broad-research-frame", default="data/processed/sp500_research_frame.csv")
    parser.add_argument("--discovery-output-dir", default="reports/symbol_discovery_rd")
    parser.add_argument("--combined-review", default="reports/daily_ops_review.md")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--horizons", default="5,20")
    parser.add_argument("--skip-symbol-discovery", action="store_true")
    parser.add_argument("--rerun-symbol-discovery", action="store_true")
    parser.add_argument("--auto-approve-discovery-promotions", action="store_true")
    parser.add_argument("--promotion-max-symbols", type=int, default=5)
    parser.add_argument("--promotion-min-discovery-score", type=float, default=60.0)
    parser.add_argument("--promotion-min-lane-count", type=int, default=0)
    parser.add_argument("--promotion-min-appearances", type=int, default=0)
    parser.add_argument("--promotion-min-monitoring-age-days", type=int, default=0)
    parser.add_argument("--promotion-max-sector-symbols", type=int, default=None)
    parser.add_argument("--rerun-after-close-heavy", action="store_true")
    parser.add_argument("--skip-intraday-risk", action="store_true")
    parser.add_argument("--skip-dashboard-sync", action="store_true")
    parser.add_argument("--intraday-risk-write-ledger", action="store_true")
    parser.add_argument("--intraday-interval", default="1m")
    parser.add_argument("--intraday-period", default="1d")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    while True:
        run_once(args)
        if not args.loop or args.dry_run:
            break
        sleep_seconds = max(args.interval_minutes, 1.0) * 60.0
        print(f"\nSleeping {sleep_seconds / 60.0:.1f} minutes before next operations cycle.")
        time_module.sleep(sleep_seconds)


def run_once(args: argparse.Namespace) -> str:
    now = datetime.now(ZoneInfo(args.timezone))
    mode = resolve_mode(
        requested_mode="auto",
        now=now,
        market_open_time=parse_clock_time(args.market_open_time),
        after_close_time=parse_clock_time(args.after_close_time),
    )
    print(f"SignalForge daily operations mode: {mode}")
    print(f"Local time: {now.isoformat(timespec='seconds')}")

    for step in paper_workflow_steps(args):
        run_step(step, dry_run=args.dry_run)
    if not getattr(args, "skip_intraday_risk", False) and mode == "intraday-monitor":
        for step in intraday_risk_steps(args):
            run_step(step, dry_run=args.dry_run)

    latest_price_date = latest_price_date_from_prices(Path(args.paper_prices))
    if mode == "after-close" and not args.skip_symbol_discovery:
        if should_run_symbol_discovery(
            Path(args.discovery_output_dir) / "summary.json",
            latest_price_date=latest_price_date,
            rerun=args.rerun_symbol_discovery,
        ):
            for step in symbol_discovery_steps(args):
                run_step(step, dry_run=args.dry_run)
        else:
            print(
                "\nSymbol discovery already ran for "
                f"{latest_price_date}; skipping broad discovery refresh."
            )
        for step in promotion_plan_steps(args):
            run_step(step, dry_run=args.dry_run)
        for step in review_steps(args):
            run_step(step, dry_run=args.dry_run)
        if not args.dry_run:
            write_combined_review(
                Path(args.combined_review),
                discovery_output_dir=Path(args.discovery_output_dir),
            )
            print(f"\nWrote combined daily operations review to {args.combined_review}")

    if not getattr(args, "skip_dashboard_sync", False):
        for step in dashboard_sync_steps(args):
            run_step(step, dry_run=args.dry_run)

    return mode


def paper_workflow_steps(args: argparse.Namespace) -> list[OperationStep]:
    paper_ledger = getattr(args, "paper_ledger", "data/paper/paper_trading_ledger.csv")
    paper_exit_rules_config = getattr(args, "paper_exit_rules_config", "config/paper.yaml")
    command = [
        sys.executable,
        "scripts/run_daily_paper_workflow.py",
        "--mode",
        "auto",
        "--timezone",
        args.timezone,
        "--market-open-time",
        args.market_open_time,
        "--after-close-time",
        args.after_close_time,
        "--universe",
        args.paper_universe,
        "--prices",
        args.paper_prices,
        "--paper-ledger",
        paper_ledger,
        "--paper-exit-rules-config",
        paper_exit_rules_config,
        "--research-frame",
        args.paper_research_frame,
        "--start",
        args.start,
        "--horizons",
        args.horizons,
        "--paper-min-score",
        str(args.paper_min_score),
    ]
    if args.paper_allow_fractional_shares:
        command.append("--paper-allow-fractional-shares")
    else:
        command.append("--no-paper-allow-fractional-shares")
    if args.rerun_after_close_heavy:
        command.append("--rerun-after-close-heavy")
    if args.dry_run:
        command.append("--dry-run")
    return [OperationStep("run-paper-workflow-cycle", tuple(command))]


def intraday_risk_steps(args: argparse.Namespace) -> list[OperationStep]:
    paper_ledger = getattr(args, "paper_ledger", "data/paper/paper_trading_ledger.csv")
    paper_exit_rules_config = getattr(args, "paper_exit_rules_config", "config/paper.yaml")
    command = [
        sys.executable,
        "scripts/run_intraday_risk_monitor.py",
        "--ledger",
        paper_ledger,
        "--daily-prices",
        args.paper_prices,
        "--exit-rules-config",
        paper_exit_rules_config,
        "--interval",
        getattr(args, "intraday_interval", "1m"),
        "--period",
        getattr(args, "intraday_period", "1d"),
    ]
    if getattr(args, "intraday_risk_write_ledger", False):
        command.append("--write-ledger")
    return [OperationStep("run-intraday-risk-monitor", tuple(command))]


def dashboard_sync_steps(args: argparse.Namespace) -> list[OperationStep]:
    return [
        OperationStep(
            "sync-dashboard-data",
            (
                "npm",
                "--prefix",
                "web",
                "run",
                "sync",
            ),
        )
    ]


def symbol_discovery_steps(args: argparse.Namespace) -> list[OperationStep]:
    return [
        OperationStep(
            "build-broad-universe",
            (
                sys.executable,
                "scripts/build_broad_universe.py",
                "--source",
                "sp500",
                "--output",
                args.broad_universe,
            ),
        ),
        OperationStep(
            "download-broad-yahoo-prices",
            (
                sys.executable,
                "scripts/download_yahoo_prices.py",
                "--universe",
                args.broad_universe,
                "--start",
                args.start,
                "--output",
                args.broad_prices,
            ),
        ),
        OperationStep(
            "build-broad-research-frame",
            (
                sys.executable,
                "scripts/build_research_frame.py",
                "--prices",
                args.broad_prices,
                "--universe",
                args.broad_universe,
                "--horizons",
                args.horizons,
                "--output",
                args.broad_research_frame,
            ),
        ),
        OperationStep(
            "run-symbol-discovery-rd",
            (
                sys.executable,
                "scripts/run_symbol_discovery_rd.py",
                "--research-frame",
                args.broad_research_frame,
                "--universe",
                args.broad_universe,
                "--existing-watchlist",
                args.paper_universe,
                "--output-dir",
                args.discovery_output_dir,
            ),
        ),
    ]


def review_steps(args: argparse.Namespace) -> list[OperationStep]:
    return [
        OperationStep(
            "run-paper-review-bundle",
            (
                sys.executable,
                "scripts/run_daily_review_bundle.py",
            ),
        )
    ]


def promotion_plan_steps(args: argparse.Namespace) -> list[OperationStep]:
    command = [
        sys.executable,
        "scripts/promote_discovery_candidates.py",
        "--promotion-candidates",
        str(Path(args.discovery_output_dir) / "promotion_candidates.csv"),
        "--universe",
        args.paper_universe,
        "--max-symbols",
        str(args.promotion_max_symbols),
        "--min-discovery-score",
        str(args.promotion_min_discovery_score),
        "--min-lane-count",
        str(args.promotion_min_lane_count),
        "--min-appearances",
        str(args.promotion_min_appearances),
        "--min-monitoring-age-days",
        str(args.promotion_min_monitoring_age_days),
    ]
    if args.promotion_max_sector_symbols is not None:
        command.extend(["--max-sector-symbols", str(args.promotion_max_sector_symbols)])
    if args.auto_approve_discovery_promotions:
        command.append("--approve")

    return [
        OperationStep(
            "write-discovery-promotion-plan",
            tuple(command),
        )
    ]


def should_run_symbol_discovery(
    summary_path: Path,
    *,
    latest_price_date: str | None,
    rerun: bool,
) -> bool:
    if rerun:
        return True
    if latest_price_date is None or not summary_path.exists():
        return True
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return True
    return summary.get("as_of_date") != latest_price_date


def run_step(step: OperationStep, *, dry_run: bool) -> None:
    command_text = " ".join(step.command)
    print(f"\n[{step.name}] {command_text}")
    if not dry_run:
        subprocess.run(step.command, check=True, cwd=Path(__file__).parents[1])


def write_combined_review(output_path: Path, *, discovery_output_dir: Path) -> None:
    paper = _load_json(Path("reports/paper_tracking_summary.json"))
    actionability = _load_json(Path("reports/paper_actionability_summary.json"))
    discovery = _load_json(discovery_output_dir / "summary.json")
    promotion = _load_json(Path("reports/symbol_discovery_promotion_plan_summary.json"))
    top_discovery = _top_discovery_candidates(discovery_output_dir / "candidates.csv")
    lines = [
        "# SignalForge Daily Operations Review",
        "",
        "## Paper Trading",
        "",
        f"- Latest snapshot: `{paper.get('latest_snapshot')}`",
        f"- Latest price date: `{paper.get('latest_price_date')}`",
        f"- Equity: `${float(paper.get('paper_equity', 0.0)):.2f}`",
        f"- Total return: `{float(paper.get('paper_total_return', 0.0)):.2%}`",
        f"- Current drawdown: `{float(paper.get('paper_current_drawdown', 0.0)):.2%}`",
        f"- Open positions: `{paper.get('open_positions', 0)}`",
        f"- Closed positions: `{paper.get('closed_positions', 0)}`",
        f"- Latest audit status: `{paper.get('latest_audit_status', '')}`",
        f"- Assessment: `{paper.get('assessment', '')}`",
        "",
        "## Paper Actionability",
        "",
        f"- Candidates reviewed: `{actionability.get('candidate_count', 0)}`",
        f"- Model-planned rows: `{actionability.get('model_planned_count', 0)}`",
        f"- Actionable new orders: `{actionability.get('actionable_new_order_count', 0)}`",
        f"- Blocked by active symbols: `{actionability.get('blocked_by_active_symbol_count', 0)}`",
        "",
        "## Symbol Discovery R&D",
        "",
        f"- As-of date: `{discovery.get('as_of_date')}`",
        f"- Source universe count: `{discovery.get('source_universe_count')}`",
        f"- Eligible after filters: `{discovery.get('eligible_after_filters')}`",
        f"- Monitored candidates: `{discovery.get('monitored_candidate_count')}`",
        f"- Promotion-review candidates: `{discovery.get('promotion_candidate_count')}`",
        f"- Ready to promote: `{promotion.get('ready_to_promote_count', 0)}`",
        "",
        "## Top Discovery Candidates",
        "",
        top_discovery,
        "",
        "## Source Reports",
        "",
        "- `reports/paper_tracking_report.md`",
        "- `reports/paper_actionability_report.md`",
        f"- `{discovery_output_dir / 'report.md'}`",
        "- `reports/symbol_discovery_promotion_plan_report.md`",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _top_discovery_candidates(path: Path, *, count: int = 10) -> str:
    if not path.exists():
        return "_No discovery candidates file._"
    import pandas as pd

    frame = pd.read_csv(path).head(count)
    if frame.empty:
        return "_No rows._"
    columns = [
        "symbol",
        "name",
        "sector",
        "discovery_score",
        "lane_count",
        "return_20d",
        "return_60d",
        "promotion_blockers",
    ]
    display = frame.loc[:, [column for column in columns if column in frame]].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    rows = [list(display.columns), ["---"] * len(display.columns)]
    rows.extend(display.astype(str).values.tolist())
    return "\n".join("| " + " | ".join(str(value) for value in row) + " |" for row in rows)


if __name__ == "__main__":
    main()
