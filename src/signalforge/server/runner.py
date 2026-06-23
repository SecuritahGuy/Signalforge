from __future__ import annotations

import asyncio
import os
import sys
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

from signalforge.server.models import RunState, ScriptStatus


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_runs: dict[str, RunState] = {}
_queues: dict[str, asyncio.Queue[str | None]] = {}

SCRIPT_REGISTRY: list[dict] = [
    # ── Paper management ──────────────────────────────────────────────
    {
        "name": "update_paper_ledger",
        "category": "paper",
        "description": "Reconcile fills/exits and append new planned orders to the paper ledger.",
        "path": "scripts/update_paper_ledger.py",
        "args": [
            {"name": "--ledger", "default": "data/paper/paper_trading_ledger.csv"},
            {"name": "--prices", "default": "data/raw/yahoo_prices.csv"},
            {"name": "--skip-add-plans", "default": False, "type": "bool"},
        ],
    },
    {
        "name": "run_paper_monitor",
        "category": "paper",
        "description": "Mark open paper positions and write a monitoring report (no buys/sells).",
        "path": "scripts/run_paper_monitor.py",
    },
    {
        "name": "run_paper_realism_audit",
        "category": "paper",
        "description": "Audit the paper account for realism and ledger consistency.",
        "path": "scripts/run_paper_realism_audit.py",
    },
    {
        "name": "run_paper_tracking_report",
        "category": "paper",
        "description": "Summarize paper-trading history against the research backtest.",
        "path": "scripts/run_paper_tracking_report.py",
    },
    {
        "name": "run_paper_actionability_report",
        "category": "paper",
        "description": "Explain which daily paper candidates are actionable after live constraints.",
        "path": "scripts/run_paper_actionability_report.py",
    },
    {
        "name": "run_paper_portfolio",
        "category": "paper",
        "description": "Generate a daily paper buy watchlist and order ledger from a trained model.",
        "path": "scripts/run_paper_portfolio.py",
        "args": [
            {"name": "--research-frame", "default": "data/processed/research_frame.csv"},
            {"name": "--output-prefix", "default": "reports/paper_portfolio"},
            {"name": "--as-of-date", "default": ""},
            {"name": "--min-score", "default": 0.01, "type": "float"},
        ],
    },
    {
        "name": "compact_paper_ledger",
        "category": "paper",
        "description": "Remove skipped-only rows from the persistent paper ledger.",
        "path": "scripts/compact_paper_ledger.py",
        "args": [
            {"name": "--ledger", "default": "data/paper/paper_trading_ledger.csv"},
            {"name": "--dry-run", "default": False, "type": "bool"},
        ],
    },
    {
        "name": "run_daily_paper_workflow",
        "category": "paper",
        "description": "Run the full time-of-day-aware paper workflow (reconcile → monitor → after-close).",
        "path": "scripts/run_daily_paper_workflow.py",
        "args": [
            {"name": "--mode", "default": "auto", "choices": ["auto", "reconcile", "intraday-monitor", "after-close"]},
            {"name": "--loop", "default": False, "type": "bool"},
            {"name": "--skip-backtest", "default": False, "type": "bool"},
            {"name": "--skip-visibility", "default": False, "type": "bool"},
            {"name": "--dry-run", "default": False, "type": "bool"},
        ],
    },
    # ── Discovery & Promotion ─────────────────────────────────────────
    {
        "name": "run_symbol_discovery_rd",
        "category": "discovery",
        "description": "Run R&D symbol discovery with monitoring state tracking and promotion candidate identification.",
        "path": "scripts/run_symbol_discovery_rd.py",
    },
    {
        "name": "run_stock_discovery",
        "category": "discovery",
        "description": "Generate explainable stock discovery watchlists from a broad feature frame.",
        "path": "scripts/run_stock_discovery.py",
        "args": [
            {"name": "--research-frame", "default": "data/processed/research_frame.csv"},
            {"name": "--output-dir", "default": "reports/discovery"},
            {"name": "--top-n", "default": 25, "type": "int"},
        ],
    },
    {
        "name": "promote_discovery_candidates",
        "category": "discovery",
        "description": "Promote eligible discovery candidates into the traded paper universe.",
        "path": "scripts/promote_discovery_candidates.py",
        "args": [
            {"name": "--max-symbols", "default": 5, "type": "int"},
            {"name": "--min-discovery-score", "default": 60.0, "type": "float"},
            {"name": "--approve", "default": False, "type": "bool"},
        ],
    },
    # ── Data ───────────────────────────────────────────────────────────
    {
        "name": "download_yahoo_prices",
        "category": "data",
        "description": "Download normalized daily prices from Yahoo Finance for the tracked universe.",
        "path": "scripts/download_yahoo_prices.py",
    },
    {
        "name": "build_broad_universe",
        "category": "data",
        "description": "Build a broad SP500 or US-listed universe CSV for stock discovery.",
        "path": "scripts/build_broad_universe.py",
        "args": [
            {"name": "--source", "default": "sp500", "choices": ["sp500", "us_listed"]},
            {"name": "--output", "default": "data/reference/sp500_universe.csv"},
        ],
    },
    {
        "name": "build_research_frame",
        "category": "data",
        "description": "Build a processed model-ready research frame from prices + universe + labels/features.",
        "path": "scripts/build_research_frame.py",
        "args": [
            {"name": "--prices", "default": "data/raw/yahoo_prices.csv"},
            {"name": "--universe", "default": "data/reference/tracked_universe.csv"},
            {"name": "--horizon", "default": 5, "type": "int"},
            {"name": "--output", "default": "data/processed/research_frame.csv"},
        ],
    },
    {
        "name": "enrich_fundamentals",
        "category": "data",
        "description": "Join latest available fundamentals into an existing research frame.",
        "path": "scripts/enrich_fundamentals.py",
    },
    # ── Model & Experiments ──────────────────────────────────────────
    {
        "name": "train_baseline",
        "category": "model",
        "description": "Train the first walk-forward baseline model (ridge/elasticnet/random_forest).",
        "path": "scripts/train_baseline.py",
        "args": [
            {"name": "--research-frame", "default": "data/processed/research_frame.csv"},
            {"name": "--model-type", "default": "ridge", "choices": ["ridge", "elasticnet", "random_forest"]},
        ],
    },
    {
        "name": "run_experiments",
        "category": "model",
        "description": "Run the full baseline experiment grid (multi-model, multi-feature-set, multi-horizon).",
        "path": "scripts/run_experiments.py",
        "args": [
            {"name": "--research-frame", "default": "data/processed/research_frame.csv"},
            {"name": "--horizons", "default": "5,20"},
            {"name": "--fast", "default": False, "type": "bool"},
        ],
    },
    {
        "name": "run_rd_experiments",
        "category": "model",
        "description": "Run R&D feature-ablation and portfolio-rule experiments on top of predictions.",
        "path": "scripts/run_rd_experiments.py",
    },
    # ── Backtest ───────────────────────────────────────────────────────
    {
        "name": "run_backtest",
        "category": "backtest",
        "description": "Backtest discovery lane selections over historical dates.",
        "path": "scripts/run_backtest.py",
        "args": [
            {"name": "--research-frame", "default": "data/processed/research_frame.csv"},
            {"name": "--output", "default": "reports/discovery_backtest"},
            {"name": "--rebalance", "default": "monthly", "choices": ["monthly", "weekly"]},
        ],
    },
    {
        "name": "run_portfolio_backtest",
        "category": "backtest",
        "description": "Simulate a simple long-only portfolio from discovery backtest selections.",
        "path": "scripts/run_portfolio_backtest.py",
        "args": [
            {"name": "--backtest-trades", "default": "", "required": True},
            {"name": "--research-frame", "default": "", "required": True},
        ],
    },
    {
        "name": "run_paper_style_backtest",
        "category": "backtest",
        "description": "Run a historical backtest through the full paper-trading ledger lifecycle.",
        "path": "scripts/run_paper_style_backtest.py",
    },
    # ── Visibility & Review ──────────────────────────────────────────
    {
        "name": "run_visibility_report",
        "category": "review",
        "description": "Build model visibility artifacts (score buckets, feature importance, prediction drift).",
        "path": "scripts/run_visibility_report.py",
    },
    {
        "name": "run_daily_review_bundle",
        "category": "review",
        "description": "Run the daily review bundle from existing artifacts (audit, history, tracking).",
        "path": "scripts/run_daily_review_bundle.py",
        "args": [
            {"name": "--include-rd", "default": False, "type": "bool"},
            {"name": "--dry-run", "default": False, "type": "bool"},
        ],
    },
    # ── Risk & Operations ──────────────────────────────────────────────
    {
        "name": "evaluate_exit_rules",
        "category": "risk",
        "description": "Evaluate configured paper exit rules against open positions (read-only).",
        "path": "scripts/evaluate_exit_rules.py",
    },
    {
        "name": "run_intraday_risk_monitor",
        "category": "risk",
        "description": "Monitor open paper positions with lightweight intraday risk marks from Yahoo Finance.",
        "path": "scripts/run_intraday_risk_monitor.py",
    },
    {
        "name": "run_daily_operations",
        "category": "risk",
        "description": "Run the all-day operations loop (paper + discovery + promotion + dashboard sync).",
        "path": "scripts/run_daily_operations.py",
        "args": [
            {"name": "--loop", "default": False, "type": "bool"},
            {"name": "--interval-minutes", "default": 30.0, "type": "float"},
            {"name": "--skip-intraday-risk", "default": False, "type": "bool"},
            {"name": "--dry-run", "default": False, "type": "bool"},
        ],
    },
]

CATEGORY_ORDER = ["paper", "discovery", "data", "model", "backtest", "review", "risk"]
CATEGORY_LABELS = {
    "paper": "Paper Management",
    "discovery": "Discovery & Promotion",
    "data": "Data Pipeline",
    "model": "Model & Experiments",
    "backtest": "Backtest",
    "review": "Visibility & Review",
    "risk": "Risk & Operations",
}


def get_registered_scripts() -> list[dict]:
    """Return list of registered scripts with metadata, grouped by category."""
    by_cat: dict[str, list[dict]] = {}
    for s in SCRIPT_REGISTRY:
        cat = s.get("category", "other")
        by_cat.setdefault(cat, []).append(s)

    result = []
    for cat in CATEGORY_ORDER:
        items = by_cat.get(cat, [])
        if not items:
            continue
        result.append({
            "category": cat,
            "label": CATEGORY_LABELS.get(cat, cat),
            "scripts": [
                {
                    "name": s["name"],
                    "description": s["description"],
                    "args": s.get("args", []),
                }
                for s in items
            ],
        })
    return result


def get_run_state(run_id: str) -> RunState | None:
    return _runs.get(run_id)


def _resolve_script(name: str) -> dict | None:
    for s in SCRIPT_REGISTRY:
        if s["name"] == name:
            return s
    return None


async def run_script(
    script_name: str,
    overrides: dict[str, object] | None = None,
) -> str:
    meta = _resolve_script(script_name)
    if meta is None:
        raise ValueError(f"Unknown script: {script_name}")

    run_id = uuid.uuid4().hex[:12]
    run = RunState(run_id=run_id, script_name=script_name, status=ScriptStatus.pending)
    _runs[run_id] = run
    _queues[run_id] = asyncio.Queue()

    asyncio.create_task(_execute(run_id, meta, overrides or {}))
    return run_id


async def _execute(run_id: str, meta: dict, overrides: dict[str, object]) -> None:
    run = _runs[run_id]
    run.status = ScriptStatus.running
    run.started_at = datetime.now(timezone.utc)
    queue = _queues[run_id]

    script_path = _REPO_ROOT / meta["path"]
    cmd = [sys.executable, str(script_path)]

    for arg in meta.get("args", []):
        name = arg["name"]
        flag = name.lstrip("-")
        flag_underscore = flag.replace("-", "_")
        if flag_underscore in overrides:
            val = overrides[flag_underscore]
        elif flag in overrides:
            val = overrides[flag]
        elif "default" in arg:
            val = arg["default"]
        else:
            continue

        if arg.get("type") == "bool":
            if val:
                cmd.append(name)
        elif val == "" or val is None:
            continue
        else:
            cmd.extend([name, str(val)])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=_REPO_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            run.output.append(text)
            await queue.put(text)

        exit_code = await proc.wait()
        run.exit_code = exit_code
        run.status = ScriptStatus.completed if exit_code == 0 else ScriptStatus.failed
    except Exception as exc:
        msg = f"ERROR: {exc}"
        run.output.append(msg)
        await queue.put(msg)
        run.status = ScriptStatus.failed
        run.exit_code = -1
    finally:
        run.finished_at = datetime.now(timezone.utc)
        await queue.put(None)
        _queues.pop(run_id, None)


async def stream_output(run_id: str) -> AsyncIterator[str | None]:
    queue = _queues.get(run_id)
    if queue is None:
        run = _runs.get(run_id)
        if run is None:
            yield None
            return
        for line in run.output:
            yield line
        yield None
        return

    while True:
        line = await queue.get()
        yield line
        if line is None:
            return
