from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from signalforge.discovery import DiscoveryConfig, run_stock_discovery
from signalforge.discovery_backtest import (
    DiscoveryLaneBacktestConfig,
    run_discovery_lane_backtest,
    write_discovery_lane_backtest_outputs,
)
from signalforge.discovery_report import write_discovery_outputs
from signalforge.fundamentals import (
    enrich_research_frame_with_fundamentals,
    load_fundamentals_csv,
)
from signalforge.portfolio_backtest import (
    PortfolioBacktestConfig,
    run_portfolio_backtest,
    write_portfolio_backtest_outputs,
)
from signalforge.run_manifest import utc_now_iso, write_run_manifest


@dataclass(frozen=True)
class WorkflowConfig:
    research_frame: str | Path
    universe_source: str = "sp500"
    output_root: str | Path = "reports"
    as_of_date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    rebalance: str = "monthly"
    horizons: tuple[int, ...] = (5, 20, 60)
    top_n_per_lane: int = 25
    max_positions: int = 25
    starting_capital: float = 100_000.0
    cost_bps: float = 10.0
    exclude_watchlist: str | Path | None = None
    no_market_cap_filter: bool = False
    fundamentals: str | Path | None = None
    run_discovery: bool = True
    run_lane_backtest: bool = True
    run_portfolio_backtest: bool = True
    backtest_trades: str | Path | None = None
    selected_lanes: tuple[str, ...] = ()
    position_sizing_method: str = "equal_weight"
    price_col: str = "adj_close"
    min_price: float = 5.0
    min_avg_dollar_volume_20d: float = 5_000_000.0
    min_market_cap: float = 300_000_000.0
    earnings_blackout_days: int = 1


@dataclass(frozen=True)
class WorkflowResult:
    output_root: Path
    summary: dict
    artifacts: dict[str, Path]
    stage_artifacts: dict[str, dict[str, Path]]


def run_workflow(config: WorkflowConfig) -> WorkflowResult:
    """Run the lightweight SignalForge research workflow end to end."""
    _validate_workflow_config(config)
    created_at = utc_now_iso()
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    original_research_frame = pd.read_csv(config.research_frame)
    research_frame, used_research_frame_path = _maybe_enrich_research_frame(
        original_research_frame,
        config,
        output_root=output_root,
    )
    exclude_frame = _read_optional_csv(config.exclude_watchlist)
    stage_artifacts: dict[str, dict[str, Path]] = {}
    stage_status = {
        "discovery": _skipped_status(config.run_discovery),
        "lane_backtest": _skipped_status(config.run_lane_backtest),
        "portfolio_backtest": _skipped_status(config.run_portfolio_backtest),
    }

    discovery_config = _discovery_config(config)
    if config.run_discovery:
        discovery_result = run_stock_discovery(
            research_frame,
            as_of_date=config.as_of_date,
            existing_watchlist=exclude_frame,
            config=discovery_config,
        )
        discovery_output = output_root / "discovery"
        discovery_artifacts = write_discovery_outputs(discovery_result, discovery_output)
        discovery_artifacts["manifest"] = write_run_manifest(
            discovery_output,
            run_type="discovery",
            as_of_date=discovery_result.as_of_date.date().isoformat(),
            parameters=_discovery_manifest_parameters(config),
            inputs={
                "research_frame": used_research_frame_path,
                "watchlist": config.exclude_watchlist,
            },
            outputs=discovery_artifacts,
            code_cwd=Path.cwd(),
        )
        stage_artifacts["discovery"] = discovery_artifacts
        stage_status["discovery"] = _completed_status(discovery_artifacts)

    lane_trades = None
    lane_trades_path: str | Path | None = config.backtest_trades
    if config.run_lane_backtest:
        lane_config = DiscoveryLaneBacktestConfig(
            rebalance=config.rebalance,
            top_n_per_lane=config.top_n_per_lane,
            horizons=config.horizons,
            price_col=config.price_col,
            discovery_config=discovery_config,
        )
        lane_result = run_discovery_lane_backtest(
            research_frame,
            start_date=config.start_date,
            end_date=config.end_date,
            existing_watchlist=exclude_frame,
            config=lane_config,
        )
        lane_output = output_root / "lane_backtest"
        lane_artifacts = write_discovery_lane_backtest_outputs(lane_result, lane_output)
        lane_artifacts["manifest"] = write_run_manifest(
            lane_output,
            run_type="backtest",
            start_date=config.start_date,
            end_date=config.end_date,
            parameters=_lane_backtest_manifest_parameters(config),
            inputs={
                "research_frame": used_research_frame_path,
                "watchlist": config.exclude_watchlist,
            },
            outputs=lane_artifacts,
            code_cwd=Path.cwd(),
        )
        stage_artifacts["lane_backtest"] = lane_artifacts
        stage_status["lane_backtest"] = _completed_status(lane_artifacts)
        lane_trades = lane_result.trades
        lane_trades_path = lane_artifacts["trades"]

    if config.run_portfolio_backtest:
        if lane_trades is None:
            if config.backtest_trades is None:
                raise ValueError(
                    "portfolio backtest requires lane backtest output or --backtest-trades "
                    "when --skip-lane-backtest is used."
                )
            lane_trades = pd.read_csv(config.backtest_trades)

        portfolio_config = PortfolioBacktestConfig(
            starting_capital=config.starting_capital,
            rebalance=config.rebalance,
            selected_lanes=config.selected_lanes,
            max_positions=config.max_positions,
            position_sizing_method=config.position_sizing_method,
            cost_bps=config.cost_bps,
            price_col=config.price_col,
        )
        portfolio_result = run_portfolio_backtest(
            lane_trades,
            research_frame,
            config=portfolio_config,
        )
        portfolio_output = output_root / "portfolio_backtest"
        portfolio_artifacts = write_portfolio_backtest_outputs(
            portfolio_result,
            portfolio_output,
        )
        portfolio_artifacts["manifest"] = write_run_manifest(
            portfolio_output,
            run_type="portfolio_backtest",
            parameters=_portfolio_manifest_parameters(config),
            inputs={
                "backtest_trades": lane_trades_path,
                "research_frame": used_research_frame_path,
            },
            outputs=portfolio_artifacts,
            code_cwd=Path.cwd(),
        )
        stage_artifacts["portfolio_backtest"] = portfolio_artifacts
        stage_status["portfolio_backtest"] = _completed_status(portfolio_artifacts)

    summary = _workflow_summary(
        config,
        created_at_utc=created_at,
        used_research_frame_path=used_research_frame_path,
        stage_status=stage_status,
        stage_artifacts=stage_artifacts,
        output_root=output_root,
    )
    summary_path = output_root / "workflow_summary.json"
    report_path = output_root / "workflow_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    report_path.write_text(render_workflow_report(summary))

    top_artifacts = {
        "summary": summary_path,
        "report": report_path,
        **_relative_stage_outputs(stage_artifacts, output_root=output_root),
    }
    if config.fundamentals is not None:
        top_artifacts["enriched_research_frame"] = Path(used_research_frame_path).relative_to(
            output_root
        ).as_posix()
    manifest_path = write_run_manifest(
        output_root,
        run_type="workflow",
        parameters=_workflow_manifest_parameters(config),
        inputs={
            "research_frame": config.research_frame,
            "exclude_watchlist": config.exclude_watchlist,
            "fundamentals": config.fundamentals,
            "backtest_trades": config.backtest_trades,
        },
        outputs=top_artifacts,
        code_cwd=Path.cwd(),
    )
    artifacts = {
        "summary": summary_path,
        "report": report_path,
        "manifest": manifest_path,
    }
    return WorkflowResult(
        output_root=output_root,
        summary=summary,
        artifacts=artifacts,
        stage_artifacts=stage_artifacts,
    )


def render_workflow_report(summary: dict) -> str:
    """Render a top-level workflow report."""
    lines = [
        "# SignalForge Workflow Report",
        "",
        f"- Input research frame: {summary['research_frame']}",
        f"- Used research frame: {summary['used_research_frame']}",
        f"- Date range: {_date_range_label(summary)}",
        f"- Exclude watchlist: {summary.get('exclude_watchlist') or ''}",
        "",
        "## Stages Run",
        "",
    ]
    for stage, status in summary["stages"].items():
        lines.append(f"- {stage}: {status['status']}")

    lines.extend(["", "## Output Folders", ""])
    for stage, folder in summary["output_folders"].items():
        lines.append(f"- {stage}: {folder}")

    lines.extend(["", "## Key Artifacts", ""])
    for stage, artifacts in summary["key_artifact_paths"].items():
        if not artifacts:
            continue
        lines.append(f"### {stage}")
        for name, path in artifacts.items():
            lines.append(f"- {name}: {path}")
        lines.append("")

    lines.extend(
        [
            "## Caveats",
            "",
            "- Lightweight orchestration only; stage logic remains in the underlying modules.",
            "- Portfolio simulation is simplified long-only research output, not a live strategy.",
            "- Results depend on feature availability and point-in-time data quality.",
            "- Transaction costs use the configured simplified basis-point model.",
            "",
        ]
    )
    return "\n".join(lines)


def _maybe_enrich_research_frame(
    research_frame: pd.DataFrame,
    config: WorkflowConfig,
    *,
    output_root: Path,
) -> tuple[pd.DataFrame, str | Path]:
    if config.fundamentals is None:
        return research_frame, config.research_frame
    fundamentals = load_fundamentals_csv(config.fundamentals)
    enriched = enrich_research_frame_with_fundamentals(research_frame, fundamentals)
    enriched_path = output_root / "research_frame_enriched.csv"
    enriched.to_csv(enriched_path, index=False)
    return enriched, enriched_path


def _discovery_config(config: WorkflowConfig) -> DiscoveryConfig:
    return DiscoveryConfig(
        top_n=config.top_n_per_lane,
        price_col=config.price_col,
        min_price=config.min_price,
        min_avg_dollar_volume_20d=config.min_avg_dollar_volume_20d,
        min_market_cap=None if config.no_market_cap_filter else config.min_market_cap,
        earnings_blackout_days=config.earnings_blackout_days,
    )


def _workflow_summary(
    config: WorkflowConfig,
    *,
    created_at_utc: str,
    used_research_frame_path: str | Path,
    stage_status: dict[str, dict],
    stage_artifacts: dict[str, dict[str, Path]],
    output_root: Path,
) -> dict:
    output_folders = {
        "discovery": str(output_root / "discovery"),
        "lane_backtest": str(output_root / "lane_backtest"),
        "portfolio_backtest": str(output_root / "portfolio_backtest"),
    }
    return {
        "created_at_utc": created_at_utc,
        "research_frame": str(config.research_frame),
        "used_research_frame": str(used_research_frame_path),
        "as_of_date": config.as_of_date,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "exclude_watchlist": None
        if config.exclude_watchlist is None
        else str(config.exclude_watchlist),
        "fundamentals": None if config.fundamentals is None else str(config.fundamentals),
        "output_root": str(output_root),
        "output_folders": output_folders,
        "discovery_ran": config.run_discovery,
        "lane_backtest_ran": config.run_lane_backtest,
        "portfolio_backtest_ran": config.run_portfolio_backtest,
        "stages": stage_status,
        "key_artifact_paths": {
            stage: {name: str(path) for name, path in artifacts.items()}
            for stage, artifacts in stage_artifacts.items()
        },
    }


def _relative_stage_outputs(
    stage_artifacts: dict[str, dict[str, Path]],
    *,
    output_root: Path,
) -> dict[str, str]:
    outputs = {}
    for stage, artifacts in stage_artifacts.items():
        for name, path in artifacts.items():
            outputs[f"{stage}_{name}"] = Path(path).relative_to(output_root).as_posix()
    return outputs


def _completed_status(artifacts: dict[str, Path]) -> dict:
    return {
        "enabled": True,
        "status": "completed",
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }


def _skipped_status(enabled: bool) -> dict:
    if enabled:
        return {"enabled": True, "status": "pending", "artifacts": {}}
    return {"enabled": False, "status": "skipped", "artifacts": {}}


def _read_optional_csv(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return pd.read_csv(path)


def _workflow_manifest_parameters(config: WorkflowConfig) -> dict:
    return {
        "universe_source": config.universe_source,
        "as_of_date": config.as_of_date,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "rebalance": config.rebalance,
        "horizons": list(config.horizons),
        "top_n_per_lane": config.top_n_per_lane,
        "max_positions": config.max_positions,
        "starting_capital": config.starting_capital,
        "cost_bps": config.cost_bps,
        "no_market_cap_filter": config.no_market_cap_filter,
        "run_discovery": config.run_discovery,
        "run_lane_backtest": config.run_lane_backtest,
        "run_portfolio_backtest": config.run_portfolio_backtest,
        "selected_lanes": list(config.selected_lanes),
        "position_sizing_method": config.position_sizing_method,
        "price_col": config.price_col,
        "min_price": config.min_price,
        "min_avg_dollar_volume_20d": config.min_avg_dollar_volume_20d,
        "min_market_cap": None if config.no_market_cap_filter else config.min_market_cap,
    }


def _discovery_manifest_parameters(config: WorkflowConfig) -> dict:
    params = _workflow_manifest_parameters(config)
    return {
        key: params[key]
        for key in (
            "as_of_date",
            "top_n_per_lane",
            "min_price",
            "min_avg_dollar_volume_20d",
            "min_market_cap",
            "no_market_cap_filter",
        )
    }


def _lane_backtest_manifest_parameters(config: WorkflowConfig) -> dict:
    params = _workflow_manifest_parameters(config)
    return {
        key: params[key]
        for key in (
            "start_date",
            "end_date",
            "rebalance",
            "horizons",
            "top_n_per_lane",
            "min_price",
            "min_avg_dollar_volume_20d",
            "min_market_cap",
            "no_market_cap_filter",
        )
    }


def _portfolio_manifest_parameters(config: WorkflowConfig) -> dict:
    return {
        "starting_capital": config.starting_capital,
        "rebalance": config.rebalance,
        "max_positions": config.max_positions,
        "cost_bps": config.cost_bps,
        "selected_lanes": list(config.selected_lanes),
        "position_sizing_method": config.position_sizing_method,
        "price_col": config.price_col,
    }


def _date_range_label(summary: dict) -> str:
    start = summary.get("start_date") or ""
    end = summary.get("end_date") or ""
    if start or end:
        return f"{start} to {end}".strip()
    return "not specified"


def _validate_workflow_config(config: WorkflowConfig) -> None:
    if config.rebalance not in {"monthly", "weekly"}:
        raise ValueError("rebalance must be 'monthly' or 'weekly'")
    if not config.horizons:
        raise ValueError("horizons must not be empty")
    if any(horizon <= 0 for horizon in config.horizons):
        raise ValueError("horizons must be positive")
    if config.top_n_per_lane <= 0:
        raise ValueError("top_n_per_lane must be positive")
    if config.max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if config.starting_capital <= 0:
        raise ValueError("starting_capital must be positive")
    if config.cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    if (
        config.run_portfolio_backtest
        and not config.run_lane_backtest
        and config.backtest_trades is None
    ):
        raise ValueError(
            "portfolio backtest requires lane backtest output or --backtest-trades "
            "when --skip-lane-backtest is used."
        )
