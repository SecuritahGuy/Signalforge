from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from signalforge.workflow import WorkflowConfig, run_workflow


def test_workflow_config_defaults():
    config = WorkflowConfig(research_frame="research_frame.csv")

    assert config.universe_source == "sp500"
    assert config.output_root == "reports"
    assert config.rebalance == "monthly"
    assert config.horizons == (5, 20, 60)
    assert config.top_n_per_lane == 25
    assert config.max_positions == 25
    assert config.starting_capital == 100_000.0
    assert config.cost_bps == 10.0
    assert config.run_discovery is True
    assert config.run_lane_backtest is True
    assert config.run_portfolio_backtest is True


def test_workflow_discovery_only_writes_top_level_outputs(tmp_path):
    frame_path = tmp_path / "research_frame.csv"
    output_root = tmp_path / "workflow"
    _synthetic_research_frame().to_csv(frame_path, index=False)

    result = run_workflow(
        _base_config(
            frame_path,
            output_root,
            run_lane_backtest=False,
            run_portfolio_backtest=False,
        )
    )

    assert (output_root / "discovery").is_dir()
    assert not (output_root / "lane_backtest").exists()
    assert not (output_root / "portfolio_backtest").exists()
    assert result.artifacts["summary"].exists()
    assert result.artifacts["report"].exists()
    assert result.artifacts["manifest"].exists()

    summary = json.loads((output_root / "workflow_summary.json").read_text())
    assert summary["discovery_ran"] is True
    assert summary["lane_backtest_ran"] is False
    assert summary["portfolio_backtest_ran"] is False
    assert summary["stages"]["discovery"]["status"] == "completed"
    assert summary["stages"]["lane_backtest"]["status"] == "skipped"
    assert "# SignalForge Workflow Report" in (output_root / "workflow_report.md").read_text()

    manifest = json.loads((output_root / "manifest.json").read_text())
    assert manifest["run_type"] == "workflow"
    assert manifest["inputs"]["research_frame"] == str(frame_path)
    assert "workflow_summary.json" in manifest["outputs"]
    assert "discovery/summary.json" in manifest["outputs"]


def test_workflow_full_run_creates_expected_subfolders_and_artifacts(tmp_path):
    frame_path = tmp_path / "research_frame.csv"
    output_root = tmp_path / "workflow"
    _synthetic_research_frame().to_csv(frame_path, index=False)

    result = run_workflow(_base_config(frame_path, output_root))

    assert (output_root / "discovery" / "summary.json").exists()
    assert (output_root / "lane_backtest" / "backtest_trades.csv").exists()
    assert (output_root / "portfolio_backtest" / "portfolio_daily_returns.csv").exists()
    assert (output_root / "portfolio_backtest" / "manifest.json").exists()
    assert set(result.stage_artifacts) == {
        "discovery",
        "lane_backtest",
        "portfolio_backtest",
    }

    summary = json.loads((output_root / "workflow_summary.json").read_text())
    assert summary["stages"]["portfolio_backtest"]["status"] == "completed"
    assert "portfolio_backtest" in summary["key_artifact_paths"]


def test_workflow_skip_flags_write_skipped_statuses(tmp_path):
    frame_path = tmp_path / "research_frame.csv"
    output_root = tmp_path / "workflow"
    _synthetic_research_frame().to_csv(frame_path, index=False)

    run_workflow(
        _base_config(
            frame_path,
            output_root,
            run_discovery=False,
            run_lane_backtest=False,
            run_portfolio_backtest=False,
        )
    )

    summary = json.loads((output_root / "workflow_summary.json").read_text())
    assert summary["stages"]["discovery"]["status"] == "skipped"
    assert summary["stages"]["lane_backtest"]["status"] == "skipped"
    assert summary["stages"]["portfolio_backtest"]["status"] == "skipped"


def test_workflow_portfolio_requires_lane_backtest_or_external_trades():
    config = WorkflowConfig(
        research_frame="unused.csv",
        run_discovery=False,
        run_lane_backtest=False,
        run_portfolio_backtest=True,
    )

    with pytest.raises(ValueError, match="portfolio backtest requires lane backtest"):
        run_workflow(config)


def test_workflow_handles_optional_fundamentals_path(tmp_path):
    frame_path = tmp_path / "research_frame.csv"
    fundamentals_path = tmp_path / "fundamentals.csv"
    output_root = tmp_path / "workflow"
    _synthetic_research_frame().to_csv(frame_path, index=False)
    _synthetic_fundamentals().to_csv(fundamentals_path, index=False)

    run_workflow(
        _base_config(
            frame_path,
            output_root,
            fundamentals=fundamentals_path,
            no_market_cap_filter=False,
            min_market_cap=1.0,
            run_lane_backtest=False,
            run_portfolio_backtest=False,
        )
    )

    enriched_path = output_root / "research_frame_enriched.csv"
    assert enriched_path.exists()
    enriched = pd.read_csv(enriched_path)
    assert "market_cap" in enriched.columns
    assert enriched["market_cap"].notna().all()

    manifest = json.loads((output_root / "manifest.json").read_text())
    assert manifest["inputs"]["fundamentals"] == str(fundamentals_path)
    assert "research_frame_enriched.csv" in manifest["outputs"]


def test_run_workflow_cli_discovery_only_smoke(tmp_path):
    frame_path = tmp_path / "research_frame.csv"
    output_root = tmp_path / "workflow_cli"
    _synthetic_research_frame().to_csv(frame_path, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_workflow.py",
            "--research-frame",
            str(frame_path),
            "--output-root",
            str(output_root),
            "--as-of-date",
            "2024-03-15",
            "--no-market-cap-filter",
            "--min-price",
            "0",
            "--min-avg-dollar-volume-20d",
            "0",
            "--skip-lane-backtest",
            "--skip-portfolio-backtest",
        ],
        check=True,
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )

    assert "wrote workflow artifacts" in completed.stdout
    assert (output_root / "workflow_summary.json").exists()
    assert (output_root / "discovery" / "report.md").exists()
    assert (output_root / "manifest.json").exists()


def _base_config(
    frame_path: Path,
    output_root: Path,
    **overrides,
) -> WorkflowConfig:
    values = {
        "research_frame": frame_path,
        "output_root": output_root,
        "as_of_date": "2024-03-15",
        "start_date": "2024-01-31",
        "end_date": "2024-02-29",
        "rebalance": "monthly",
        "horizons": (1, 2),
        "top_n_per_lane": 1,
        "max_positions": 2,
        "starting_capital": 1_000.0,
        "cost_bps": 10.0,
        "no_market_cap_filter": True,
        "min_price": 0.0,
        "min_avg_dollar_volume_20d": 0.0,
    }
    values.update(overrides)
    return WorkflowConfig(**values)


def _synthetic_research_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=75, freq="D")
    rows = []
    for symbol_index, symbol in enumerate(["AAA", "BBB", "CCC"]):
        for day_index, date in enumerate(dates):
            price = 20 + symbol_index * 5 + day_index * (1.0 + symbol_index / 10)
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "name": symbol,
                    "sector": "Technology" if symbol != "CCC" else "Industrials",
                    "industry": "Software",
                    "adj_close": price,
                    "avg_dollar_volume_20d": 20_000_000,
                    "return_1d": 0.02 + symbol_index / 100,
                    "return_5d": 0.05 + symbol_index / 100,
                    "return_20d": 0.15 - symbol_index / 100,
                    "return_60d": 0.40 - symbol_index / 20,
                    "sector_rank_return_20d": 1.0 - symbol_index / 10,
                    "sector_rank_return_60d": 1.0 - symbol_index / 10,
                    "sector_rank_momentum_20d": 1.0 - symbol_index / 10,
                    "sector_return_20d": 0.08,
                    "sector_return_60d": 0.15,
                    "stock_minus_market_return_20d": 0.10 - symbol_index / 100,
                    "stock_minus_sector_return_20d": 0.08 - symbol_index / 100,
                    "stock_minus_sector_return_60d": 0.15 - symbol_index / 100,
                    "relative_volume_20d": 1.2 + symbol_index,
                    "volume_change_5d": 0.10 + symbol_index / 10,
                    "volume_zscore_20d": 1.0 + symbol_index,
                    "price_above_sma_20": 0.05,
                    "price_above_sma_50": 0.08,
                    "price_above_sma_200": 0.04,
                    "distance_from_52w_high": -0.02,
                    "drawdown_60d": -0.05 - symbol_index / 10,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_fundamentals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "as_of_date": "2023-12-31",
                "market_cap": 1_000_000_000 + index * 100_000_000,
                "revenue_growth": 0.20 - index / 100,
                "eps_growth": 0.25 - index / 100,
                "gross_margin": 0.60 - index / 20,
                "return_on_equity": 0.20 - index / 100,
                "debt_to_equity": 0.5 + index / 10,
                "forward_pe": 20.0 + index,
                "source": "test",
            }
            for index, symbol in enumerate(["AAA", "BBB", "CCC"])
        ]
    )
