from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from signalforge.discovery import DiscoveryConfig
from signalforge.discovery_backtest import (
    DiscoveryLaneBacktestConfig,
    aggregate_lane_backtest,
    forward_return_payload,
    latest_rows_as_of,
    run_discovery_lane_backtest,
    select_historical_as_of_dates,
    write_discovery_lane_backtest_outputs,
)


def test_select_historical_as_of_dates_monthly_and_weekly():
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-15"])})

    monthly = select_historical_as_of_dates(frame, rebalance="monthly")
    weekly = select_historical_as_of_dates(
        frame,
        start_date="2024-01-01",
        end_date="2024-02-01",
        rebalance="weekly",
    )

    assert monthly == (pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-15"))
    assert weekly == (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-31"))


def test_latest_rows_as_of_selects_latest_symbol_rows():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02"]),
            "symbol": ["AAA", "AAA", "BBB"],
            "adj_close": [10.0, 12.0, 20.0],
        }
    )

    latest = latest_rows_as_of(frame, "2024-01-02")

    assert latest.sort_values("symbol")["symbol"].tolist() == ["AAA", "BBB"]
    assert latest.sort_values("symbol")["adj_close"].tolist() == [10.0, 20.0]


def test_forward_return_payload_uses_trading_row_horizon_and_handles_missing_future():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"]),
            "symbol": ["AAA", "AAA", "AAA"],
            "adj_close": [10.0, 11.0, 13.0],
        }
    )

    payload = forward_return_payload(
        frame,
        symbol="AAA",
        as_of_date="2024-01-01",
        selection_price=10.0,
        horizons=(1, 2, 3),
    )

    assert payload["forward_price_1d"] == 11.0
    assert round(payload["forward_return_1d"], 4) == 0.1
    assert payload["forward_price_2d"] == 13.0
    assert round(payload["forward_return_2d"], 4) == 0.3
    assert pd.isna(payload["forward_return_3d"])


def test_aggregate_lane_backtest_metrics_by_lane_and_horizon():
    trades = pd.DataFrame(
        {
            "lane": ["momentum_breakouts", "momentum_breakouts", "sector_leaders"],
            "forward_return_5d": [0.10, -0.05, 0.02],
            "forward_return_20d": [0.20, pd.NA, -0.01],
        }
    )

    summary = aggregate_lane_backtest(trades, horizons=(5, 20))

    momentum_5d = summary.loc[
        (summary["lane"] == "momentum_breakouts") & (summary["horizon"] == 5)
    ].iloc[0]
    assert momentum_5d["selections"] == 2
    assert momentum_5d["avg_forward_return"] == 0.025
    assert momentum_5d["win_rate"] == 0.5

    momentum_20d = summary.loc[
        (summary["lane"] == "momentum_breakouts") & (summary["horizon"] == 20)
    ].iloc[0]
    assert momentum_20d["selections"] == 1
    assert momentum_20d["avg_forward_return"] == 0.20


def test_discovery_lane_backtest_outputs_files(tmp_path):
    frame = _synthetic_research_frame()
    config = DiscoveryLaneBacktestConfig(
        rebalance="monthly",
        top_n_per_lane=1,
        horizons=(1, 2),
        discovery_config=DiscoveryConfig(
            top_n=1,
            min_price=0,
            min_avg_dollar_volume_20d=0,
            min_market_cap=None,
        ),
    )

    result = run_discovery_lane_backtest(
        frame,
        start_date="2024-01-31",
        end_date="2024-02-29",
        config=config,
    )
    paths = write_discovery_lane_backtest_outputs(result, tmp_path)

    assert paths["trades"].exists()
    assert paths["summary"].exists()
    assert paths["report"].exists()
    trades = pd.read_csv(paths["trades"])
    assert {
        "as_of_date",
        "symbol",
        "lane",
        "rank",
        "score",
        "selection_price",
        "forward_price_1d",
        "forward_return_1d",
        "sector",
        "lane_reason",
    }.issubset(trades.columns)
    report = paths["report"].read_text()
    assert "# Discovery Lane Backtest Report" in report
    assert "No transaction costs" in report


def test_run_backtest_cli_on_synthetic_dataset(tmp_path):
    frame_path = tmp_path / "research_frame.csv"
    output_dir = tmp_path / "backtest_output"
    _synthetic_research_frame().to_csv(frame_path, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_backtest.py",
            "--research-frame",
            str(frame_path),
            "--output",
            str(output_dir),
            "--start-date",
            "2024-01-31",
            "--end-date",
            "2024-02-29",
            "--rebalance",
            "monthly",
            "--top-n-per-lane",
            "1",
            "--horizons",
            "1",
            "2",
            "--no-market-cap-filter",
            "--min-price",
            "0",
            "--min-avg-dollar-volume-20d",
            "0",
        ],
        check=True,
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )

    assert "wrote discovery lane backtest artifacts" in completed.stdout
    assert (output_dir / "backtest_trades.csv").exists()
    assert (output_dir / "backtest_summary.csv").exists()
    assert (output_dir / "backtest_report.md").exists()
    assert (output_dir / "manifest.json").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["run_type"] == "backtest"
    assert manifest["start_date"] == "2024-01-31"
    assert manifest["end_date"] == "2024-02-29"
    assert manifest["parameters"]["rebalance"] == "monthly"
    assert manifest["parameters"]["horizons"] == [1, 2]
    assert manifest["inputs"]["research_frame"] == str(frame_path)
    assert {
        "backtest_trades.csv",
        "backtest_summary.csv",
        "backtest_report.md",
        "manifest.json",
    }.issubset(set(manifest["outputs"]))


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
                    "distance_from_52w_high": -0.02,
                    "drawdown_60d": -0.05 - symbol_index / 10,
                }
            )
    return pd.DataFrame(rows)
