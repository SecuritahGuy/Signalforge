from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from signalforge.portfolio_backtest import (
    PortfolioBacktestConfig,
    build_portfolio_targets,
    calculate_max_drawdown,
    calculate_summary_metrics,
    calculate_turnover,
    deduplicate_lane_selections,
    run_portfolio_backtest,
    write_portfolio_backtest_outputs,
)


def test_inverse_volatility_allocation_weights_by_volatility():
    selections = _selection_frame()
    returns = pd.DataFrame(
        {
            "AAA": [0.01, -0.01, 0.02, -0.02, 0.01],
            "BBB": [0.001, -0.001, 0.002, -0.002, 0.001],
            "CCC": [0.05, -0.05, 0.04, -0.06, 0.03],
        }
    )
    targets = build_portfolio_targets(
        selections,
        max_positions=3,
        position_sizing_method="inverse_volatility",
        returns=returns,
    )

    assert len(targets) == 3
    assert set(targets["symbol"]) == {"AAA", "BBB", "CCC"}
    assert abs(targets["weight"].sum() - 1.0) < 1e-10
    # BBB has lowest volatility -> highest weight
    bbb_weight = targets.loc[targets["symbol"] == "BBB", "weight"].iloc[0]
    aaa_weight = targets.loc[targets["symbol"] == "AAA", "weight"].iloc[0]
    ccc_weight = targets.loc[targets["symbol"] == "CCC", "weight"].iloc[0]
    assert bbb_weight > aaa_weight > ccc_weight


def test_inverse_volatility_raises_without_returns():
    selections = _selection_frame()
    with pytest.raises(ValueError, match="returns frame is required"):
        build_portfolio_targets(
            selections,
            max_positions=2,
            position_sizing_method="inverse_volatility",
        )


def test_inverse_volatility_raises_on_empty_returns():
    selections = _selection_frame()
    with pytest.raises(ValueError, match="returns frame is required"):
        build_portfolio_targets(
            selections,
            max_positions=2,
            position_sizing_method="inverse_volatility",
            returns=pd.DataFrame(),
        )


def test_inverse_volatility_skips_symbols_without_volatility():
    selections = _selection_frame()
    returns = pd.DataFrame({"AAA": [0.01, -0.01, 0.02]})
    targets = build_portfolio_targets(
        selections,
        max_positions=3,
        position_sizing_method="inverse_volatility",
        returns=returns,
    )

    assert len(targets) == 3
    assert targets.loc[targets["symbol"] == "AAA", "weight"].iloc[0] == 1.0
    assert targets.loc[targets["symbol"] == "BBB", "weight"].iloc[0] == 0.0
    assert targets.loc[targets["symbol"] == "CCC", "weight"].iloc[0] == 0.0


def test_inverse_volatility_respects_max_position_weight():
    selections = _selection_frame()
    returns = pd.DataFrame(
        {"AAA": [0.01, -0.01], "BBB": [0.005, -0.005], "CCC": [0.02, -0.02]}
    )
    targets = build_portfolio_targets(
        selections,
        max_positions=3,
        position_sizing_method="inverse_volatility",
        max_position_weight=0.40,
        returns=returns,
    )

    assert all(targets["weight"] <= 0.40 + 1e-6)
    assert abs(targets["weight"].sum() - 1.0) < 1e-10


def test_build_portfolio_targets_raises_for_unsupported_method():
    selections = _selection_frame()
    with pytest.raises(ValueError, match="unsupported position_sizing_method"):
        build_portfolio_targets(
            selections,
            max_positions=2,
            position_sizing_method="risk_parity",
        )


def test_equal_weight_allocation_dedupes_multi_lane_symbols():
    selections = _selection_frame()

    deduped = deduplicate_lane_selections(selections)
    targets = build_portfolio_targets(selections, max_positions=2)

    aaa = deduped.loc[deduped["symbol"] == "AAA"].iloc[0]
    assert aaa["lanes_matched"] == "momentum_breakouts, sector_leaders"
    assert aaa["score"] == 90.0
    assert targets["symbol"].tolist() == ["CCC", "AAA"]
    assert targets["weight"].tolist() == [0.5, 0.5]


def test_turnover_and_transaction_cost_are_calculated_from_traded_weight():
    assert calculate_turnover({"AAA": 1.0}, {"BBB": 1.0}) == 2.0

    result = run_portfolio_backtest(
        _selection_frame().loc[lambda frame: frame["symbol"].isin(["AAA", "CCC"])],
        _price_frame(),
        config=PortfolioBacktestConfig(
            starting_capital=1_000.0,
            max_positions=2,
            cost_bps=10.0,
        ),
    )

    first_day = result.daily_returns.iloc[0]
    assert round(first_day["transaction_cost"], 6) == 1.0
    assert round(first_day["net_return"], 6) == -0.001
    assert round(first_day["portfolio_value"], 6) == 999.0
    assert round(result.trades["trade_weight"].sum(), 6) == 1.0
    assert round(result.trades["transaction_cost"].sum(), 6) == 1.0


def test_daily_net_returns_follow_synthetic_price_series():
    result = run_portfolio_backtest(
        _selection_frame().loc[lambda frame: frame["symbol"].isin(["AAA", "BBB"])],
        _price_frame(),
        config=PortfolioBacktestConfig(
            starting_capital=1_000.0,
            max_positions=2,
            cost_bps=0.0,
        ),
    )

    daily = result.daily_returns.set_index("date")

    assert round(daily.loc[pd.Timestamp("2024-01-31"), "portfolio_value"], 6) == 1_000.0
    assert round(daily.loc[pd.Timestamp("2024-02-01"), "net_return"], 6) == 0.0
    assert round(daily.loc[pd.Timestamp("2024-02-02"), "net_return"], 6) == 0.10
    assert round(daily.loc[pd.Timestamp("2024-02-02"), "portfolio_value"], 6) == 1_100.0


def test_max_drawdown_and_summary_metrics():
    assert round(calculate_max_drawdown(pd.Series([100.0, 110.0, 99.0, 120.0])), 6) == -0.1

    result = run_portfolio_backtest(
        _selection_frame().loc[lambda frame: frame["symbol"].isin(["AAA", "BBB"])],
        _price_frame(),
        config=PortfolioBacktestConfig(starting_capital=1_000.0, max_positions=2),
    )
    summary = calculate_summary_metrics(
        result.daily_returns,
        result.trades,
        result.holdings,
        starting_capital=1_000.0,
    )

    assert round(summary["ending_capital"], 6) == 1_100.0
    assert round(summary["total_return"], 6) == 0.10
    assert summary["number_of_rebalances"] == 1
    assert summary["number_of_positions_average"] == 2
    assert round(summary["average_turnover"], 6) == 1.0
    assert "sharpe_ratio" in summary


def test_portfolio_backtest_outputs_are_written(tmp_path):
    result = run_portfolio_backtest(
        _selection_frame(),
        _price_frame(),
        config=PortfolioBacktestConfig(starting_capital=1_000.0, max_positions=2),
    )

    paths = write_portfolio_backtest_outputs(result, tmp_path)

    assert paths["daily_returns"].exists()
    assert paths["holdings"].exists()
    assert paths["trades"].exists()
    assert paths["summary"].exists()
    assert paths["report"].exists()
    assert "# Portfolio Backtest Report" in paths["report"].read_text()
    summary = json.loads(paths["summary"].read_text())
    assert summary["starting_capital"] == 1_000.0


def test_run_portfolio_backtest_cli_writes_manifest(tmp_path):
    trades_path = tmp_path / "backtest_trades.csv"
    research_frame_path = tmp_path / "research_frame.csv"
    output_dir = tmp_path / "portfolio_output"
    _selection_frame().to_csv(trades_path, index=False)
    _price_frame().to_csv(research_frame_path, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_portfolio_backtest.py",
            "--backtest-trades",
            str(trades_path),
            "--research-frame",
            str(research_frame_path),
            "--output",
            str(output_dir),
            "--starting-capital",
            "1000",
            "--rebalance",
            "monthly",
            "--max-positions",
            "2",
            "--cost-bps",
            "10",
            "--lanes",
            "momentum_breakouts",
            "sector_leaders",
        ],
        check=True,
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )

    assert "wrote portfolio backtest artifacts" in completed.stdout
    assert (output_dir / "portfolio_daily_returns.csv").exists()
    assert (output_dir / "portfolio_holdings.csv").exists()
    assert (output_dir / "portfolio_trades.csv").exists()
    assert (output_dir / "portfolio_summary.json").exists()
    assert (output_dir / "portfolio_report.md").exists()
    assert (output_dir / "manifest.json").exists()

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["run_type"] == "portfolio_backtest"
    assert manifest["inputs"]["backtest_trades"] == str(trades_path)
    assert manifest["inputs"]["research_frame"] == str(research_frame_path)
    assert manifest["parameters"]["starting_capital"] == 1_000.0
    assert manifest["parameters"]["selected_lanes"] == [
        "momentum_breakouts",
        "sector_leaders",
    ]
    assert {
        "portfolio_daily_returns.csv",
        "portfolio_holdings.csv",
        "portfolio_trades.csv",
        "portfolio_summary.json",
        "portfolio_report.md",
        "manifest.json",
    }.issubset(set(manifest["outputs"]))


def _selection_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "as_of_date": pd.Timestamp("2024-01-31"),
                "symbol": "AAA",
                "lane": "momentum_breakouts",
                "rank": 1,
                "score": 90.0,
            },
            {
                "as_of_date": pd.Timestamp("2024-01-31"),
                "symbol": "AAA",
                "lane": "sector_leaders",
                "rank": 2,
                "score": 80.0,
            },
            {
                "as_of_date": pd.Timestamp("2024-01-31"),
                "symbol": "BBB",
                "lane": "momentum_breakouts",
                "rank": 3,
                "score": 70.0,
            },
            {
                "as_of_date": pd.Timestamp("2024-01-31"),
                "symbol": "CCC",
                "lane": "value_recoveries",
                "rank": 1,
                "score": 95.0,
            },
        ]
    )


def _price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2024-01-31", "symbol": "AAA", "adj_close": 100.0},
            {"date": "2024-01-31", "symbol": "BBB", "adj_close": 100.0},
            {"date": "2024-01-31", "symbol": "CCC", "adj_close": 50.0},
            {"date": "2024-02-01", "symbol": "AAA", "adj_close": 110.0},
            {"date": "2024-02-01", "symbol": "BBB", "adj_close": 90.0},
            {"date": "2024-02-01", "symbol": "CCC", "adj_close": 55.0},
            {"date": "2024-02-02", "symbol": "AAA", "adj_close": 121.0},
            {"date": "2024-02-02", "symbol": "BBB", "adj_close": 99.0},
            {"date": "2024-02-02", "symbol": "CCC", "adj_close": 60.5},
        ]
    )
