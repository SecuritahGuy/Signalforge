from pathlib import Path

import pandas as pd
import pytest

from signalforge.backtest import BacktestConfig
from signalforge.data import load_price_csv, load_universe_csv, split_benchmark_prices
from signalforge.exceptions import DataError
from signalforge.features import add_sector_relative_features, build_price_features
from signalforge.research import build_research_frame, run_momentum_smoke_backtest
from signalforge.validation import walk_forward_splits


def test_load_price_csv_normalizes_contract(tmp_path: Path):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        "date,symbol,open,high,low,close,adj_close,volume\n"
        "2024-01-02,aapl,101,103,100,102,102,1000\n"
        "2024-01-01,aapl,100,101,99,100,100,900\n"
    )

    prices = load_price_csv(csv_path)

    assert prices["symbol"].tolist() == ["AAPL", "AAPL"]
    assert prices["date"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]


def test_load_universe_csv_requires_core_metadata(tmp_path: Path):
    csv_path = tmp_path / "universe.csv"
    csv_path.write_text("symbol,name,category,sector,industry\nAAPL,Apple,tech,IT,Hardware\n")

    universe = load_universe_csv(csv_path)

    assert universe.loc[0, "symbol"] == "AAPL"


def test_split_benchmark_prices_requires_benchmark():
    prices = _synthetic_prices(["AAPL"], periods=3)

    with pytest.raises(DataError, match="benchmark symbol"):
        split_benchmark_prices(prices)


def test_build_price_features_uses_past_and_current_rows_only():
    prices = _synthetic_prices(["AAPL"], periods=65)

    features = build_price_features(prices)

    assert round(features.loc[5, "return_5d"], 4) == 0.05
    assert pd.isna(features.loc[18, "avg_dollar_volume_20d"])
    assert not pd.isna(features.loc[19, "avg_dollar_volume_20d"])


def test_sector_relative_features_rank_within_date_and_sector():
    prices = _synthetic_prices(["AAPL", "MSFT"], periods=65)
    features = build_price_features(prices)
    universe = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "sector": ["Information Technology", "Information Technology"],
        }
    )

    result = add_sector_relative_features(features, universe)
    latest = result.loc[result["date"] == result["date"].max()]

    assert set(latest["sector_rank_momentum_20d"]) == {0.5, 1.0}


def test_research_smoke_path_builds_labels_splits_and_backtest():
    prices = _synthetic_prices(["AAPL", "MSFT", "NVDA", "SPY"], periods=95)
    universe = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "NVDA"],
            "sector": ["Information Technology"] * 3,
        }
    )

    research_frame = build_research_frame(prices, universe, horizon=5)
    model_ready = research_frame.dropna(subset=["momentum_20d", "fwd_5d_return"])
    splits = list(
        walk_forward_splits(
            model_ready,
            first_train_start="2024-01-01",
            first_validation_start="2024-02-01",
            validation_months=1,
            purge_days=5,
        )
    )
    backtest = run_momentum_smoke_backtest(
        model_ready,
        config=BacktestConfig(long_fraction=0.34, short_fraction=0.34, max_position_weight=1.0),
    )

    assert "fwd_5d_excess_return" in research_frame.columns
    assert "fwd_5d_exec_return" in research_frame.columns
    assert "fwd_5d_exec_excess_return" in research_frame.columns
    assert splits
    assert not backtest.empty
    assert {"gross_return", "cost", "net_return"}.issubset(backtest.columns)


def test_build_research_frame_supports_multiple_label_horizons():
    prices = _synthetic_prices(["AAPL", "SPY"], periods=45)
    universe = pd.DataFrame({"symbol": ["AAPL"], "sector": ["Information Technology"]})

    research_frame = build_research_frame(prices, universe, horizons=(5, 20))

    assert "fwd_5d_excess_return" in research_frame.columns
    assert "fwd_20d_excess_return" in research_frame.columns
    assert "fwd_20d_exec_excess_return" in research_frame.columns


def _synthetic_prices(symbols: list[str], *, periods: int) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        for day_index, date in enumerate(dates):
            price = 100 + day_index + symbol_index
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": price - 0.5,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price,
                    "adj_close": price,
                    "volume": 1_000_000 + symbol_index * 10_000 + day_index,
                }
            )
    return pd.DataFrame(rows)
