from __future__ import annotations

from pathlib import Path

import pandas as pd

PRICE_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "adj_close", "volume")
UNIVERSE_COLUMNS = ("symbol", "name", "category", "sector", "industry")


def load_price_csv(path: str | Path) -> pd.DataFrame:
    """Load normalized daily OHLCV data from CSV."""
    frame = pd.read_csv(path)
    validate_price_frame(frame)
    frame = frame.loc[:, PRICE_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].str.upper()
    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_universe_csv(path: str | Path) -> pd.DataFrame:
    """Load the research universe definition from CSV."""
    frame = pd.read_csv(path)
    missing = set(UNIVERSE_COLUMNS).difference(frame.columns)
    if missing:
        raise KeyError(f"universe is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].str.upper()
    return frame


def split_benchmark_prices(
    prices: pd.DataFrame,
    *,
    benchmark_symbol: str = "SPY",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a price panel into tradable symbols and a benchmark series."""
    benchmark_symbol = benchmark_symbol.upper()
    validate_price_frame(prices)
    benchmark = prices.loc[prices["symbol"].str.upper() == benchmark_symbol].copy()
    if benchmark.empty:
        raise ValueError(f"benchmark symbol {benchmark_symbol!r} not found in prices")
    tradable = prices.loc[prices["symbol"].str.upper() != benchmark_symbol].copy()
    return tradable.reset_index(drop=True), benchmark.reset_index(drop=True)


def validate_price_frame(frame: pd.DataFrame) -> None:
    missing = set(PRICE_COLUMNS).difference(frame.columns)
    if missing:
        raise KeyError(f"prices are missing required columns: {sorted(missing)}")
    if frame[list(PRICE_COLUMNS)].isna().any().any():
        raise ValueError("prices contain null values in required columns")
