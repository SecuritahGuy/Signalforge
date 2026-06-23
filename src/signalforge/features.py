from __future__ import annotations

import numpy as np
import pandas as pd


def build_price_features(
    prices: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    price_col: str = "adj_close",
    volume_col: str = "volume",
    return_windows: tuple[int, ...] = (5, 10, 20, 40, 60, 120),
    sma_windows: tuple[int, ...] = (10, 20, 50, 200),
    range_windows: tuple[int, ...] = (10, 20, 60, 120),
    volume_windows: tuple[int, ...] = (10, 20, 40, 120),
) -> pd.DataFrame:
    """Build leakage-safe daily technical features from adjusted prices.

    Rolling features use values available at the close of each row's date.
    Label generation remains separate so future returns are never included here.

    Parameters
    ----------
    return_windows : windows for return/momentum/volatility features.
    sma_windows : windows for simple-moving-average features.
    range_windows : windows for high/low range and drawdown features.
    volume_windows : windows for average volume and dollar-volume features.
    """
    required = {date_col, symbol_col, price_col, volume_col}
    missing = required.difference(prices.columns)
    if missing:
        raise KeyError(f"prices are missing required columns: {sorted(missing)}")

    frame = prices.sort_values([symbol_col, date_col]).reset_index(drop=True).copy()
    grouped = frame.groupby(symbol_col, sort=False)

    frame["adj_open"] = frame["open"] * frame[price_col].div(frame["close"])
    frame["return_1d"] = grouped[price_col].pct_change()
    frame["next_open"] = grouped["adj_open"].shift(-1)
    for window in return_windows:
        frame[f"return_{window}d"] = grouped[price_col].pct_change(window)
        frame[f"exit_close_{window}d"] = grouped[price_col].shift(-window)
        frame[f"momentum_{window}d"] = frame[f"return_{window}d"]
        frame[f"volatility_{window}d"] = grouped["return_1d"].transform(
            lambda series, window=window: series.rolling(window, min_periods=window).std()
        )

    for window in return_windows[1:]:
        skip = 5
        ahead = window + skip
        frame[f"return_{window}d_skip_{skip}d"] = grouped[price_col].transform(
            lambda series, shift_=skip, shift_ahead_=ahead: (
                series.shift(shift_).div(series.shift(shift_ahead_)).sub(1.0)
            )
        )

    for i in range(len(return_windows) - 1):
        w1, w2 = return_windows[i], return_windows[i + 1]
        if f"volatility_{w1}d" in frame.columns and f"volatility_{w2}d" in frame.columns:
            frame[f"volatility_ratio_{w1}d_{w2}d"] = (
                frame[f"volatility_{w1}d"].div(frame[f"volatility_{w2}d"])
            )

    frame["volume_change_5d"] = grouped[volume_col].pct_change(5)
    for window in volume_windows:
        avg = grouped[volume_col].transform(
            lambda series, window=window: series.rolling(
                window, min_periods=window
            ).mean()
        )
        frame[f"avg_volume_{window}d"] = avg
        frame[f"relative_volume_{window}d"] = frame[volume_col].div(avg)

    frame["dollar_volume"] = frame[price_col] * frame[volume_col]
    for window in volume_windows:
        avg_dv = grouped["dollar_volume"].transform(
            lambda series, window=window: series.rolling(
                window, min_periods=window
            ).mean()
        )
        frame[f"avg_dollar_volume_{window}d"] = avg_dv
        frame[f"log_avg_dollar_volume_{window}d"] = np.log1p(avg_dv)
    frame["log_dollar_volume"] = np.log1p(frame["dollar_volume"])

    for window in sma_windows:
        sma = grouped[price_col].transform(
            lambda series, window=window: series.rolling(window, min_periods=window).mean()
        )
        frame[f"price_above_sma_{window}"] = frame[price_col].div(sma).sub(1.0)

    for window in range_windows:
        rolling_high = grouped[price_col].transform(
            lambda series, window=window: series.rolling(window, min_periods=window).max()
        )
        rolling_low = grouped[price_col].transform(
            lambda series, window=window: series.rolling(window, min_periods=window).min()
        )
        frame[f"distance_from_{window}d_high"] = frame[price_col].div(rolling_high).sub(1.0)
        frame[f"drawdown_{window}d"] = frame[f"distance_from_{window}d_high"]
        frame[f"high_low_range_{window}d"] = rolling_high.div(rolling_low).sub(1.0)

    return frame


def add_sector_relative_features(
    features: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    sector_col: str = "sector",
    return_windows: tuple[int, ...] = (5, 10, 20, 40, 60, 120),
) -> pd.DataFrame:
    """Add sector-relative ranks for momentum, volatility, and return features."""
    required_universe = {symbol_col, sector_col}
    missing_universe = required_universe.difference(universe.columns)
    if missing_universe:
        raise KeyError(f"universe is missing required columns: {sorted(missing_universe)}")

    frame = features.merge(universe[[symbol_col, sector_col]], on=symbol_col, how="left")
    new_cols: dict[str, pd.Series] = {}
    rank_columns = set()
    for window in return_windows:
        for prefix in ("return", "momentum", "volatility"):
            rank_columns.add(f"{prefix}_{window}d")
        rank_columns.add(f"relative_volume_{window}d")
        rank_columns.add(f"log_avg_dollar_volume_{window}d")
    for column in rank_columns:
        if column in frame.columns:
            new_cols[f"sector_rank_{column}"] = frame.groupby(
                [date_col, sector_col], dropna=False
            )[column].rank(pct=True)

    for window in return_windows:
        return_col = f"return_{window}d"
        if return_col in frame.columns:
            sector_return = frame.groupby([date_col, sector_col], dropna=False)[
                return_col
            ].transform("mean")
            new_cols[f"sector_return_{window}d"] = sector_return
            new_cols[f"stock_minus_sector_return_{window}d"] = frame[return_col].sub(sector_return)

    return frame.assign(**new_cols)


def add_market_relative_features(
    features: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    price_col: str = "adj_close",
    return_windows: tuple[int, ...] = (5, 10, 20, 40, 60, 120),
    beta_windows: tuple[int, ...] = (60, 120),
) -> pd.DataFrame:
    """Add benchmark-relative return, beta, and correlation features."""
    if date_col not in benchmark_prices.columns or price_col not in benchmark_prices.columns:
        raise KeyError(f"benchmark_prices must include {date_col!r} and {price_col!r}")

    benchmark = benchmark_prices.sort_values(date_col).reset_index(drop=True).copy()
    benchmark[date_col] = pd.to_datetime(benchmark[date_col])
    benchmark["market_return_1d"] = benchmark[price_col].pct_change()
    for window in return_windows:
        benchmark[f"market_return_{window}d"] = benchmark[price_col].pct_change(window)

    market_columns = [date_col, "market_return_1d"] + [
        f"market_return_{w}d" for w in return_windows
    ]
    frame = features.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.merge(benchmark[market_columns], on=date_col, how="left")
    for window in return_windows:
        return_col = f"return_{window}d"
        if return_col in frame.columns:
            frame[f"stock_minus_market_return_{window}d"] = frame[return_col].sub(
                frame[f"market_return_{window}d"]
            )

    frame = frame.sort_values([symbol_col, date_col]).reset_index(drop=True)
    for window in beta_windows:
        frame[f"beta_{window}d"] = np.nan
        frame[f"correlation_to_market_{window}d"] = np.nan
        for _, group in frame.groupby(symbol_col, sort=False):
            frame.loc[group.index, f"beta_{window}d"] = _rolling_beta(
                group, window
            ).to_numpy()
            frame.loc[group.index, f"correlation_to_market_{window}d"] = _rolling_market_corr(
                group, window
            ).to_numpy()
    return frame


def _rolling_beta(group: pd.DataFrame, window: int = 60) -> pd.Series:
    cov = group["return_1d"].rolling(window, min_periods=window).cov(
        group["market_return_1d"]
    )
    variance = group["market_return_1d"].rolling(window, min_periods=window).var()
    return cov.div(variance)


def _rolling_market_corr(group: pd.DataFrame, window: int = 60) -> pd.Series:
    return group["return_1d"].rolling(window, min_periods=window).corr(group["market_return_1d"])
