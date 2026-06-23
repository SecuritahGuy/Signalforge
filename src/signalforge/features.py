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
) -> pd.DataFrame:
    """Build leakage-safe daily technical features from adjusted prices.

    Rolling features use values available at the close of each row's date.
    Label generation remains separate so future returns are never included here.
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
    for window in (5, 20, 60):
        frame[f"return_{window}d"] = grouped[price_col].pct_change(window)
        frame[f"exit_close_{window}d"] = grouped[price_col].shift(-window)
        frame[f"momentum_{window}d"] = frame[f"return_{window}d"]
        frame[f"volatility_{window}d"] = grouped["return_1d"].transform(
            lambda series, window=window: series.rolling(window, min_periods=window).std()
        )

    frame["return_20d_skip_5d"] = grouped[price_col].transform(
        lambda series: series.shift(5).div(series.shift(25)).sub(1.0)
    )
    frame["return_60d_skip_5d"] = grouped[price_col].transform(
        lambda series: series.shift(5).div(series.shift(65)).sub(1.0)
    )
    frame["volatility_ratio_5d_20d"] = frame["volatility_5d"].div(frame["volatility_20d"])
    frame["volatility_ratio_20d_60d"] = frame["volatility_20d"].div(frame["volatility_60d"])

    frame["volume_change_5d"] = grouped[volume_col].pct_change(5)
    frame["avg_volume_20d"] = grouped[volume_col].transform(
        lambda series: series.rolling(20, min_periods=20).mean()
    )
    frame["relative_volume_20d"] = frame[volume_col].div(frame["avg_volume_20d"])
    frame["dollar_volume"] = frame[price_col] * frame[volume_col]
    frame["avg_dollar_volume_20d"] = grouped["dollar_volume"].transform(
        lambda series: series.rolling(20, min_periods=20).mean()
    )
    frame["log_dollar_volume"] = np.log1p(frame["dollar_volume"])
    frame["log_avg_dollar_volume_20d"] = np.log1p(frame["avg_dollar_volume_20d"])

    for window in (20, 50):
        sma = grouped[price_col].transform(
            lambda series, window=window: series.rolling(window, min_periods=window).mean()
        )
        frame[f"price_above_sma_{window}"] = frame[price_col].div(sma).sub(1.0)

    for window in (20, 60):
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
) -> pd.DataFrame:
    """Add sector-relative ranks for momentum and volatility features."""
    required_universe = {symbol_col, sector_col}
    missing_universe = required_universe.difference(universe.columns)
    if missing_universe:
        raise KeyError(f"universe is missing required columns: {sorted(missing_universe)}")

    frame = features.merge(universe[[symbol_col, sector_col]], on=symbol_col, how="left")
    for column in (
        "return_20d",
        "return_60d",
        "momentum_20d",
        "volatility_20d",
        "relative_volume_20d",
        "log_avg_dollar_volume_20d",
    ):
        if column in frame.columns:
            frame[f"sector_rank_{column}"] = frame.groupby([date_col, sector_col], dropna=False)[
                column
            ].rank(
                pct=True
            )

    for window in (5, 20, 60):
        return_col = f"return_{window}d"
        if return_col in frame.columns:
            sector_return_col = f"sector_return_{window}d"
            frame[sector_return_col] = frame.groupby([date_col, sector_col], dropna=False)[
                return_col
            ].transform("mean")
            frame[f"stock_minus_sector_return_{window}d"] = frame[return_col].sub(
                frame[sector_return_col]
            )
    return frame


def add_market_relative_features(
    features: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """Add benchmark-relative return, beta, and correlation features."""
    if date_col not in benchmark_prices.columns or price_col not in benchmark_prices.columns:
        raise KeyError(f"benchmark_prices must include {date_col!r} and {price_col!r}")

    benchmark = benchmark_prices.sort_values(date_col).reset_index(drop=True).copy()
    benchmark[date_col] = pd.to_datetime(benchmark[date_col])
    benchmark["market_return_1d"] = benchmark[price_col].pct_change()
    for window in (5, 20, 60):
        benchmark[f"market_return_{window}d"] = benchmark[price_col].pct_change(window)

    market_columns = [
        date_col,
        "market_return_1d",
        "market_return_5d",
        "market_return_20d",
        "market_return_60d",
    ]
    frame = features.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.merge(benchmark[market_columns], on=date_col, how="left")
    for window in (5, 20, 60):
        frame[f"stock_minus_market_return_{window}d"] = frame[f"return_{window}d"].sub(
            frame[f"market_return_{window}d"]
        )

    frame = frame.sort_values([symbol_col, date_col]).reset_index(drop=True)
    frame["beta_60d"] = np.nan
    frame["correlation_to_market_60d"] = np.nan
    for _, group in frame.groupby(symbol_col, sort=False):
        frame.loc[group.index, "beta_60d"] = _rolling_beta_60d(group).to_numpy()
        frame.loc[group.index, "correlation_to_market_60d"] = _rolling_market_corr_60d(
            group
        ).to_numpy()
    return frame


def _rolling_beta_60d(group: pd.DataFrame) -> pd.Series:
    covariance = group["return_1d"].rolling(60, min_periods=60).cov(group["market_return_1d"])
    variance = group["market_return_1d"].rolling(60, min_periods=60).var()
    return covariance.div(variance)


def _rolling_market_corr_60d(group: pd.DataFrame) -> pd.Series:
    return group["return_1d"].rolling(60, min_periods=60).corr(group["market_return_1d"])
