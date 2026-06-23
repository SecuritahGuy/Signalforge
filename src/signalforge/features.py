from __future__ import annotations

import numpy as np
import pandas as pd

from signalforge.exceptions import FeatureError
from signalforge.logging_config import get_logger

logger = get_logger(__name__)


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
    lagged_features: bool = False,
    calendar_features: bool = False,
    cross_sectional_features: bool = False,
    technical_indicators: bool = False,
    interaction_features: bool = False,
    factor_proxies: bool = False,
) -> pd.DataFrame:
    """Build leakage-safe daily technical features from adjusted prices.

    Parameters
    ----------
    lagged_features : if True, add lagged versions of key return/volatility columns.
    calendar_features : if True, add day-of-week, month, quarter features.
    cross_sectional_features : if True, add per-date z-scores across the universe.
    technical_indicators : if True, add RSI, MACD, Bollinger Bands, ATR.
    interaction_features : if True, add return×volatility and volatility×volume pairs.
    factor_proxies : if True, add momentum-factor, low-vol, and quality proxies.
    """
    required = {date_col, symbol_col, price_col, volume_col}
    missing = required.difference(prices.columns)
    if missing:
        raise FeatureError(f"prices are missing required columns: {sorted(missing)}")

    frame = prices.sort_values([symbol_col, date_col]).reset_index(drop=True).copy()
    logger.debug("building price features for %d rows", len(frame))
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

    if lagged_features:
        _add_lagged_features(frame, grouped, return_windows)
    if calendar_features:
        _add_calendar_features(frame, date_col)
    if cross_sectional_features:
        _add_cross_sectional_features(frame, date_col, return_windows)
    if technical_indicators:
        _add_technical_indicators(frame, grouped, price_col)
    if interaction_features:
        _add_interaction_features(frame, return_windows, volume_windows)
    if factor_proxies:
        _add_factor_proxies(frame, grouped, return_windows)

    logger.debug(
        "built %d features (%d columns) for %d rows",
        sum(1 for c in frame.columns if c not in prices.columns),
        len(frame.columns),
        len(frame),
    )
    return frame


def _add_lagged_features(
    frame: pd.DataFrame,
    grouped: pd.core.groupby.DataFrameGroupBy,
    return_windows: tuple[int, ...],
) -> None:
    for window in return_windows[:3]:
        ret_col = f"return_{window}d"
        vol_col = f"volatility_{window}d"
        frame[f"{ret_col}_lag_1"] = grouped[ret_col].shift(1)
        frame[f"{vol_col}_lag_1"] = grouped[vol_col].shift(1)


def _add_calendar_features(frame: pd.DataFrame, date_col: str) -> None:
    dates = pd.to_datetime(frame[date_col])
    frame["day_of_week"] = dates.dt.dayofweek
    frame["day_of_week_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7)
    frame["day_of_week_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7)
    frame["month"] = dates.dt.month
    frame["month_sin"] = np.sin(2 * np.pi * (frame["month"] - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (frame["month"] - 1) / 12)
    frame["quarter"] = dates.dt.quarter
    frame["days_to_month_end"] = dates.dt.days_in_month - dates.dt.day
    frame["is_month_end"] = dates.dt.is_month_end.astype(int)


def _add_cross_sectional_features(
    frame: pd.DataFrame,
    date_col: str,
    return_windows: tuple[int, ...],
) -> None:
    zscore_cols = [
        f"return_{window}d" for window in [return_windows[0], return_windows[2]]
    ] + [
        f"volatility_{window}d" for window in [return_windows[0], return_windows[2], return_windows[-1]]
    ] + [
        "relative_volume_20d",
        "log_avg_dollar_volume_20d",
    ]
    for col in zscore_cols:
        if col in frame.columns:
            zscore = frame.groupby(date_col, sort=False)[col].transform(
                lambda x: (x - x.mean()) / x.std()
            )
            frame[f"zscore_{col}"] = zscore


def _add_technical_indicators(
    frame: pd.DataFrame,
    grouped: pd.core.groupby.DataFrameGroupBy,
    price_col: str,
) -> None:
    delta = grouped[price_col].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.groupby(frame["symbol"], sort=False).transform(
        lambda s: s.rolling(14, min_periods=14).mean()
    )
    avg_loss = loss.groupby(frame["symbol"], sort=False).transform(
        lambda s: s.rolling(14, min_periods=14).mean()
    )
    rs = avg_gain / avg_loss.replace(0, np.nan)
    frame["rsi_14"] = 100 - (100 / (1 + rs))

    ema_12 = grouped[price_col].transform(
        lambda s: s.ewm(span=12, adjust=False).mean()
    )
    ema_26 = grouped[price_col].transform(
        lambda s: s.ewm(span=26, adjust=False).mean()
    )
    macd = ema_12 - ema_26
    signal = macd.groupby(frame["symbol"], sort=False).transform(
        lambda s: s.ewm(span=9, adjust=False).mean()
    )
    frame["macd_12_26"] = macd
    frame["macd_signal_9"] = signal
    frame["macd_histogram_12_26_9"] = macd - signal

    sma_20 = grouped[price_col].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    std_20 = grouped[price_col].transform(
        lambda s: s.rolling(20, min_periods=20).std()
    )
    upper = sma_20 + 2 * std_20
    lower = sma_20 - 2 * std_20
    frame["bollinger_pct_b_20_2"] = (frame[price_col] - lower) / (upper - lower)
    frame["bollinger_width_20_2"] = (upper - lower) / sma_20

    high = frame["high"]
    low = frame["low"]
    prev_close = grouped[price_col].shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    frame["atr_14"] = true_range.groupby(frame["symbol"], sort=False).transform(
        lambda s: s.rolling(14, min_periods=14).mean()
    )


def _add_interaction_features(
    frame: pd.DataFrame,
    return_windows: tuple[int, ...],
    volume_windows: tuple[int, ...],
) -> None:
    pairs = [
        (f"return_{return_windows[1]}d", f"volatility_{return_windows[1]}d"),
        (f"return_{return_windows[0]}d", f"volatility_{return_windows[0]}d"),
    ]
    if len(volume_windows) > 0:
        pairs.append((f"volatility_{return_windows[1]}d", f"relative_volume_{volume_windows[0]}d"))
    if len(return_windows) > 0:
        pairs.append((f"return_{return_windows[2]}d", "volume_change_5d"))

    for col1, col2 in pairs:
        if col1 in frame.columns and col2 in frame.columns:
            frame[f"{col1}_x_{col2}"] = frame[col1] * frame[col2]


def _add_factor_proxies(
    frame: pd.DataFrame,
    grouped: pd.core.groupby.DataFrameGroupBy,
    return_windows: tuple[int, ...],
) -> None:
    if len(return_windows) >= 6:
        ret_long = f"return_{return_windows[-1]}d"
        ret_short = f"return_{return_windows[1]}d"
        if ret_long in frame.columns and ret_short in frame.columns:
            frame["momentum_factor"] = frame[ret_long] - frame[ret_short]

    vol_col = f"volatility_{return_windows[2] if len(return_windows) > 2 else return_windows[-1]}d"
    if vol_col in frame.columns:
        rank = frame.groupby("date", sort=False)[vol_col].rank(pct=True)
        frame["low_vol_factor"] = 1 - rank

    if "return_1d" in frame.columns:
        stability = grouped["return_1d"].transform(
            lambda s: s.rolling(60, min_periods=60).std()
        )
        frame["quality_factor"] = -stability


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
        raise FeatureError(f"universe is missing required columns: {sorted(missing_universe)}")

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
        raise FeatureError(f"benchmark_prices must include {date_col!r} and {price_col!r}")

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
