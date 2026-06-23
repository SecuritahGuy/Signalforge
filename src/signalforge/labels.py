from __future__ import annotations

import pandas as pd


def forward_return(
    prices: pd.DataFrame,
    *,
    price_col: str = "adj_close",
    horizon: int = 5,
    group_col: str = "symbol",
) -> pd.Series:
    """Compute point-in-time forward returns by symbol.

    The value at date t is the return from t close to t + horizon close.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    required = {group_col, price_col}
    missing = required.difference(prices.columns)
    if missing:
        raise KeyError(f"prices is missing required columns: {sorted(missing)}")

    sorted_prices = _sort_panel(prices, group_col=group_col)
    future_price = sorted_prices.groupby(group_col, sort=False)[price_col].shift(-horizon)
    return future_price.div(sorted_prices[price_col]).sub(1.0).rename(f"fwd_{horizon}d_return")


def excess_forward_return(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    date_col: str = "date",
    price_col: str = "adj_close",
    horizon: int = 5,
    group_col: str = "symbol",
) -> pd.Series:
    """Compute symbol forward return minus benchmark forward return."""
    symbol_returns = forward_return(
        prices,
        price_col=price_col,
        horizon=horizon,
        group_col=group_col,
    )

    benchmark_sorted = benchmark.sort_values(date_col).reset_index(drop=True)
    if date_col not in benchmark_sorted.columns or price_col not in benchmark_sorted.columns:
        raise KeyError(f"benchmark must include {date_col!r} and {price_col!r}")

    benchmark_returns = (
        benchmark_sorted[price_col].shift(-horizon).div(benchmark_sorted[price_col]).sub(1.0)
    )
    benchmark_frame = benchmark_sorted[[date_col]].assign(_benchmark_return=benchmark_returns)

    aligned = _sort_panel(prices, group_col=group_col)[[date_col]].merge(
        benchmark_frame,
        on=date_col,
        how="left",
    )
    return symbol_returns.sub(aligned["_benchmark_return"]).rename(f"fwd_{horizon}d_excess_return")


def executable_forward_return(
    prices: pd.DataFrame,
    *,
    date_col: str = "date",
    open_col: str = "open",
    close_col: str = "close",
    price_col: str = "adj_close",
    horizon: int = 5,
    group_col: str = "symbol",
) -> pd.Series:
    """Compute next-open to future-close return on an adjusted price basis."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    required = {group_col, open_col, close_col, price_col}
    missing = required.difference(prices.columns)
    if missing:
        raise KeyError(f"prices is missing required columns: {sorted(missing)}")

    sorted_prices = _sort_panel(prices, group_col=group_col, date_col=date_col)
    adjusted_open = sorted_prices[open_col] * sorted_prices[price_col].div(
        sorted_prices[close_col]
    )
    grouped_open = adjusted_open.groupby(sorted_prices[group_col], sort=False)
    grouped_close = sorted_prices.groupby(group_col, sort=False)[price_col]
    entry_price = grouped_open.shift(-1)
    exit_price = grouped_close.shift(-horizon)
    return exit_price.div(entry_price).sub(1.0).rename(f"fwd_{horizon}d_exec_return")


def executable_excess_forward_return(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    date_col: str = "date",
    open_col: str = "open",
    close_col: str = "close",
    price_col: str = "adj_close",
    horizon: int = 5,
    group_col: str = "symbol",
) -> pd.Series:
    """Compute executable symbol return minus executable benchmark return."""
    symbol_returns = executable_forward_return(
        prices,
        date_col=date_col,
        open_col=open_col,
        close_col=close_col,
        price_col=price_col,
        horizon=horizon,
        group_col=group_col,
    )
    benchmark_returns = executable_forward_return(
        benchmark,
        date_col=date_col,
        open_col=open_col,
        close_col=close_col,
        price_col=price_col,
        horizon=horizon,
        group_col=group_col,
    )
    benchmark_frame = _sort_panel(benchmark, group_col=group_col, date_col=date_col)[
        [date_col]
    ].assign(_benchmark_exec_return=benchmark_returns)
    aligned = _sort_panel(prices, group_col=group_col, date_col=date_col)[[date_col]].merge(
        benchmark_frame,
        on=date_col,
        how="left",
    )
    return symbol_returns.sub(aligned["_benchmark_exec_return"]).rename(
        f"fwd_{horizon}d_exec_excess_return"
    )


def _sort_panel(frame: pd.DataFrame, *, group_col: str, date_col: str = "date") -> pd.DataFrame:
    if date_col not in frame.columns:
        raise KeyError(f"frame is missing required column: {date_col}")
    return frame.sort_values([group_col, date_col]).reset_index(drop=True)
