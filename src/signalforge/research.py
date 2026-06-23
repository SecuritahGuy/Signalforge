from __future__ import annotations

import pandas as pd

from signalforge.backtest import BacktestConfig, long_short_daily_returns
from signalforge.data import split_benchmark_prices
from signalforge.features import (
    add_market_relative_features,
    add_sector_relative_features,
    build_price_features,
)
from signalforge.labels import (
    excess_forward_return,
    executable_excess_forward_return,
    executable_forward_return,
    forward_return,
)


def build_research_frame(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    benchmark_symbol: str = "SPY",
    horizon: int = 5,
    horizons: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Create the first model-ready frame from normalized prices and universe data."""
    label_horizons = horizons or (horizon,)
    tradable_prices, benchmark_prices = split_benchmark_prices(
        prices,
        benchmark_symbol=benchmark_symbol,
    )
    features = build_price_features(tradable_prices)
    features = add_sector_relative_features(features, universe)
    features = add_market_relative_features(features, benchmark_prices)
    for label_horizon in label_horizons:
        features[f"fwd_{label_horizon}d_return"] = forward_return(
            tradable_prices,
            horizon=label_horizon,
        )
        features[f"fwd_{label_horizon}d_excess_return"] = excess_forward_return(
            tradable_prices,
            benchmark_prices,
            horizon=label_horizon,
        )
        features[f"fwd_{label_horizon}d_exec_return"] = executable_forward_return(
            tradable_prices,
            horizon=label_horizon,
        )
        features[f"fwd_{label_horizon}d_exec_excess_return"] = (
            executable_excess_forward_return(
                tradable_prices,
                benchmark_prices,
                horizon=label_horizon,
            )
        )
    return features


def run_momentum_smoke_backtest(
    research_frame: pd.DataFrame,
    *,
    score_col: str = "momentum_20d",
    return_col: str = "fwd_5d_return",
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Run a simple benchmark backtest using momentum as a placeholder score."""
    scored = research_frame.rename(columns={score_col: "score", return_col: "forward_return"})
    return long_short_daily_returns(scored, config=config)
