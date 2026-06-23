import pandas as pd

from signalforge.backtest import BacktestConfig
from signalforge.diagnostics import (
    daily_portfolio_diagnostics,
    monthly_portfolio_returns,
    symbol_contribution_diagnostics,
)


def test_daily_and_monthly_diagnostics_from_predictions():
    predictions = _predictions()

    daily = daily_portfolio_diagnostics(
        predictions,
        realized_return_col="fwd_5d_return",
        config=BacktestConfig(long_fraction=0.25, short_fraction=0.25, max_position_weight=1.0),
    )
    monthly = monthly_portfolio_returns(daily)

    assert not daily.empty
    assert {"raw_equity", "risk_equity_full", "split_id"}.issubset(daily.columns)
    assert monthly.loc[0, "month"] == "2024-01"
    assert "risk_return" in monthly.columns


def test_symbol_contribution_diagnostics_counts_long_and_short_days():
    predictions = _predictions()

    contributions = symbol_contribution_diagnostics(
        predictions,
        realized_return_col="fwd_5d_return",
        config=BacktestConfig(long_fraction=0.25, short_fraction=0.25, max_position_weight=1.0),
    )

    assert set(contributions["symbol"]) == {"A", "D"}
    assert contributions["long_days"].sum() == 2
    assert contributions["short_days"].sum() == 2


def test_symbol_contribution_diagnostics_respects_symbol_trade_cap():
    predictions = _predictions()

    contributions = symbol_contribution_diagnostics(
        predictions,
        realized_return_col="fwd_5d_return",
        config=BacktestConfig(
            long_fraction=0.25,
            short_fraction=0.25,
            max_position_weight=1.0,
            max_symbol_trades=1,
        ),
    )

    assert contributions["long_days"].max() <= 1
    assert contributions["short_days"].max() <= 1


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 4 + ["2024-01-03"] * 4),
            "symbol": ["A", "B", "C", "D"] * 2,
            "split_id": [1] * 8,
            "prediction": [0.4, 0.2, -0.1, -0.3, 0.5, 0.1, -0.2, -0.4],
            "fwd_5d_return": [0.02, 0.01, -0.01, -0.02, 0.03, 0.01, -0.01, -0.03],
        }
    )
