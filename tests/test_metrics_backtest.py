import pandas as pd

from signalforge.backtest import (
    BacktestConfig,
    apply_risk_controls,
    build_daily_positions,
    event_based_long_only_backtest,
    long_only_capital_backtest,
    long_short_daily_returns,
)
from signalforge.metrics import hit_rate, max_drawdown


def test_backtest_accounts_for_costs():
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 4),
            "symbol": ["A", "B", "C", "D"],
            "score": [0.4, 0.3, -0.2, -0.5],
            "forward_return": [0.02, 0.01, -0.01, -0.02],
        }
    )

    result = long_short_daily_returns(
        scored,
        config=BacktestConfig(
            long_fraction=0.25,
            short_fraction=0.25,
            max_position_weight=1.0,
            transaction_cost_bps=10.0,
            slippage_bps=0.0,
        ),
    )

    assert result.loc[0, "gross_return"] == 0.04
    assert result.loc[0, "cost"] == 0.002
    assert result.loc[0, "net_return"] == 0.038
    assert result.loc[0, "risk_net_return"] == 0.038


def test_risk_controls_apply_lagged_vol_target_and_drawdown_cooldown():
    daily = pd.DataFrame({"net_return": [0.01, 0.01, -0.25, 0.05, 0.05]})

    result = apply_risk_controls(
        daily,
        config=BacktestConfig(
            target_volatility=0.10,
            volatility_lookback=2,
            max_leverage=1.0,
            max_drawdown_stop=0.10,
            cooldown_days=2,
        ),
    )

    assert {"leverage", "risk_net_return", "risk_trading_enabled", "risk_equity"}.issubset(
        result.columns
    )
    assert result.loc[3, "risk_net_return"] == 0
    assert result.loc[4, "risk_net_return"] == 0


def test_daily_positions_respect_symbol_trade_cap():
    day = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "score": [0.4, 0.3, -0.2, -0.5],
            "forward_return": [0.02, 0.01, -0.01, -0.02],
        }
    )
    counts = {"A": 1}

    positions = build_daily_positions(
        day,
        symbol_trade_counts=counts,
        config=BacktestConfig(
            long_fraction=0.25,
            short_fraction=0.25,
            max_position_weight=1.0,
            max_symbol_trades=1,
        ),
    )

    assert "A" not in set(positions["symbol"])
    assert "B" in set(positions["symbol"])


def test_long_only_capital_backtest_uses_whole_shares_and_cash_drag():
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "symbol": ["A", "B", "C"],
            "score": [0.9, 0.8, 0.1],
            "forward_return": [0.10, 0.05, -0.01],
            "adj_close": [150.0, 900.0, 50.0],
        }
    )

    result = long_only_capital_backtest(
        scored,
        config=BacktestConfig(
            initial_capital=2_000,
            long_fraction=0.67,
            max_position_weight=0.20,
            transaction_cost_bps=0,
            slippage_bps=0,
        ),
    )

    assert result.loc[0, "positions"] == 1
    assert result.loc[0, "invested"] == 300
    assert result.loc[0, "capital"] == 2_030


def test_long_only_capital_backtest_respects_rebalance_interval():
    scored = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "symbol": ["A", "A", "A"],
            "score": [0.9, 0.9, 0.9],
            "forward_return": [0.10, 0.10, 0.10],
            "adj_close": [100.0, 100.0, 100.0],
        }
    )

    result = long_only_capital_backtest(
        scored,
        config=BacktestConfig(
            initial_capital=2_000,
            long_fraction=1.0,
            max_position_weight=0.20,
            rebalance_interval_days=2,
            transaction_cost_bps=0,
            slippage_bps=0,
        ),
    )

    assert result.loc[0, "positions"] == 1
    assert result.loc[1, "positions"] == 0
    assert result.loc[2, "positions"] == 1


def test_event_based_long_only_backtest_writes_fills_and_skips():
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "symbol": ["A", "B"],
            "score": [0.9, 0.8],
            "next_open": [100.0, 1_000.0],
            "exit_close": [110.0, 1_100.0],
            "avg_dollar_volume_20d": [1_000_000.0, 1_000_000.0],
        }
    )

    equity, ledger = event_based_long_only_backtest(
        signals,
        config=BacktestConfig(
            initial_capital=2_000,
            long_fraction=1.0,
            max_position_weight=0.20,
            transaction_cost_bps=0,
            slippage_bps=0,
        ),
    )

    assert equity.loc[0, "capital"] == 2_040
    assert set(ledger["status"]) == {"filled", "skipped"}
    assert "size_too_small" in set(ledger["skip_reason"])


def test_event_based_long_only_backtest_respects_min_score():
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "symbol": ["A", "B"],
            "score": [0.02, -0.01],
            "next_open": [100.0, 100.0],
            "exit_close": [110.0, 110.0],
            "avg_dollar_volume_20d": [1_000_000.0, 1_000_000.0],
        }
    )

    _, ledger = event_based_long_only_backtest(
        signals,
        config=BacktestConfig(
            initial_capital=2_000,
            long_fraction=1.0,
            max_position_weight=0.20,
            transaction_cost_bps=0,
            slippage_bps=0,
            min_score=0.0,
        ),
    )

    assert "score_below_threshold" in set(ledger["skip_reason"])


def test_event_based_long_only_backtest_applies_drawdown_cooldown():
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "symbol": ["A", "A", "A"],
            "score": [0.9, 0.9, 0.9],
            "next_open": [100.0, 100.0, 100.0],
            "exit_close": [50.0, 150.0, 150.0],
            "avg_dollar_volume_20d": [1_000_000.0, 1_000_000.0, 1_000_000.0],
        }
    )

    equity, ledger = event_based_long_only_backtest(
        signals,
        config=BacktestConfig(
            initial_capital=2_000,
            long_fraction=1.0,
            max_position_weight=1.0,
            transaction_cost_bps=0,
            slippage_bps=0,
            max_drawdown_stop=0.20,
            cooldown_days=1,
        ),
    )

    assert equity.loc[0, "capital"] == 1_000
    assert equity.loc[1, "positions"] == 0
    assert equity.loc[2, "positions"] == 1
    assert len(ledger.loc[ledger["status"] == "filled"]) == 2


def test_basic_trading_metrics():
    returns = pd.Series([0.1, -0.05, 0.02])

    assert hit_rate(returns) == 2 / 3
    assert round(max_drawdown(returns), 4) == -0.05
