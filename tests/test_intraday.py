import pandas as pd

from signalforge.intraday import (
    evaluate_intraday_risk_exits,
    normalize_yfinance_intraday,
)
from signalforge.paper import (
    ExitRulesConfig,
    PaperTradingConfig,
    StopLossConfig,
    TrailingStopConfig,
    build_planned_orders,
    reconcile_fills,
)


def _prices(symbol: str, dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "symbol": [symbol] * len(dates),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "adj_close": closes,
            "volume": [1_000_000] * len(dates),
        }
    )


def _open_ledger(config: PaperTradingConfig, prices: pd.DataFrame) -> pd.DataFrame:
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["A"],
            "sector": ["Tech"],
            "score": [0.02],
            "adj_close": [100.0],
        }
    )
    ledger = build_planned_orders(scored, config=config)
    return reconcile_fills(ledger, prices, config=config)


def _marks(price: float, high: float | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-03 10:15:00"]),
            "symbol": ["A"],
            "price": [price],
            "open": [price],
            "high": [high if high is not None else price],
            "low": [price],
            "close": [price],
            "volume": [100_000],
        }
    )


def test_intraday_stop_loss_closes_position():
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
        exit_rules=ExitRulesConfig(stop_loss=StopLossConfig(enabled=True, pct=-0.08)),
    )
    prices = _prices("A", ["2024-01-01", "2024-01-02"], [100, 100])
    ledger = _open_ledger(config, prices)

    evaluated, decisions = evaluate_intraday_risk_exits(
        ledger,
        _marks(91),
        daily_prices=prices,
        config=config,
    )

    assert evaluated.loc[0, "status"] == "closed"
    assert evaluated.loc[0, "exit_reason"] == "intraday_stop_loss"
    assert decisions.loc[0, "triggered"]


def test_intraday_trailing_stop_uses_prior_daily_high():
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
        exit_rules=ExitRulesConfig(
            trailing_stop=TrailingStopConfig(
                enabled=True,
                activate_at_return=0.12,
                trail_from_high_pct=-0.06,
            )
        ),
    )
    prices = _prices("A", ["2024-01-01", "2024-01-02", "2024-01-03"], [100, 100, 120])
    ledger = _open_ledger(config, prices)

    evaluated, decisions = evaluate_intraday_risk_exits(
        ledger,
        _marks(112),
        daily_prices=prices,
        config=config,
    )

    assert evaluated.loc[0, "status"] == "closed"
    assert evaluated.loc[0, "exit_reason"] == "intraday_trailing_stop"
    assert evaluated.loc[0, "highest_close_since_entry"] == 120
    assert decisions.loc[0, "trailing_stop_activated"]


def test_intraday_missing_mark_does_not_close_position():
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    prices = _prices("A", ["2024-01-01", "2024-01-02"], [100, 100])
    ledger = _open_ledger(config, prices)

    evaluated, decisions = evaluate_intraday_risk_exits(
        ledger,
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-03 10:15:00"]),
                "symbol": ["B"],
                "price": [90],
            }
        ),
        daily_prices=prices,
        config=config,
    )

    assert evaluated.loc[0, "status"] == "open"
    assert decisions.loc[0, "action"] == "no_mark"


def test_normalize_yfinance_intraday_multi_symbol_frame():
    columns = pd.MultiIndex.from_product(
        [["A"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )
    raw = pd.DataFrame(
        [[100, 102, 99, 101, 101, 1000]],
        index=pd.to_datetime(["2024-01-03 10:15:00"]),
        columns=columns,
    )

    marks = normalize_yfinance_intraday(raw, ["A"])

    assert marks.loc[0, "symbol"] == "A"
    assert marks.loc[0, "price"] == 101
    assert marks.loc[0, "source"] == "yahoo_intraday"
