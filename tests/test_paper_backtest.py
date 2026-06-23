import pandas as pd

from signalforge.paper import PaperTradingConfig
from signalforge.paper_backtest import run_paper_style_backtest


def test_paper_style_backtest_uses_paper_lifecycle():
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-29"]),
            "symbol": ["A", "A", "B"],
            "score": [0.03, 0.04, 0.02],
            "adj_close": [100.0, 101.0, 50.0],
            "sector": ["Tech", "Tech", "Health"],
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-29", "2024-01-30"]
            ),
            "symbol": ["A", "A", "A", "A", "B"],
            "open": [100.0, 101.0, 105.0, 110.0, 51.0],
            "high": [100.0, 101.0, 105.0, 110.0, 51.0],
            "low": [100.0, 101.0, 105.0, 110.0, 51.0],
            "close": [100.0, 101.0, 105.0, 110.0, 51.0],
            "adj_close": [100.0, 101.0, 105.0, 110.0, 51.0],
            "volume": [1_000_000] * 5,
        }
    )

    result = run_paper_style_backtest(
        signals,
        prices,
        config=PaperTradingConfig(
            initial_capital=1_000,
            position_weight=0.20,
            long_fraction=1.0,
            min_score=0.0,
            transaction_cost_bps=0,
            slippage_bps=0,
        ),
        score_col="score",
    )

    assert "planned" not in set(result.ledger.loc[result.ledger["symbol"] == "A", "status"])
    assert "symbol_already_open" in set(result.ledger["skip_reason"])
    assert "closed" in set(result.ledger["status"])
    assert result.summary["closed_positions"] == 1
    assert result.summary["ending_equity"] > 1_000
    assert {"equity", "net_return", "drawdown"}.issubset(result.daily_equity.columns)


def test_paper_style_backtest_enforces_cash_at_fill_time():
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "symbol": ["A", "B"],
            "score": [0.03, 0.02],
            "adj_close": [100.0, 100.0],
            "sector": ["Tech", "Health"],
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]
            ),
            "symbol": ["A", "A", "B", "B"],
            "open": [100.0, 199.0, 100.0, 199.0],
            "high": [100.0, 199.0, 100.0, 199.0],
            "low": [100.0, 199.0, 100.0, 199.0],
            "close": [100.0, 199.0, 100.0, 199.0],
            "adj_close": [100.0, 199.0, 100.0, 199.0],
            "volume": [1_000_000] * 4,
        }
    )

    result = run_paper_style_backtest(
        signals,
        prices,
        config=PaperTradingConfig(
            initial_capital=400,
            position_weight=0.50,
            long_fraction=1.0,
            min_score=0.0,
            transaction_cost_bps=0,
            slippage_bps=0,
        ),
        score_col="score",
    )

    assert list(result.ledger["status"]) == ["open", "skipped"]
    assert result.ledger.loc[1, "skip_reason"] == "insufficient_cash_at_fill"
    assert result.daily_equity["cash"].min() >= 0
