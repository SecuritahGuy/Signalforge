import numpy as np
import pandas as pd
import pytest

from signalforge.paper import (
    ExitRulesConfig,
    PaperTradingConfig,
    RebalanceConfig,
    ScoreDeteriorationConfig,
    SectorStopConfig,
    StopLossConfig,
    TimeDecayConfig,
    TrailingStopConfig,
    TrailingVolatilityStopConfig,
    build_planned_orders,
    mark_paper_positions,
    reconcile_exits,
    reconcile_fills,
    summarize_paper_account,
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


def _open_ledger(
    *,
    symbol: str = "A",
    score: float = 0.02,
    prices: pd.DataFrame | None = None,
    config: PaperTradingConfig | None = None,
) -> pd.DataFrame:
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": [symbol],
            "sector": ["Tech"],
            "score": [score],
            "adj_close": [100.0],
        }
    )
    cfg = config or PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    fill_prices = prices if prices is not None else _prices(symbol, ["2024-01-01", "2024-01-02"], [100, 100])
    ledger = build_planned_orders(scored, config=cfg)
    return reconcile_fills(ledger, fill_prices, config=cfg)


def test_build_planned_orders_creates_persistent_schema():
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
            "symbol": ["A", "B", "C"],
            "sector": ["Tech", "Health", "Energy"],
            "score": [0.03, 0.02, -0.01],
            "adj_close": [100.0, 900.0, 50.0],
        }
    )

    ledger = build_planned_orders(
        scored,
        config=PaperTradingConfig(
            initial_capital=2_000,
            position_weight=0.10,
            long_fraction=1.0,
            min_score=0.0,
            horizon=20,
        ),
    )

    assert ledger.loc[0, "status"] == "planned"
    assert ledger.loc[0, "shares"] == 2
    assert ledger.loc[1, "status"] == "skipped"
    assert ledger.loc[1, "skip_reason"] == "size_too_small"
    assert ledger.loc[2, "skip_reason"] == "score_below_threshold"
    assert ledger.loc[0, "target_exit_date"] == pd.Timestamp("2024-01-29")


def test_build_planned_orders_respects_available_cash_and_open_symbols():
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "symbol": ["A", "B"],
            "sector": ["Tech", "Health"],
            "score": [0.03, 0.02],
            "adj_close": [100.0, 100.0],
        }
    )

    ledger = build_planned_orders(
        scored,
        config=PaperTradingConfig(
            initial_capital=2_000,
            position_weight=0.10,
            long_fraction=1.0,
            min_score=0.0,
        ),
        available_cash=150,
        excluded_symbols={"A"},
    )

    assert ledger.loc[0, "skip_reason"] == "symbol_already_open"
    assert ledger.loc[1, "skip_reason"] == "insufficient_cash"


def test_reconcile_fills_and_exits_updates_paper_account():
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["A"],
            "sector": ["Tech"],
            "score": [0.03],
            "adj_close": [100.0],
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-29"]),
            "symbol": ["A", "A", "A"],
            "open": [100.0, 101.0, 110.0],
            "high": [101.0, 102.0, 111.0],
            "low": [99.0, 100.0, 109.0],
            "close": [100.0, 101.0, 110.0],
            "adj_close": [100.0, 101.0, 110.0],
            "volume": [1_000_000, 1_000_000, 1_000_000],
        }
    )
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
    )

    ledger = build_planned_orders(scored, config=config)
    ledger = reconcile_fills(ledger, prices, config=config)
    assert ledger.loc[0, "status"] == "open"
    assert ledger.loc[0, "fill_date"] == pd.Timestamp("2024-01-02")
    assert ledger.loc[0, "entry_price"] == 101

    ledger = reconcile_exits(ledger, prices, config=config)
    assert ledger.loc[0, "status"] == "closed"
    assert ledger.loc[0, "exit_date"] == pd.Timestamp("2024-01-29")
    assert ledger.loc[0, "net_pnl"] == 18

    summary = summarize_paper_account(ledger, prices, initial_capital=2_000)
    assert summary["closed_positions"] == 1
    assert summary["realized_pnl"] == 18
    assert summary["equity"] == 2_018


def test_reconcile_fills_skips_orders_that_exceed_fill_time_cash():
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "symbol": ["A", "B"],
            "sector": ["Tech", "Health"],
            "score": [0.03, 0.02],
            "adj_close": [100.0, 100.0],
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
            "volume": [1_000_000, 1_000_000, 1_000_000, 1_000_000],
        }
    )
    config = PaperTradingConfig(
        initial_capital=400,
        position_weight=0.50,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
    )

    ledger = build_planned_orders(scored, config=config)
    ledger = reconcile_fills(ledger, prices, config=config)

    assert list(ledger["status"]) == ["open", "skipped"]
    assert ledger.loc[1, "skip_reason"] == "insufficient_cash_at_fill"

    summary = summarize_paper_account(ledger, prices, initial_capital=400)
    assert summary["cash"] == 2
    assert summary["equity"] == 400


def test_reconcile_fills_dedupes_active_symbol_before_marking_account():
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "symbol": ["A", "A"],
            "sector": ["Tech", "Tech"],
            "score": [0.03, 0.02],
            "adj_close": [100.0, 100.0],
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["A", "A"],
            "open": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 100.0],
            "close": [100.0, 100.0],
            "adj_close": [100.0, 100.0],
            "volume": [1_000_000, 1_000_000],
        }
    )
    config = PaperTradingConfig(
        initial_capital=500,
        position_weight=0.20,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
    )

    ledger = build_planned_orders(scored, config=config)
    ledger = reconcile_fills(ledger, prices, config=config)

    assert list(ledger["status"]) == ["open", "skipped"]
    assert ledger.loc[1, "skip_reason"] == "duplicate_active_symbol"
    assert summarize_paper_account(ledger, prices, initial_capital=500)["open_positions"] == 1


def test_mark_paper_positions_labels_hold_and_waiting_fill():
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["A", "B"],
            "sector": ["Tech", "Health"],
            "score": [0.03, 0.02],
            "adj_close": [100.0, 50.0],
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "symbol": ["A", "A", "A"],
            "open": [100.0, 101.0, 104.0],
            "high": [101.0, 102.0, 105.0],
            "low": [99.0, 100.0, 103.0],
            "close": [100.0, 101.0, 104.0],
            "adj_close": [100.0, 101.0, 104.0],
            "volume": [1_000_000, 1_000_000, 1_000_000],
        }
    )
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    open_ledger = build_planned_orders(scored.head(1), config=config)
    open_ledger = reconcile_fills(open_ledger, prices, config=config)
    planned_ledger = build_planned_orders(scored.tail(1), config=config)
    ledger = pd.concat([open_ledger, planned_ledger], ignore_index=True)

    marks = mark_paper_positions(ledger, prices)

    assert set(marks["action"]) == {"hold", "waiting_for_fill"}
    open_mark = marks.loc[marks["status"] == "open"].iloc[0]
    assert open_mark["latest_price"] == 104
    assert open_mark["unrealized_pnl"] == 6


def test_horizon_exit_records_reason():
    prices = _prices("A", ["2024-01-01", "2024-01-02", "2024-01-29"], [100, 100, 110])
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    ledger = _open_ledger(prices=prices, config=config)

    ledger = reconcile_exits(ledger, prices, config=config)

    assert ledger.loc[0, "status"] == "closed"
    assert ledger.loc[0, "exit_reason"] == "horizon"


def test_stop_loss_triggers_and_does_not_trigger_above_threshold():
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
        exit_rules=ExitRulesConfig(stop_loss=StopLossConfig(enabled=True, pct=-0.08)),
    )
    trigger_prices = _prices("A", ["2024-01-01", "2024-01-02", "2024-01-03"], [100, 100, 91])
    hold_prices = _prices("A", ["2024-01-01", "2024-01-02", "2024-01-03"], [100, 100, 93])

    triggered = reconcile_exits(_open_ledger(prices=trigger_prices, config=config), trigger_prices, config=config)
    held = reconcile_exits(_open_ledger(prices=hold_prices, config=config), hold_prices, config=config)

    assert triggered.loc[0, "status"] == "closed"
    assert triggered.loc[0, "exit_reason"] == "stop_loss"
    assert held.loc[0, "status"] == "open"


def test_trailing_stop_activation_and_exit():
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
    before_activation = _prices("A", ["2024-01-01", "2024-01-02", "2024-01-03"], [100, 100, 110])
    after_drawdown = _prices(
        "A",
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        [100, 100, 120, 112],
    )

    held = reconcile_exits(_open_ledger(prices=before_activation, config=config), before_activation, config=config)
    exited = reconcile_exits(_open_ledger(prices=after_drawdown, config=config), after_drawdown, config=config)

    assert held.loc[0, "status"] == "open"
    assert exited.loc[0, "status"] == "closed"
    assert exited.loc[0, "exit_reason"] == "trailing_stop"
    assert exited.loc[0, "trailing_stop_activated"]
    assert exited.loc[0, "highest_close_since_entry"] == 120


def test_score_deterioration_respects_min_hold_and_absolute_threshold():
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
        exit_rules=ExitRulesConfig(
            score_deterioration=ScoreDeteriorationConfig(
                enabled=True,
                min_days_held=5,
                exit_below_score=0.005,
                exit_if_score_declines_pct=0.60,
            )
        ),
    )
    prices = _prices(
        "A",
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"],
        [100, 100, 101, 101, 101, 101, 101],
    )
    early_scores = pd.DataFrame({"date": pd.to_datetime(["2024-01-03"]), "symbol": ["A"], "score": [0.001]})
    late_scores = pd.DataFrame({"date": pd.to_datetime(["2024-01-09"]), "symbol": ["A"], "score": [0.001]})

    held = reconcile_exits(_open_ledger(prices=prices, config=config), prices, scores=early_scores, config=config)
    exited = reconcile_exits(_open_ledger(prices=prices, config=config), prices, scores=late_scores, config=config)

    assert held.loc[0, "status"] == "open"
    assert exited.loc[0, "status"] == "closed"
    assert exited.loc[0, "exit_reason"] == "score_deterioration"


def test_score_deterioration_percent_decline_and_missing_score_hold():
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
        exit_rules=ExitRulesConfig(
            score_deterioration=ScoreDeteriorationConfig(
                enabled=True,
                min_days_held=1,
                exit_below_score=-1.0,
                exit_if_score_declines_pct=0.60,
            )
        ),
    )
    prices = _prices("A", ["2024-01-01", "2024-01-02", "2024-01-03"], [100, 100, 101])
    scores = pd.DataFrame({"date": pd.to_datetime(["2024-01-03"]), "symbol": ["A"], "score": [0.008]})

    missing = reconcile_exits(_open_ledger(prices=prices, config=config), prices, config=config)
    exited = reconcile_exits(_open_ledger(score=0.02, prices=prices, config=config), prices, scores=scores, config=config)

    assert missing.loc[0, "status"] == "open"
    assert exited.loc[0, "status"] == "closed"
    assert exited.loc[0, "exit_reason"] == "score_deterioration"


def test_exit_rule_priority_prefers_stop_loss_over_trailing_stop_and_horizon():
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
        exit_rules=ExitRulesConfig(
            stop_loss=StopLossConfig(enabled=True, pct=-0.08),
            trailing_stop=TrailingStopConfig(
                enabled=True,
                activate_at_return=0.01,
                trail_from_high_pct=-0.01,
            ),
        ),
    )
    prices = _prices("A", ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], [100, 100, 120, 80])
    ledger = _open_ledger(prices=prices, config=config)

    ledger = reconcile_exits(ledger, prices, config=config)

    assert ledger.loc[0, "exit_reason"] == "stop_loss"


def test_rebalance_exit_triggers_on_low_score_after_min_hold():
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
        exit_rules=ExitRulesConfig(
            rebalance=RebalanceConfig(
                enabled=True,
                min_days_held=3,
                exit_below_score=0.01,
            )
        ),
    )
    prices = _prices(
        "A",
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        [100, 100, 101, 101, 101],
    )
    early_scores = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-03"]), "symbol": ["A"], "score": [0.005]}
    )
    late_scores = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-05"]), "symbol": ["A"], "score": [0.005]}
    )

    held = reconcile_exits(
        _open_ledger(prices=prices, config=config), prices, scores=early_scores, config=config
    )
    exited = reconcile_exits(
        _open_ledger(prices=prices, config=config), prices, scores=late_scores, config=config
    )

    assert held.loc[0, "status"] == "open"
    assert exited.loc[0, "status"] == "closed"
    assert exited.loc[0, "exit_reason"] == "rebalance"


def test_rebalance_does_not_exit_when_score_is_above_threshold():
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
        exit_rules=ExitRulesConfig(
            rebalance=RebalanceConfig(
                enabled=True,
                min_days_held=1,
                exit_below_score=0.01,
            )
        ),
    )
    prices = _prices("A", ["2024-01-01", "2024-01-02", "2024-01-03"], [100, 100, 101])
    scores = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-03"]), "symbol": ["A"], "score": [0.02]}
    )

    result = reconcile_exits(
        _open_ledger(prices=prices, config=config), prices, scores=scores, config=config
    )

    assert result.loc[0, "status"] == "open"


def test_rebalance_does_not_exit_when_disabled():
    prices = _prices("A", ["2024-01-01", "2024-01-02", "2024-01-29"], [100, 100, 110])
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
        exit_rules=ExitRulesConfig(
            rebalance=RebalanceConfig(enabled=False, min_days_held=1, exit_below_score=0.01),
            stop_loss=StopLossConfig(enabled=False),
            trailing_stop=TrailingStopConfig(enabled=False),
            score_deterioration=ScoreDeteriorationConfig(enabled=False),
        ),
    )
    scores = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-03"]), "symbol": ["A"], "score": [0.005]}
    )

    result = reconcile_exits(
        _open_ledger(prices=prices, config=config), prices, scores=scores, config=config
    )

    assert result.loc[0, "status"] == "closed"
    assert result.loc[0, "exit_reason"] == "horizon"


def test_old_ledger_without_exit_columns_loads_successfully():
    prices = _prices("A", ["2024-01-01", "2024-01-02", "2024-01-29"], [100, 100, 100])
    ledger = _open_ledger(prices=prices).drop(
        columns=[
            "actual_exit_trigger_date",
            "exit_reason",
            "exit_signal_value",
            "exit_rule_version",
            "highest_close_since_entry",
            "trailing_stop_activated",
        ]
    )

    reconciled = reconcile_exits(ledger, prices)

    assert "exit_reason" in reconciled.columns
    assert reconciled.loc[0, "exit_reason"] == "horizon"


def _prices_from_closes(closes: list[float], start: str = "2023-11-20") -> pd.DataFrame:
    """Build prices DataFrame from a list of close values."""
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "A",
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "adj_close": closes,
            "volume": 1_000_000,
        }
    )


def test_trailing_volatility_stop_activation_and_exit():
    """Trailing volatility stop activates after high return and exits on vol-based drawdown."""
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
        exit_rules=ExitRulesConfig(
            trailing_volatility_stop=TrailingVolatilityStopConfig(
                enabled=True,
                activate_at_return=0.10,
                volatility_lookback=3,
                volatility_multiple=0.5,
                tightest_trail_pct=-0.03,
                widest_trail_pct=-0.05,
            ),
            trailing_stop=TrailingStopConfig(enabled=False),
        ),
    )
    # 45 flat days at 100 for vol calc + lookback, then jump + drop
    closes = [100.0] * 45 + [100, 115, 108]
    prices = _prices_from_closes(closes)
    exited = reconcile_exits(
        _open_ledger(prices=prices, config=config), prices, config=config
    )
    assert exited.loc[0, "status"] == "closed"
    assert exited.loc[0, "exit_reason"] == "trailing_volatility_stop"
    assert exited.loc[0, "trailing_stop_activated"]


def test_trailing_volatility_stop_holds_within_vol_based_threshold():
    """Trailing volatility stop does not exit when drawdown is within vol-based threshold."""
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
        exit_rules=ExitRulesConfig(
            trailing_volatility_stop=TrailingVolatilityStopConfig(
                enabled=True,
                activate_at_return=0.10,
                volatility_lookback=3,
                volatility_multiple=0.5,
                tightest_trail_pct=-0.03,
                widest_trail_pct=-0.05,
            ),
            trailing_stop=TrailingStopConfig(enabled=False),
        ),
    )
    # Same flat base, jump to 115, but smaller drop (within trail)
    closes = [100.0] * 45 + [100, 115, 113]
    prices = _prices_from_closes(closes)
    held = reconcile_exits(
        _open_ledger(prices=prices, config=config), prices, config=config
    )
    assert held.loc[0, "status"] == "open"


def test_trailing_volatility_stop_priority_over_horizon():
    """Trailing volatility stop fires before horizon when both could apply."""
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        transaction_cost_bps=0,
        slippage_bps=0,
        exit_rules=ExitRulesConfig(
            trailing_volatility_stop=TrailingVolatilityStopConfig(
                enabled=True,
                activate_at_return=0.05,
                volatility_lookback=3,
                volatility_multiple=0.5,
                tightest_trail_pct=-0.03,
                widest_trail_pct=-0.05,
            ),
            trailing_stop=TrailingStopConfig(enabled=False),
        ),
    )
    # Jump high to activate, then drop to trigger
    closes = [100.0] * 45 + [100, 130, 120]
    prices = _prices_from_closes(closes)
    exited = reconcile_exits(
        _open_ledger(prices=prices, config=config), prices, config=config
    )
    assert exited.loc[0, "status"] == "closed"
    assert exited.loc[0, "exit_reason"] == "trailing_volatility_stop"


def test_time_decay_exit_triggers_after_max_hold():
    """Time decay exit fires after days_held reaches max_hold computed from entry score."""
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        horizon=60,
        transaction_cost_bps=0,
        slippage_bps=0,
        exit_rules=ExitRulesConfig(
            time_decay=TimeDecayConfig(
                enabled=True,
                half_life_days=10,
                min_days_hold=2,
                min_score_for_decay=0.005,
            ),
        ),
    )
    # entry_score=0.02 → max_hold = 10 * log2(0.02/0.005) = 20 business days
    # Use 70 flat days: fill ~Jan 2, 20+ days later triggers time_decay
    closes = [100.0] * 70
    prices = _prices_from_closes(closes)
    exited = reconcile_exits(
        _open_ledger(prices=prices, config=config), prices, config=config
    )
    assert exited.loc[0, "status"] == "closed"
    assert exited.loc[0, "exit_reason"] == "time_decay"
    assert exited.loc[0, "exit_signal_value"] > 0


def test_time_decay_does_not_exit_before_max_hold():
    """Time decay does not fire when days_held is below max_hold."""
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        horizon=60,
        transaction_cost_bps=0,
        slippage_bps=0,
        exit_rules=ExitRulesConfig(
            time_decay=TimeDecayConfig(
                enabled=True,
                half_life_days=10,
                min_days_hold=2,
                min_score_for_decay=0.005,
            ),
        ),
    )
    # Only 5 business days after fill → well below max_hold of 20
    closes = [100.0] * 45 + [100] * 5
    prices = _prices_from_closes(closes)
    held = reconcile_exits(
        _open_ledger(prices=prices, config=config), prices, config=config
    )
    assert held.loc[0, "status"] == "open"


def test_time_decay_does_not_fire_when_disabled():
    """Time decay does not fire when configured disabled."""
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        horizon=5,
        transaction_cost_bps=0,
        slippage_bps=0,
        exit_rules=ExitRulesConfig(
            time_decay=TimeDecayConfig(enabled=False),
        ),
    )
    closes = [100.0] * 70
    prices = _prices_from_closes(closes)
    result = reconcile_exits(
        _open_ledger(prices=prices, config=config), prices, config=config
    )
    # Falls through to horizon (horizon=5, so ~5 business days)
    assert result.loc[0, "status"] == "closed"
    assert result.loc[0, "exit_reason"] == "horizon"


def _sector_universe() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["A", "B", "C", "D"],
        "sector": ["Tech", "Tech", "Tech", "Health"],
    })


def _sector_prices(drop_start: int) -> pd.DataFrame:
    """Build flat then declining prices for 4 symbols to test sector-stop.

    After ``drop_start``, Tech symbols (A, B, C) decline by ~5% each day
    so the rolling sector return falls below -5%.  Symbol D (Health)
    stays flat to test that only the relevant sector is affected.
    """
    dates = pd.date_range("2024-01-01", periods=drop_start + 6, freq="B")
    rows = []
    for sym in ["A", "B", "C", "D"]:
        for i, d in enumerate(dates):
            if sym == "D":
                close = 100.0
            elif i < drop_start:
                close = 100.0
            else:
                close = 100.0 * (0.95 ** (i - drop_start + 1))
            rows.append({
                "date": d,
                "symbol": sym,
                "open": close, "high": close, "low": close,
                "close": close, "adj_close": close,
                "volume": 1_000_000,
            })
    return pd.DataFrame(rows)


def test_sector_stop_triggers_on_sector_decline():
    """Sector stop fires when the rolling mean sector return is below threshold."""
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        horizon=60,
        transaction_cost_bps=0,
        slippage_bps=0,
        exit_rules=ExitRulesConfig(
            sector_stop=SectorStopConfig(
                enabled=True,
                sector_decline_pct=-0.05,
                lookback_days=3,
                min_sector_records=2,
            ),
        ),
    )
    universe = _sector_universe()
    prices = _sector_prices(drop_start=45)
    exited = reconcile_exits(
        _open_ledger(prices=prices, config=config), prices,
        config=config, universe=universe,
    )
    assert exited.loc[0, "status"] == "closed"
    assert exited.loc[0, "exit_reason"] == "sector_stop"


def test_sector_stop_holds_when_sector_stable():
    """Sector stop does not fire when sector return is above threshold."""
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        horizon=60,
        transaction_cost_bps=0,
        slippage_bps=0,
        exit_rules=ExitRulesConfig(
            sector_stop=SectorStopConfig(
                enabled=True,
                sector_decline_pct=-0.05,
                lookback_days=3,
                min_sector_records=2,
            ),
        ),
    )
    universe = _sector_universe()
    # All prices flat (only 20 dates) → sector return ~0, above threshold
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    rows = []
    for sym in ["A", "B", "C", "D"]:
        for d in dates:
            rows.append({
                "date": d, "symbol": sym,
                "open": 100.0, "high": 100.0, "low": 100.0,
                "close": 100.0, "adj_close": 100.0, "volume": 1_000_000,
            })
    prices = pd.DataFrame(rows)
    held = reconcile_exits(
        _open_ledger(prices=prices, config=config), prices,
        config=config, universe=universe,
    )
    assert held.loc[0, "status"] == "open"


def test_sector_stop_does_not_fire_when_disabled():
    """Sector stop does not fire when configured disabled."""
    config = PaperTradingConfig(
        initial_capital=2_000,
        position_weight=0.10,
        long_fraction=1.0,
        min_score=0.0,
        horizon=5,
        transaction_cost_bps=0,
        slippage_bps=0,
        exit_rules=ExitRulesConfig(
            sector_stop=SectorStopConfig(enabled=False),
        ),
    )
    universe = _sector_universe()
    prices = _sector_prices(drop_start=45)
    result = reconcile_exits(
        _open_ledger(prices=prices, config=config), prices,
        config=config, universe=universe,
    )
    # Falls through to horizon (horizon=5)
    assert result.loc[0, "exit_reason"] == "horizon"
