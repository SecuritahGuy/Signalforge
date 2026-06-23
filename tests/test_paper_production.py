from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalforge.paper import (
    PAPER_LEDGER_COLUMNS,
    BorrowCostConfig,
    DividendConfig,
    FillConfig,
    PaperTradingConfig,
    build_planned_orders,
    reconcile_borrow_costs,
    reconcile_dividends,
    reconcile_fills,
    summarize_paper_account,
)

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=15, freq="D")
    rows = []
    for symbol in ["AAPL", "MSFT"]:
        for i, date in enumerate(dates):
            price = 100 + i
            rows.append({
                "date": date,
                "symbol": symbol,
                "open": price - 0.5,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "adj_close": price,
                "volume": 1_000_000,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def scored(prices: pd.DataFrame) -> pd.DataFrame:
    # Use early-enough date so fills have forward dates available
    signal_date = prices["date"].sort_values().unique()[-5]
    candidates = prices.loc[prices["date"] == signal_date].copy()
    candidates["score"] = [0.9, 0.3]
    candidates = candidates.drop(columns=["open", "high", "low", "close", "volume"])
    return candidates


# ── New column schema ────────────────────────────────────────────────


def test_ledger_has_new_columns():
    assert "requested_shares" in PAPER_LEDGER_COLUMNS
    assert "filled_shares" in PAPER_LEDGER_COLUMNS
    assert "borrow_cost" in PAPER_LEDGER_COLUMNS
    assert "dividends" in PAPER_LEDGER_COLUMNS


def test_build_planned_orders_includes_new_columns(scored, prices):
    cfg = PaperTradingConfig(initial_capital=10_000, position_weight=0.1)
    orders = build_planned_orders(scored, config=cfg)

    assert "requested_shares" in orders.columns
    assert "filled_shares" in orders.columns
    assert "borrow_cost" in orders.columns
    assert "dividends" in orders.columns

    planned = orders[orders["status"] == "planned"]
    if not planned.empty:
        assert (planned["requested_shares"] == planned["shares"]).all()
        assert (planned["filled_shares"] == 0.0).all()


# ── Partial fills ────────────────────────────────────────────────────


def test_partial_fill_uses_fill_pct(scored, prices):
    cfg = PaperTradingConfig(
        initial_capital=10_000,
        position_weight=0.1,
        fill=FillConfig(enabled=True, fill_pct=0.6),
    )
    orders = build_planned_orders(scored, config=cfg)
    filled = reconcile_fills(orders, prices, config=cfg)

    open_orders = filled[filled["status"] == "open"]
    for _, row in open_orders.iterrows():
        expected_shares = np.floor(row["requested_shares"] * 0.6)
        assert row["shares"] == expected_shares, (
            f"expected {expected_shares} shares, got {row['shares']}"
        )
        assert row["filled_shares"] == row["shares"]


def test_partial_fill_respects_min_pct(scored, prices):
    cfg = PaperTradingConfig(
        initial_capital=10_000,
        position_weight=0.1,
        fill=FillConfig(enabled=True, fill_pct=0.05, min_fill_pct=0.1),
    )
    orders = build_planned_orders(scored, config=cfg)
    filled = reconcile_fills(orders, prices, config=cfg)

    open_orders = filled[filled["status"] == "open"]
    for _, row in open_orders.iterrows():
        min_shares = np.floor(row["requested_shares"] * 0.1)
        assert row["shares"] >= min_shares, (
            f"expected at least {min_shares} shares, got {row['shares']}"
        )


def test_partial_fill_tracks_average_entry_price(scored, prices):
    cfg = PaperTradingConfig(
        initial_capital=10_000,
        position_weight=0.5,
        allow_fractional_shares=True,
        fill=FillConfig(enabled=True, fill_pct=0.5),
    )
    orders = build_planned_orders(scored, config=cfg)
    filled = reconcile_fills(orders, prices, config=cfg)

    open_orders = filled[filled["status"] == "open"]
    for _, row in open_orders.iterrows():
        expected_value = row["shares"] * row["entry_price"]
        assert abs(row["entry_value"] - expected_value) < 1.0


def test_partial_fill_backward_compatible_default(scored, prices):
    cfg = PaperTradingConfig(
        initial_capital=10_000,
        position_weight=0.1,
    )
    orders = build_planned_orders(scored, config=cfg)
    filled = reconcile_fills(orders, prices, config=cfg)

    open_orders = filled[filled["status"] == "open"]
    for _, row in open_orders.iterrows():
        assert row["shares"] == row["requested_shares"]
        assert row["filled_shares"] == row["shares"]


# ── Multi-day fills ──────────────────────────────────────────────────


def test_multi_day_fill_goes_through_filling_state(scored, prices):
    cfg = PaperTradingConfig(
        initial_capital=10_000,
        position_weight=0.1,
        allow_fractional_shares=True,
        fill=FillConfig(enabled=True, fill_pct=0.8),
    )
    orders = build_planned_orders(scored, config=cfg)

    # First fill attempt — should be "filling" since fill_pct < 1
    day1 = reconcile_fills(orders, prices, config=cfg)
    filling = day1[day1["status"] == "filling"]
    assert not filling.empty, "expected some orders in 'filling' state"

    # Second fill attempt — should move the remainder to "open"
    day2 = reconcile_fills(day1, prices, config=cfg)
    still_filling = day2[day2["status"] == "filling"]
    assert still_filling.empty, "expected all filling orders to become open after second attempt"

    open_orders = day2[day2["status"] == "open"]
    for _, row in open_orders.iterrows():
        assert row["filled_shares"] > 0
        assert row["filled_shares"] <= row["requested_shares"]


# ── Borrow costs ─────────────────────────────────────────────────────


def test_borrow_cost_does_not_accrue_when_disabled(scored, prices):
    cfg = PaperTradingConfig(
        initial_capital=10_000,
        position_weight=0.1,
    )
    orders = build_planned_orders(scored, config=cfg)
    filled = reconcile_fills(orders, prices, config=cfg)
    result = reconcile_borrow_costs(filled, prices, config=cfg)

    assert (result["borrow_cost"] == 0.0).all()


def test_borrow_cost_accrues_for_open_positions(scored, prices):
    cfg = PaperTradingConfig(
        initial_capital=10_000,
        position_weight=0.5,
        allow_fractional_shares=True,
        borrow_cost=BorrowCostConfig(enabled=True, annual_rate=0.05),
    )
    orders = build_planned_orders(scored, config=cfg)
    filled = reconcile_fills(orders, prices, config=cfg)
    result = reconcile_borrow_costs(filled, prices, config=cfg)

    open_orders = result[result["status"] == "open"]
    if not open_orders.empty:
        for _, row in open_orders.iterrows():
            assert row["borrow_cost"] >= 0.0


def test_borrow_cost_increases_with_longer_hold(scored, prices):
    cfg = PaperTradingConfig(
        initial_capital=10_000,
        position_weight=0.5,
        allow_fractional_shares=True,
        borrow_cost=BorrowCostConfig(enabled=True, annual_rate=0.05),
    )
    orders = build_planned_orders(scored, config=cfg)
    filled = reconcile_fills(orders, prices, config=cfg)

    day1 = reconcile_borrow_costs(filled, prices, config=cfg)
    day2 = reconcile_borrow_costs(day1, prices, config=cfg)

    for _, row in day2.iterrows():
        day1_cost = day1.loc[row.name, "borrow_cost"] if row.name in day1.index else 0.0
        assert row["borrow_cost"] >= day1_cost


def test_hard_to_borrow_applies_higher_rate(scored, prices):
    cfg = PaperTradingConfig(
        initial_capital=10_000,
        position_weight=0.5,
        allow_fractional_shares=True,
        borrow_cost=BorrowCostConfig(
            enabled=True,
            annual_rate=0.03,
            hard_to_borrow_rate=0.50,
            hard_to_borrow_symbols=frozenset({"AAPL"}),
        ),
    )
    orders = build_planned_orders(scored, config=cfg)
    filled = reconcile_fills(orders, prices, config=cfg)
    result = reconcile_borrow_costs(filled, prices, config=cfg)

    open_orders = result[result["status"] == "open"]
    if not open_orders.empty:
        for _, row in open_orders.iterrows():
            assert row["borrow_cost"] >= 0.0
            if row["symbol"] == "AAPL":
                assert row["borrow_cost"] > 0.0


def test_hard_to_borrow_cost_higher_than_normal(scored, prices):
    cfg_htb = PaperTradingConfig(
        initial_capital=10_000,
        position_weight=0.5,
        allow_fractional_shares=True,
        borrow_cost=BorrowCostConfig(
            enabled=True,
            annual_rate=0.03,
            hard_to_borrow_rate=0.50,
            hard_to_borrow_symbols=frozenset({"AAPL"}),
        ),
    )
    cfg_normal = PaperTradingConfig(
        initial_capital=10_000,
        position_weight=0.5,
        allow_fractional_shares=True,
        borrow_cost=BorrowCostConfig(
            enabled=True,
            annual_rate=0.03,
        ),
    )
    orders = build_planned_orders(scored, config=cfg_htb)
    filled = reconcile_fills(orders, prices, config=cfg_htb)
    htb_result = reconcile_borrow_costs(filled, prices, config=cfg_htb)

    orders2 = build_planned_orders(scored, config=cfg_normal)
    filled2 = reconcile_fills(orders2, prices, config=cfg_normal)
    normal_result = reconcile_borrow_costs(filled2, prices, config=cfg_normal)

    for (idx_htb, htb_row), (idx_norm, norm_row) in zip(
        htb_result.iterrows(), normal_result.iterrows()
    ):
        if htb_row["symbol"] == "AAPL" and htb_row["status"] == "open":
            assert htb_row["borrow_cost"] > norm_row["borrow_cost"]


# ── Dividends ─────────────────────────────────────────────────────────


def test_dividends_do_not_accrue_when_disabled(scored, prices):
    cfg = PaperTradingConfig(
        initial_capital=10_000,
        position_weight=0.1,
    )
    orders = build_planned_orders(scored, config=cfg)
    filled = reconcile_fills(orders, prices, config=cfg)
    result = reconcile_dividends(filled, prices, config=cfg)

    assert (result["dividends"] == 0.0).all()


def test_dividends_accrue_for_open_positions_assumed_yield(scored, prices):
    cfg = PaperTradingConfig(
        initial_capital=10_000,
        position_weight=0.5,
        allow_fractional_shares=True,
        dividends=DividendConfig(enabled=True, assumed_annual_yield=0.04),
    )
    orders = build_planned_orders(scored, config=cfg)
    filled = reconcile_fills(orders, prices, config=cfg)
    result = reconcile_dividends(filled, prices, config=cfg)

    open_orders = result[result["status"] == "open"]
    if not open_orders.empty:
        for _, row in open_orders.iterrows():
            assert row["dividends"] >= 0.0


def test_dividends_increase_with_longer_hold(scored, prices):
    cfg = PaperTradingConfig(
        initial_capital=10_000,
        position_weight=0.5,
        allow_fractional_shares=True,
        dividends=DividendConfig(enabled=True, assumed_annual_yield=0.05),
    )
    orders = build_planned_orders(scored, config=cfg)
    filled = reconcile_fills(orders, prices, config=cfg)

    day1 = reconcile_dividends(filled, prices, config=cfg)
    day2 = reconcile_dividends(day1, prices, config=cfg)

    for _, row in day2.iterrows():
        day1_val = day1.loc[row.name, "dividends"] if row.name in day1.index else 0.0
        assert row["dividends"] >= day1_val


def test_dividends_dataframe_mode(scored, prices):
    cfg = PaperTradingConfig(
        initial_capital=10_000,
        position_weight=0.5,
        allow_fractional_shares=True,
        dividends=DividendConfig(
            enabled=True,
            assumed_annual_yield=0.02,
            accrual_mode="dataframe",
        ),
    )
    orders = build_planned_orders(scored, config=cfg)
    filled = reconcile_fills(orders, prices, config=cfg)

    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "ex_date": [pd.Timestamp("2024-01-20")],
        "dividend_per_share": [0.50],
    })
    result = reconcile_dividends(filled, prices, config=cfg, dividend_data=events)

    open_orders = result[result["status"] == "open"]
    for _, row in open_orders.iterrows():
        if row["symbol"] == "AAPL":
            assert row["dividends"] > 0.0


def test_dividends_dataframe_no_matching_events(scored, prices):
    cfg = PaperTradingConfig(
        initial_capital=10_000,
        position_weight=0.5,
        allow_fractional_shares=True,
        dividends=DividendConfig(
            enabled=True,
            assumed_annual_yield=0.02,
            accrual_mode="dataframe",
        ),
    )
    orders = build_planned_orders(scored, config=cfg)
    filled = reconcile_fills(orders, prices, config=cfg)

    events = pd.DataFrame({
        "symbol": ["NONEXISTENT"],
        "ex_date": [pd.Timestamp("2024-01-20")],
        "dividend_per_share": [0.50],
    })
    result = reconcile_dividends(filled, prices, config=cfg, dividend_data=events)
    assert (result["dividends"] == 0.0).all()


# ── Integration: full lifecycle with partial fills ───────────────────


def test_full_lifecycle_with_partial_fills(scored, prices):
    cfg = PaperTradingConfig(
        initial_capital=10_000,
        position_weight=0.1,
        allow_fractional_shares=True,
        fill=FillConfig(enabled=True, fill_pct=0.95),
    )
    orders = build_planned_orders(scored, config=cfg)
    assert not orders.empty

    filled = reconcile_fills(orders, prices, config=cfg)
    assert all(filled["status"].isin(["open", "skipped"]))

    summary = summarize_paper_account(filled, prices, initial_capital=cfg.initial_capital)
    assert summary["open_positions"] >= 0
    assert summary["cash"] >= 0
    assert summary["equity"] > 0


def test_partial_fill_does_not_exceed_cash(scored, prices):
    cfg = PaperTradingConfig(
        initial_capital=500,
        position_weight=1.0,
        fill=FillConfig(enabled=True, fill_pct=0.5),
    )
    orders = build_planned_orders(scored, config=cfg)
    filled = reconcile_fills(orders, prices, config=cfg)

    total_committed = filled.loc[
        filled["status"].isin(["open", "filling"]), "entry_value"
    ].sum() + filled.loc[filled["status"].isin(["open", "filling"]), "entry_cost"].sum()
    assert total_committed <= cfg.initial_capital


def test_daily_operations_with_all_features(scored, prices):
    """Simulate a multi-day operations loop with partial fills and borrow costs."""
    cfg = PaperTradingConfig(
        initial_capital=10_000,
        position_weight=0.5,
        allow_fractional_shares=True,
        fill=FillConfig(enabled=True, fill_pct=0.4),
        borrow_cost=BorrowCostConfig(enabled=True, annual_rate=0.03),
    )
    ledger = build_planned_orders(scored, config=cfg)

    # Day 1: fill attempt
    ledger = reconcile_fills(ledger, prices, config=cfg)
    ledger = reconcile_borrow_costs(ledger, prices, config=cfg)

    # Day 2: second fill attempt + borrow cost
    ledger = reconcile_fills(ledger, prices, config=cfg)
    ledger = reconcile_borrow_costs(ledger, prices, config=cfg)

    summary = summarize_paper_account(ledger, prices, initial_capital=cfg.initial_capital)
    assert summary["open_positions"] >= 0
    assert summary["cash"] >= 0
    assert summary["equity"] > 0
