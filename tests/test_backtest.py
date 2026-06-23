from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalforge.backtest import (
    BacktestConfig,
    _event_equity_row,
    _event_skip_reason,
    _lagged_volatility_leverage,
    _liquidity_cap,
    _position_rows,
    _select_side,
    _shares_for_trade,
    _skipped_trade_row,
    _symbol_cap_reached,
    _validate_config,
    apply_risk_controls,
    long_short_daily_returns,
)
from signalforge.exceptions import BacktestError


# ── Config validation ──────────────────────────────────────────────

class TestValidateConfig:
    def test_invalid_long_fraction_zero(self):
        with pytest.raises(BacktestError, match="long_fraction"):
            _validate_config(BacktestConfig(long_fraction=0))

    def test_invalid_long_fraction_above_one(self):
        with pytest.raises(BacktestError, match="long_fraction"):
            _validate_config(BacktestConfig(long_fraction=1.5))

    def test_invalid_short_fraction_zero(self):
        with pytest.raises(BacktestError, match="short_fraction"):
            _validate_config(BacktestConfig(short_fraction=0))

    def test_invalid_max_position_weight_zero(self):
        with pytest.raises(BacktestError, match="max_position_weight"):
            _validate_config(BacktestConfig(max_position_weight=0))

    def test_negative_costs(self):
        with pytest.raises(BacktestError, match="costs"):
            _validate_config(BacktestConfig(transaction_cost_bps=-1))

    def test_negative_slippage(self):
        with pytest.raises(BacktestError, match="costs"):
            _validate_config(BacktestConfig(slippage_bps=-1))

    def test_target_volatility_zero(self):
        with pytest.raises(BacktestError, match="target_volatility"):
            _validate_config(BacktestConfig(target_volatility=0))

    def test_volatility_lookback_too_small(self):
        with pytest.raises(BacktestError, match="volatility_lookback"):
            _validate_config(BacktestConfig(volatility_lookback=1))

    def test_max_leverage_zero(self):
        with pytest.raises(BacktestError, match="max_leverage"):
            _validate_config(BacktestConfig(max_leverage=0))

    def test_max_drawdown_stop_too_large(self):
        with pytest.raises(BacktestError, match="max_drawdown_stop"):
            _validate_config(BacktestConfig(max_drawdown_stop=1.0))

    def test_max_drawdown_stop_zero(self):
        with pytest.raises(BacktestError, match="max_drawdown_stop"):
            _validate_config(BacktestConfig(max_drawdown_stop=0.0))

    def test_negative_cooldown_days(self):
        with pytest.raises(BacktestError, match="cooldown_days"):
            _validate_config(BacktestConfig(cooldown_days=-1))

    def test_max_symbol_trades_zero(self):
        with pytest.raises(BacktestError, match="max_symbol_trades must be positive"):
            _validate_config(BacktestConfig(max_symbol_trades=0))

    def test_initial_capital_zero(self):
        with pytest.raises(BacktestError, match="initial_capital must be positive"):
            _validate_config(BacktestConfig(initial_capital=0))

    def test_negative_min_trade_dollars(self):
        with pytest.raises(BacktestError, match="min_trade_dollars"):
            _validate_config(BacktestConfig(min_trade_dollars=-1))

    def test_min_score_non_finite(self):
        with pytest.raises(BacktestError, match="min_score must be finite"):
            _validate_config(BacktestConfig(min_score=np.nan))

    def test_rebalance_interval_zero(self):
        with pytest.raises(BacktestError, match="rebalance_interval_days"):
            _validate_config(BacktestConfig(rebalance_interval_days=0))

    def test_max_adv_fraction_zero(self):
        with pytest.raises(BacktestError, match="max_adv_fraction must be positive"):
            _validate_config(BacktestConfig(max_adv_fraction=0))

    def test_valid_config_passes(self):
        cfg = BacktestConfig(
            long_fraction=0.1,
            short_fraction=0.1,
            max_position_weight=0.02,
        )
        _validate_config(cfg)


# ── Shares for trade ───────────────────────────────────────────────

class TestSharesForTrade:
    def test_whole_shares(self):
        assert _shares_for_trade(target_dollars=1000, price=150, allow_fractional=False) == 6.0

    def test_fractional_shares(self):
        result = _shares_for_trade(target_dollars=1000, price=150, allow_fractional=True)
        assert result == pytest.approx(6.6667, rel=1e-3)

    def test_zero_price(self):
        assert _shares_for_trade(target_dollars=1000, price=0, allow_fractional=True) == 0.0

    def test_negative_price(self):
        assert _shares_for_trade(target_dollars=1000, price=-10, allow_fractional=True) == 0.0


# ── Select side ────────────────────────────────────────────────────

class TestSelectSide:
    def test_selects_top_scorers(self):
        day = pd.DataFrame({
            "symbol": ["A", "B", "C"],
            "score": [0.1, 0.5, 0.3],
            "forward_return": [0.01, 0.02, 0.015],
        })
        cfg = BacktestConfig(max_symbol_trades=None, min_score=None)
        result = _select_side(
            day.sort_values("score", ascending=False),
            desired_count=2,
            symbol_trade_counts={},
            symbol_col="symbol",
            config=cfg,
        )
        assert list(result["symbol"]) == ["B", "C"]

    def test_respects_trade_cap(self):
        day = pd.DataFrame({
            "symbol": ["A", "B", "C"],
            "score": [0.9, 0.5, 0.3],
            "forward_return": [0.01, 0.02, 0.015],
        })
        cfg = BacktestConfig(max_symbol_trades=1, min_score=None)
        counts: dict[str, int] = {"A": 1}
        result = _select_side(
            day.sort_values("score", ascending=False),
            desired_count=2,
            symbol_trade_counts=counts,
            symbol_col="symbol",
            config=cfg,
        )
        assert "A" not in list(result["symbol"])
        assert list(result["symbol"]) == ["B", "C"]

    def test_returns_empty_when_no_candidates(self):
        day = pd.DataFrame({
            "symbol": ["A"],
            "score": [0.9],
            "forward_return": [0.01],
        })
        cfg = BacktestConfig(max_symbol_trades=0, min_score=None)
        result = _select_side(
            day.sort_values("score", ascending=False),
            desired_count=1,
            symbol_trade_counts={"A": 0},
            symbol_col="symbol",
            config=cfg,
        )
        assert result.empty


# ── Symbol cap reached ─────────────────────────────────────────────

class TestSymbolCapReached:
    def test_no_cap_when_none(self):
        assert not _symbol_cap_reached("A", {}, config=BacktestConfig(max_symbol_trades=None))

    def test_cap_not_reached(self):
        assert not _symbol_cap_reached("A", {"A": 1}, config=BacktestConfig(max_symbol_trades=2))

    def test_cap_reached(self):
        assert _symbol_cap_reached("A", {"A": 2}, config=BacktestConfig(max_symbol_trades=2))

    def test_symbol_not_in_counts(self):
        assert not _symbol_cap_reached("B", {"A": 5}, config=BacktestConfig(max_symbol_trades=3))


# ── Position rows ──────────────────────────────────────────────────

class TestPositionRows:
    def test_output_columns(self):
        day = pd.DataFrame({
            "symbol": ["A"],
            "forward_return": [0.02],
        })
        rows = _position_rows(
            day, symbol_col="symbol", side="long", weight=0.5, return_col="forward_return",
        )
        row = rows[0]
        assert row["symbol"] == "A"
        assert row["side"] == "long"
        assert row["weight"] == 0.5
        assert row["realized_return"] == 0.02
        assert row["contribution"] == 0.01


# ── Lagged volatility leverage ────────────────────────────────────

class TestLaggedVolatilityLeverage:
    def test_no_target_returns_ones(self):
        returns = pd.Series([0.01, -0.02, 0.03])
        result = _lagged_volatility_leverage(returns, config=BacktestConfig())
        assert (result == 1.0).all()

    def test_scales_by_inverse_volatility(self):
        returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.015])
        cfg = BacktestConfig(target_volatility=0.20, volatility_lookback=3, max_leverage=2.0)
        result = _lagged_volatility_leverage(returns, config=cfg)
        assert len(result) == 5
        assert result.iloc[0] == 1.0  # first row has no prior window
        assert result.iloc[1] == 1.0
        assert result.iloc[2] == 1.0
        assert result.iloc[3] > 0
        assert result.iloc[4] > 0
        assert result.iloc[3] <= cfg.max_leverage

    def test_clips_at_max_leverage(self):
        returns = pd.Series([0.0] * 10)
        returns.iloc[-1] = 0.001
        cfg = BacktestConfig(target_volatility=0.50, volatility_lookback=5, max_leverage=1.0)
        result = _lagged_volatility_leverage(returns, config=cfg)
        assert result.iloc[-1] <= 1.0

    def test_handles_nan_in_returns(self):
        returns = pd.Series([0.01, np.nan, 0.02, -0.01, 0.03])
        cfg = BacktestConfig(target_volatility=0.20, volatility_lookback=3, max_leverage=2.0)
        result = _lagged_volatility_leverage(returns, config=cfg)
        assert not result.isna().any()


# ── Liquidity cap ──────────────────────────────────────────────────

class TestLiquidityCap:
    def test_none_when_disabled(self):
        cfg = BacktestConfig(max_adv_fraction=None)
        assert _liquidity_cap(1_000_000, config=cfg) is None

    def test_returns_fraction_of_adv(self):
        cfg = BacktestConfig(max_adv_fraction=0.01)
        assert _liquidity_cap(1_000_000, config=cfg) == 10_000.0

    def test_zero_adv(self):
        cfg = BacktestConfig(max_adv_fraction=0.01)
        assert _liquidity_cap(0, config=cfg) == 0.0


# ── Event skip reason ──────────────────────────────────────────────

class TestEventSkipReason:
    def test_score_below_threshold(self):
        row = pd.Series({"score": -0.01, "symbol": "A", "next_open": 100, "exit_close": 110, "avg_dollar_volume_20d": 1e6})
        reason = _event_skip_reason(row, symbol_col="symbol", score_col="score", entry_price_col="next_open", exit_price_col="exit_close", avg_dollar_volume_col="avg_dollar_volume_20d", symbol_trade_counts={}, capital=2000, invested=0, config=BacktestConfig(min_score=0.0))
        assert reason == "score_below_threshold"

    def test_symbol_trade_cap(self):
        row = pd.Series({"score": 0.5, "symbol": "A", "next_open": 100, "exit_close": 110, "avg_dollar_volume_20d": 1e6})
        reason = _event_skip_reason(row, symbol_col="symbol", score_col="score", entry_price_col="next_open", exit_price_col="exit_close", avg_dollar_volume_col="avg_dollar_volume_20d", symbol_trade_counts={"A": 2}, capital=2000, invested=0, config=BacktestConfig(max_symbol_trades=1))
        assert reason == "symbol_trade_cap"

    def test_missing_entry_price(self):
        row = pd.Series({"score": 0.5, "symbol": "A", "next_open": np.nan, "exit_close": 110, "avg_dollar_volume_20d": 1e6})
        reason = _event_skip_reason(row, symbol_col="symbol", score_col="score", entry_price_col="next_open", exit_price_col="exit_close", avg_dollar_volume_col="avg_dollar_volume_20d", symbol_trade_counts={}, capital=2000, invested=0, config=BacktestConfig())
        assert reason == "missing_entry_price"

    def test_missing_exit_price(self):
        row = pd.Series({"score": 0.5, "symbol": "A", "next_open": 100, "exit_close": 0, "avg_dollar_volume_20d": 1e6})
        reason = _event_skip_reason(row, symbol_col="symbol", score_col="score", entry_price_col="next_open", exit_price_col="exit_close", avg_dollar_volume_col="avg_dollar_volume_20d", symbol_trade_counts={}, capital=2000, invested=0, config=BacktestConfig())
        assert reason == "missing_exit_price"

    def test_missing_liquidity(self):
        row = pd.Series({"score": 0.5, "symbol": "A", "next_open": 100, "exit_close": 110, "avg_dollar_volume_20d": 0})
        reason = _event_skip_reason(row, symbol_col="symbol", score_col="score", entry_price_col="next_open", exit_price_col="exit_close", avg_dollar_volume_col="avg_dollar_volume_20d", symbol_trade_counts={}, capital=2000, invested=0, config=BacktestConfig())
        assert reason == "missing_liquidity"

    def test_insufficient_cash(self):
        row = pd.Series({"score": 0.5, "symbol": "A", "next_open": 100_000, "exit_close": 110, "avg_dollar_volume_20d": 1e6})
        reason = _event_skip_reason(row, symbol_col="symbol", score_col="score", entry_price_col="next_open", exit_price_col="exit_close", avg_dollar_volume_col="avg_dollar_volume_20d", symbol_trade_counts={}, capital=100, invested=99, config=BacktestConfig(min_trade_dollars=10))
        assert reason == "insufficient_cash"

    def test_no_skip_reason(self):
        row = pd.Series({"score": 0.5, "symbol": "A", "next_open": 100, "exit_close": 110, "avg_dollar_volume_20d": 1e6})
        reason = _event_skip_reason(row, symbol_col="symbol", score_col="score", entry_price_col="next_open", exit_price_col="exit_close", avg_dollar_volume_col="avg_dollar_volume_20d", symbol_trade_counts={}, capital=2000, invested=0, config=BacktestConfig())
        assert reason is None


# ── Skipped trade row ──────────────────────────────────────────────

class TestSkippedTradeRow:
    def test_output_format(self):
        row = pd.Series({"symbol": "A", "score": 0.5})
        result = _skipped_trade_row(row, pd.Timestamp("2024-01-01"), "symbol", "insufficient_cash")
        assert result["status"] == "skipped"
        assert result["symbol"] == "A"
        assert result["skip_reason"] == "insufficient_cash"
        assert result["shares"] == 0.0
        assert result["entry_value"] == 0.0

    def test_handles_missing_score(self):
        row = pd.Series({"symbol": "A"})
        result = _skipped_trade_row(row, pd.Timestamp("2024-01-01"), "symbol", "test")
        assert np.isnan(result["score"])


# ── Event equity row ───────────────────────────────────────────────

class TestEventEquityRow:
    def test_output_format(self):
        result = _event_equity_row(pd.Timestamp("2024-01-01"), 5000, 2, 1000, 50)
        assert result["date"] == pd.Timestamp("2024-01-01")
        assert result["capital"] == 5000
        assert result["positions"] == 2
        assert result["invested"] == 1000
        assert result["cash"] == 4000
        assert result["net_pnl"] == 50


# ── BacktestConfig defaults / frozen ───────────────────────────────

class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.long_fraction == 0.1
        assert cfg.short_fraction == 0.1
        assert cfg.max_position_weight == 0.02
        assert cfg.rebalance_interval_days == 1
        assert cfg.max_adv_fraction == 0.01
        assert cfg.allow_fractional_shares is False

    def test_is_frozen(self):
        cfg = BacktestConfig()
        with pytest.raises(AttributeError):
            cfg.long_fraction = 0.2  # type: ignore[misc]


# ── long_short_daily_returns edge cases ────────────────────────────

class TestLongShortDailyReturns:
    def test_missing_required_column(self):
        with pytest.raises(BacktestError, match="scored_returns is missing"):
            long_short_daily_returns(pd.DataFrame({"date": [1]}))

    def test_empty_after_dropna(self):
        scored = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["A"],
            "score": [np.nan],
            "forward_return": [np.nan],
        })
        with pytest.raises(BacktestError):
            long_short_daily_returns(scored)

    def test_single_symbol_daily(self):
        scored = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["A", "A"],
            "score": [0.5, 0.5],
            "forward_return": [0.02, -0.01],
        })
        result = long_short_daily_returns(scored)
        assert len(result) == 2
        assert "risk_equity" in result.columns

    def test_with_vol_target_and_drawdown_stop(self):
        scored = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"] * 4),
            "symbol": ["A", "B", "C", "D"],
            "score": [0.9, 0.5, -0.3, -0.7],
            "forward_return": [0.10, 0.02, -0.01, -0.05],
        })
        result = long_short_daily_returns(
            scored,
            config=BacktestConfig(
                long_fraction=0.25,
                short_fraction=0.25,
                max_position_weight=1.0,
                target_volatility=0.15,
                volatility_lookback=5,
                max_drawdown_stop=0.15,
                cooldown_days=3,
            ),
        )
        assert {"leverage", "risk_net_return", "risk_trading_enabled", "risk_equity"}.issubset(
            result.columns
        )


# ── apply_risk_controls edge cases ────────────────────────────────

class TestApplyRiskControls:
    def test_missing_return_column(self):
        with pytest.raises(BacktestError, match="daily_returns is missing"):
            apply_risk_controls(pd.DataFrame({"foo": [1]}))

    def test_no_vol_target_no_drawdown_stop(self):
        daily = pd.DataFrame({"net_return": [0.01, -0.01, 0.02]})
        result = apply_risk_controls(daily)
        assert (result["risk_net_return"] == daily["net_return"]).all()
        assert (result["risk_trading_enabled"] == [True, True, True]).all()

    def test_cooldown_activates_and_counts_down(self):
        daily = pd.DataFrame({"net_return": [0.01, -0.50, 0.02, 0.03, 0.01]})
        result = apply_risk_controls(
            daily,
            config=BacktestConfig(
                target_volatility=None,
                max_drawdown_stop=0.20,
                cooldown_days=2,
            ),
        )
        assert bool(result.loc[0, "risk_trading_enabled"]) is True
        assert bool(result.loc[1, "risk_trading_enabled"]) is True
        assert bool(result.loc[2, "risk_trading_enabled"]) is False
        assert bool(result.loc[3, "risk_trading_enabled"]) is False
        assert result["risk_net_return"].iloc[2] == 0.0
        assert result["risk_net_return"].iloc[3] == 0.0

    def test_equity_cumprod(self):
        daily = pd.DataFrame({"net_return": [0.01, 0.02, -0.01]})
        result = apply_risk_controls(daily)
        expected = (1.0 + daily["net_return"]).cumprod()
        pd.testing.assert_series_equal(
            result["risk_equity"], expected, check_names=False,
        )
