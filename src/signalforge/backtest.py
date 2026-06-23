from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from signalforge.exceptions import BacktestError
from signalforge.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    long_fraction: float = 0.1
    short_fraction: float = 0.1
    max_position_weight: float = 0.02
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    target_volatility: float | None = None
    volatility_lookback: int = 20
    max_leverage: float = 1.0
    max_drawdown_stop: float | None = None
    cooldown_days: int = 20
    max_symbol_trades: int | None = None
    initial_capital: float | None = None
    allow_fractional_shares: bool = False
    min_trade_dollars: float = 1.0
    min_score: float | None = None
    rebalance_interval_days: int = 1
    max_adv_fraction: float | None = 0.01


def long_short_daily_returns(
    scored_returns: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    score_col: str = "score",
    return_col: str = "forward_return",
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Simulate a daily cross-sectional long/short portfolio from scores.

    Each date goes long the top score bucket and short the bottom score bucket,
    equal-weighted inside each side and capped by max position weight.
    """
    cfg = config or BacktestConfig()
    _validate_config(cfg)
    required = {date_col, symbol_col, score_col, return_col}
    missing = required.difference(scored_returns.columns)
    if missing:
        raise BacktestError(f"scored_returns is missing required columns: {sorted(missing)}")

    rows = []
    symbol_trade_counts: dict[str, int] = {}
    day_count = 0
    for date, day in scored_returns.dropna(subset=[score_col, return_col]).groupby(date_col):
        positions = build_daily_positions(
            day,
            symbol_trade_counts=symbol_trade_counts,
            symbol_col=symbol_col,
            score_col=score_col,
            return_col=return_col,
            config=cfg,
        )

        gross_return = positions["contribution"].sum()
        gross_exposure = positions["weight"].abs().sum()
        cost = gross_exposure * (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000.0

        rows.append(
            {
                date_col: date,
                "gross_return": gross_return,
                "cost": cost,
                "net_return": gross_return - cost,
                "long_count": int((positions["side"] == "long").sum()),
                "short_count": int((positions["side"] == "short").sum()),
                "gross_exposure": gross_exposure,
            }
        )
        day_count += 1

    if not rows:
        raise BacktestError("no valid rows after dropping missing scores and returns")

    logger.info(
        "long_short backtest complete: %d days, %d rows processed",
        day_count, len(scored_returns),
    )
    returns = pd.DataFrame(rows).sort_values(date_col).reset_index(drop=True)
    return apply_risk_controls(returns, config=cfg)


def build_daily_positions(
    day: pd.DataFrame,
    *,
    symbol_trade_counts: dict[str, int],
    symbol_col: str = "symbol",
    score_col: str = "score",
    return_col: str = "forward_return",
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Select long/short daily positions while respecting symbol trade caps."""
    cfg = config or BacktestConfig()
    count = len(day)
    long_count = max(1, int(count * cfg.long_fraction))
    short_count = max(1, int(count * cfg.short_fraction))
    long_weight = min(1.0 / long_count, cfg.max_position_weight)
    short_weight = min(1.0 / short_count, cfg.max_position_weight)

    ranked = day.sort_values(score_col)
    shorts = _select_side(
        ranked,
        desired_count=short_count,
        symbol_trade_counts=symbol_trade_counts,
        symbol_col=symbol_col,
        config=cfg,
    )
    longs = _select_side(
        ranked.sort_values(score_col, ascending=False),
        desired_count=long_count,
        symbol_trade_counts=symbol_trade_counts,
        symbol_col=symbol_col,
        config=cfg,
    )

    rows = []
    rows.extend(
        _position_rows(
            longs,
            symbol_col=symbol_col,
            side="long",
            weight=long_weight,
            return_col=return_col,
        )
    )
    rows.extend(
        _position_rows(
            shorts,
            symbol_col=symbol_col,
            side="short",
            weight=-short_weight,
            return_col=return_col,
        )
    )
    return pd.DataFrame(
        rows,
        columns=["symbol", "side", "weight", "realized_return", "contribution"],
    )


def apply_risk_controls(
    daily_returns: pd.DataFrame,
    *,
    return_col: str = "net_return",
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Apply lagged volatility targeting and drawdown cooldown controls."""
    cfg = config or BacktestConfig()
    if return_col not in daily_returns.columns:
        raise BacktestError(f"daily_returns is missing required column: {return_col}")

    controlled = daily_returns.copy()
    controlled["leverage"] = _lagged_volatility_leverage(controlled[return_col], config=cfg)

    risk_returns = []
    trading_enabled = []
    cooldown_remaining = 0
    equity = 1.0
    peak = 1.0
    cooldown_events = 0

    for raw_return, leverage in zip(controlled[return_col], controlled["leverage"], strict=True):
        enabled = cooldown_remaining <= 0
        adjusted_return = raw_return * leverage if enabled else 0.0
        equity *= 1.0 + adjusted_return
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0

        if enabled and cfg.max_drawdown_stop is not None and drawdown <= -cfg.max_drawdown_stop:
            cooldown_remaining = cfg.cooldown_days
            cooldown_events += 1
        elif cooldown_remaining > 0:
            cooldown_remaining -= 1

        trading_enabled.append(enabled)
        risk_returns.append(adjusted_return)

    if cooldown_events:
        logger.warning("drawdown stop triggered %d time(s) during backtest", cooldown_events)

    controlled["risk_net_return"] = risk_returns
    controlled["risk_trading_enabled"] = trading_enabled
    controlled["risk_equity"] = (1.0 + controlled["risk_net_return"]).cumprod()
    return controlled


def _validate_config(config: BacktestConfig) -> None:
    for name in ("long_fraction", "short_fraction", "max_position_weight"):
        value = getattr(config, name)
        if value <= 0 or value > 1:
            raise BacktestError(f"{name} must be in (0, 1]")
    if config.transaction_cost_bps < 0 or config.slippage_bps < 0:
        raise BacktestError("costs must be non-negative")
    if config.target_volatility is not None and config.target_volatility <= 0:
        raise BacktestError("target_volatility must be positive")
    if config.volatility_lookback <= 1:
        raise BacktestError("volatility_lookback must be greater than 1")
    if config.max_leverage <= 0:
        raise BacktestError("max_leverage must be positive")
    if config.max_drawdown_stop is not None and not 0 < config.max_drawdown_stop < 1:
        raise BacktestError("max_drawdown_stop must be in (0, 1)")
    if config.cooldown_days < 0:
        raise BacktestError("cooldown_days must be non-negative")
    if config.max_symbol_trades is not None and config.max_symbol_trades <= 0:
        raise BacktestError("max_symbol_trades must be positive")
    if config.initial_capital is not None and config.initial_capital <= 0:
        raise BacktestError("initial_capital must be positive")
    if config.min_trade_dollars < 0:
        raise BacktestError("min_trade_dollars must be non-negative")
    if config.min_score is not None and not np.isfinite(config.min_score):
        raise BacktestError("min_score must be finite")
    if config.rebalance_interval_days <= 0:
        raise BacktestError("rebalance_interval_days must be positive")
    if config.max_adv_fraction is not None and config.max_adv_fraction <= 0:
        raise BacktestError("max_adv_fraction must be positive")


def _lagged_volatility_leverage(returns: pd.Series, *, config: BacktestConfig) -> pd.Series:
    if config.target_volatility is None:
        return pd.Series(1.0, index=returns.index)

    annualized_volatility = returns.shift(1).rolling(
        config.volatility_lookback,
        min_periods=config.volatility_lookback,
    ).std() * np.sqrt(252)
    leverage = config.target_volatility / annualized_volatility
    leverage = leverage.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return leverage.clip(lower=0.0, upper=config.max_leverage)


def _select_side(
    ranked: pd.DataFrame,
    *,
    desired_count: int,
    symbol_trade_counts: dict[str, int],
    symbol_col: str,
    config: BacktestConfig,
) -> pd.DataFrame:
    selected_rows = []
    skipped_count = 0
    for _, row in ranked.iterrows():
        symbol = row[symbol_col]
        if _symbol_cap_reached(symbol, symbol_trade_counts, config=config):
            skipped_count += 1
            continue
        selected_rows.append(row)
        symbol_trade_counts[symbol] = symbol_trade_counts.get(symbol, 0) + 1
        if len(selected_rows) == desired_count:
            break
    if skipped_count:
        logger.debug("skipped %d symbols due to trade cap", skipped_count)
    if not selected_rows:
        return ranked.head(0)
    return pd.DataFrame(selected_rows)


def _symbol_cap_reached(
    symbol: str,
    symbol_trade_counts: dict[str, int],
    *,
    config: BacktestConfig,
) -> bool:
    return (
        config.max_symbol_trades is not None
        and symbol_trade_counts.get(symbol, 0) >= config.max_symbol_trades
    )


def _position_rows(
    frame: pd.DataFrame,
    *,
    symbol_col: str,
    side: str,
    weight: float,
    return_col: str,
) -> list[dict]:
    rows = []
    for _, row in frame.iterrows():
        contribution = weight * row[return_col]
        rows.append(
            {
                "symbol": row[symbol_col],
                "side": side,
                "weight": weight,
                "realized_return": row[return_col],
                "contribution": contribution,
            }
        )
    return rows


def long_only_capital_backtest(
    scored_returns: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    score_col: str = "score",
    return_col: str = "forward_return",
    price_col: str = "adj_close",
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Approximate a whole-share long-only backtest for small accounts.

    This is still research-only: it uses forward returns for mark-to-market outcomes,
    but applies whole-share sizing and cash drag from a fixed starting capital.
    """
    cfg = config or BacktestConfig(initial_capital=2_000)
    _validate_config(cfg)
    capital = cfg.initial_capital or 2_000.0
    required = {date_col, symbol_col, score_col, return_col, price_col}
    missing = required.difference(scored_returns.columns)
    if missing:
        raise BacktestError(f"scored_returns is missing required columns: {sorted(missing)}")

    rows = []
    symbol_trade_counts: dict[str, int] = {}
    grouped_days = scored_returns.dropna(subset=[score_col, return_col, price_col]).groupby(
        date_col,
        sort=True,
    )
    for day_index, (date, day) in enumerate(grouped_days):
        if day_index % cfg.rebalance_interval_days != 0:
            rows.append(
                {
                    date_col: date,
                    "capital": capital,
                    "invested": 0.0,
                    "cash": capital,
                    "positions": 0,
                    "gross_pnl": 0.0,
                    "cost": 0.0,
                    "net_pnl": 0.0,
                    "net_return": 0.0,
                    "gross_exposure": 0.0,
                }
            )
            continue

        candidates = _select_side(
            day.sort_values(score_col, ascending=False),
            desired_count=max(1, int(len(day) * cfg.long_fraction)),
            symbol_trade_counts=symbol_trade_counts,
            symbol_col=symbol_col,
            config=cfg,
        )
        target_dollars = min(capital * cfg.max_position_weight, capital)
        invested = 0.0
        gross_pnl = 0.0
        positions = 0
        for _, row in candidates.iterrows():
            shares = _shares_for_trade(
                target_dollars=target_dollars,
                price=row[price_col],
                allow_fractional=cfg.allow_fractional_shares,
            )
            trade_dollars = shares * row[price_col]
            exceeds_cash = invested + trade_dollars > capital
            below_minimum = trade_dollars < cfg.min_trade_dollars
            if shares <= 0 or below_minimum or exceeds_cash:
                continue
            invested += trade_dollars
            gross_pnl += trade_dollars * row[return_col]
            positions += 1

        cost = invested * (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000.0
        net_pnl = gross_pnl - cost
        net_return = net_pnl / capital if capital else 0.0
        capital += net_pnl
        rows.append(
            {
                date_col: date,
                "capital": capital,
                "invested": invested,
                "cash": capital - invested,
                "positions": positions,
                "gross_pnl": gross_pnl,
                "cost": cost,
                "net_pnl": net_pnl,
                "net_return": net_return,
                "gross_exposure": invested / capital if capital else 0.0,
            }
        )

    returns = pd.DataFrame(rows).sort_values(date_col).reset_index(drop=True)
    logger.info("long_only capital backtest complete: final capital %.2f", capital)
    return apply_risk_controls(returns, config=cfg)


def _shares_for_trade(
    *,
    target_dollars: float,
    price: float,
    allow_fractional: bool,
) -> float:
    if price <= 0:
        return 0.0
    shares = target_dollars / price
    return shares if allow_fractional else float(np.floor(shares))


def event_based_long_only_backtest(
    signals: pd.DataFrame,
    *,
    signal_date_col: str = "date",
    symbol_col: str = "symbol",
    score_col: str = "score",
    entry_price_col: str = "next_open",
    exit_price_col: str = "exit_close",
    avg_dollar_volume_col: str = "avg_dollar_volume_20d",
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run an event-based long-only backtest with explicit trade ledger.

    Signals are assumed to be known after the signal date close. Entry prices
    should therefore be next-session executable prices, and exit prices should
    be from the configured holding-window exit date.
    """
    cfg = config or BacktestConfig(initial_capital=2_000)
    _validate_config(cfg)
    initial_capital = cfg.initial_capital or 2_000.0
    required = {
        signal_date_col,
        symbol_col,
        score_col,
        entry_price_col,
        exit_price_col,
        avg_dollar_volume_col,
    }
    missing = required.difference(signals.columns)
    if missing:
        raise BacktestError(f"signals is missing required columns: {sorted(missing)}")

    frame = signals.copy()
    frame[signal_date_col] = pd.to_datetime(frame[signal_date_col])
    frame = frame.sort_values([signal_date_col, score_col], ascending=[True, False])

    capital = initial_capital
    ledger_rows = []
    equity_rows = []
    symbol_trade_counts: dict[str, int] = {}
    peak_capital = capital
    cooldown_remaining = 0
    skipped_reasons: dict[str, int] = {}
    for day_index, (signal_date, day) in enumerate(frame.groupby(signal_date_col, sort=True)):
        trading_enabled = cooldown_remaining <= 0
        if not trading_enabled:
            cooldown_remaining -= 1
        if day_index % cfg.rebalance_interval_days != 0 or not trading_enabled:
            equity_rows.append(_event_equity_row(signal_date, capital, 0, 0.0, 0.0))
            continue

        candidates = day.sort_values(score_col, ascending=False)
        desired_count = max(1, int(len(candidates) * cfg.long_fraction))
        opened_positions = 0
        invested = 0.0
        day_net_pnl = 0.0

        for _, row in candidates.iterrows():
            if opened_positions >= desired_count:
                break
            skip_reason = _event_skip_reason(
                row,
                symbol_col=symbol_col,
                score_col=score_col,
                entry_price_col=entry_price_col,
                exit_price_col=exit_price_col,
                avg_dollar_volume_col=avg_dollar_volume_col,
                symbol_trade_counts=symbol_trade_counts,
                capital=capital,
                invested=invested,
                config=cfg,
            )
            if skip_reason is not None:
                skipped_reasons[skip_reason] = skipped_reasons.get(skip_reason, 0) + 1
                ledger_rows.append(_skipped_trade_row(row, signal_date, symbol_col, skip_reason))
                continue

            target_dollars = min(capital * cfg.max_position_weight, capital - invested)
            liquidity_cap = _liquidity_cap(row[avg_dollar_volume_col], config=cfg)
            if liquidity_cap is not None:
                target_dollars = min(target_dollars, liquidity_cap)
            shares = _shares_for_trade(
                target_dollars=target_dollars,
                price=row[entry_price_col],
                allow_fractional=cfg.allow_fractional_shares,
            )
            entry_value = shares * row[entry_price_col]
            if shares <= 0 or entry_value < cfg.min_trade_dollars:
                ledger_rows.append(
                    _skipped_trade_row(row, signal_date, symbol_col, "size_too_small")
                )
                continue

            exit_value = shares * row[exit_price_col]
            entry_cost = entry_value * (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000.0
            exit_cost = exit_value * (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000.0
            gross_pnl = exit_value - entry_value
            net_pnl = gross_pnl - entry_cost - exit_cost

            symbol = row[symbol_col]
            symbol_trade_counts[symbol] = symbol_trade_counts.get(symbol, 0) + 1
            opened_positions += 1
            invested += entry_value
            day_net_pnl += net_pnl
            ledger_rows.append(
                {
                    "status": "filled",
                    "signal_date": signal_date,
                    "symbol": symbol,
                    "score": row[score_col],
                    "shares": shares,
                    "entry_price": row[entry_price_col],
                    "exit_price": row[exit_price_col],
                    "entry_value": entry_value,
                    "exit_value": exit_value,
                    "entry_cost": entry_cost,
                    "exit_cost": exit_cost,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "return": net_pnl / entry_value if entry_value else 0.0,
                    "skip_reason": "",
                }
            )

        capital += day_net_pnl
        peak_capital = max(peak_capital, capital)
        drawdown = capital / peak_capital - 1.0
        if cfg.max_drawdown_stop is not None and drawdown <= -cfg.max_drawdown_stop:
            cooldown_remaining = cfg.cooldown_days
        equity_rows.append(
            _event_equity_row(signal_date, capital, opened_positions, invested, day_net_pnl)
        )

    if skipped_reasons:
        logger.info("skipped trades by reason: %s", skipped_reasons)
    logger.info(
        "event-based backtest complete: final capital %.2f, %d trades",
        capital, len([r for r in ledger_rows if r.get("status") == "filled"]),
    )

    equity = pd.DataFrame(equity_rows)
    if not equity.empty:
        equity["net_return"] = equity["capital"].pct_change().fillna(
            equity["capital"].div(initial_capital).sub(1.0)
        )
        equity["drawdown"] = equity["capital"].div(equity["capital"].cummax()).sub(1.0)
    ledger = pd.DataFrame(ledger_rows)
    return equity, ledger


def _event_skip_reason(
    row: pd.Series,
    *,
    symbol_col: str,
    score_col: str,
    entry_price_col: str,
    exit_price_col: str,
    avg_dollar_volume_col: str,
    symbol_trade_counts: dict[str, int],
    capital: float,
    invested: float,
    config: BacktestConfig,
) -> str | None:
    symbol = row[symbol_col]
    if config.min_score is not None and row.get(score_col, np.nan) < config.min_score:
        return "score_below_threshold"
    if _symbol_cap_reached(symbol, symbol_trade_counts, config=config):
        return "symbol_trade_cap"
    if pd.isna(row[entry_price_col]) or row[entry_price_col] <= 0:
        return "missing_entry_price"
    if pd.isna(row[exit_price_col]) or row[exit_price_col] <= 0:
        return "missing_exit_price"
    if pd.isna(row[avg_dollar_volume_col]) or row[avg_dollar_volume_col] <= 0:
        return "missing_liquidity"
    if capital - invested < config.min_trade_dollars:
        return "insufficient_cash"
    return None


def _liquidity_cap(avg_dollar_volume: float, *, config: BacktestConfig) -> float | None:
    if config.max_adv_fraction is None:
        return None
    return avg_dollar_volume * config.max_adv_fraction


def _skipped_trade_row(
    row: pd.Series,
    signal_date: pd.Timestamp,
    symbol_col: str,
    reason: str,
) -> dict:
    return {
        "status": "skipped",
        "signal_date": signal_date,
        "symbol": row[symbol_col],
        "score": row.get("score", np.nan),
        "shares": 0.0,
        "entry_price": np.nan,
        "exit_price": np.nan,
        "entry_value": 0.0,
        "exit_value": 0.0,
        "entry_cost": 0.0,
        "exit_cost": 0.0,
        "gross_pnl": 0.0,
        "net_pnl": 0.0,
        "return": 0.0,
        "skip_reason": reason,
    }


def _event_equity_row(
    signal_date: pd.Timestamp,
    capital: float,
    positions: int,
    invested: float,
    net_pnl: float,
) -> dict:
    return {
        "date": signal_date,
        "capital": capital,
        "positions": positions,
        "invested": invested,
        "cash": capital - invested,
        "net_pnl": net_pnl,
    }
