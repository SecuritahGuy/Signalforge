from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PAPER_LEDGER_COLUMNS = (
    "order_id",
    "status",
    "planned_date",
    "symbol",
    "sector",
    "score",
    "shares",
    "reference_price",
    "estimated_entry_value",
    "estimated_cost",
    "target_exit_date",
    "actual_exit_trigger_date",
    "fill_date",
    "entry_price",
    "entry_value",
    "entry_cost",
    "exit_date",
    "exit_price",
    "exit_value",
    "exit_cost",
    "gross_pnl",
    "net_pnl",
    "return",
    "exit_reason",
    "exit_signal_value",
    "exit_rule_version",
    "highest_close_since_entry",
    "trailing_stop_activated",
    "skip_reason",
)


EXIT_RULE_VERSION = "exit_rules.v2"


@dataclass(frozen=True)
class StopLossConfig:
    enabled: bool = False
    pct: float = -0.08


@dataclass(frozen=True)
class TrailingStopConfig:
    enabled: bool = False
    activate_at_return: float = 0.12
    trail_from_high_pct: float = -0.06


@dataclass(frozen=True)
class ScoreDeteriorationConfig:
    enabled: bool = False
    min_days_held: int = 5
    exit_below_score: float = 0.005
    exit_if_score_declines_pct: float = 0.60


@dataclass(frozen=True)
class RebalanceConfig:
    enabled: bool = False
    min_days_held: int = 10
    exit_below_score: float = 0.01


@dataclass(frozen=True)
class TrailingVolatilityStopConfig:
    enabled: bool = False
    activate_at_return: float = 0.12
    volatility_lookback: int = 20
    volatility_multiple: float = 2.0
    tightest_trail_pct: float = -0.03
    widest_trail_pct: float = -0.15


@dataclass(frozen=True)
class TimeDecayConfig:
    enabled: bool = False
    half_life_days: int = 10
    min_days_hold: int = 2
    min_score_for_decay: float = 0.005


@dataclass(frozen=True)
class SectorStopConfig:
    enabled: bool = False
    sector_decline_pct: float = -0.05
    lookback_days: int = 5
    min_sector_records: int = 3


@dataclass(frozen=True)
class ExitRulesConfig:
    horizon_days: int = 20
    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)
    trailing_stop: TrailingStopConfig = field(default_factory=TrailingStopConfig)
    trailing_volatility_stop: TrailingVolatilityStopConfig = field(
        default_factory=TrailingVolatilityStopConfig
    )
    time_decay: TimeDecayConfig = field(default_factory=TimeDecayConfig)
    sector_stop: SectorStopConfig = field(default_factory=SectorStopConfig)
    score_deterioration: ScoreDeteriorationConfig = field(
        default_factory=ScoreDeteriorationConfig
    )
    rebalance: RebalanceConfig = field(default_factory=RebalanceConfig)


@dataclass(frozen=True)
class PaperTradingConfig:
    initial_capital: float = 2_000.0
    position_weight: float = 0.10
    long_fraction: float = 0.10
    min_score: float = 0.01
    horizon: int = 20
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    allow_fractional_shares: bool = False
    exit_rules: ExitRulesConfig = field(default_factory=ExitRulesConfig)


def build_planned_orders(
    scored: pd.DataFrame,
    *,
    config: PaperTradingConfig | None = None,
    available_cash: float | None = None,
    excluded_symbols: set[str] | None = None,
) -> pd.DataFrame:
    """Create paper orders from scored candidates without assuming execution."""
    cfg = config or PaperTradingConfig()
    _validate_config(cfg)
    required = {"date", "symbol", "score", "adj_close"}
    missing = required.difference(scored.columns)
    if missing:
        raise KeyError(f"scored is missing required columns: {sorted(missing)}")

    ranked = scored.sort_values("score", ascending=False).reset_index(drop=True).copy()
    planned_date = pd.Timestamp(ranked["date"].iloc[0])
    target_exit_date = _target_exit_date(planned_date, _configured_horizon_days(cfg))
    desired_count = max(1, int(len(ranked) * cfg.long_fraction))
    target_dollars = cfg.initial_capital * cfg.position_weight
    remaining_cash = cfg.initial_capital if available_cash is None else available_cash
    excluded = {symbol.upper() for symbol in (excluded_symbols or set())}
    planned_count = 0
    rows = []

    for _, row in ranked.iterrows():
        if row["symbol"].upper() in excluded:
            rows.append(
                _base_order_row(row, cfg, target_exit_date, "skipped", "symbol_already_open")
            )
            continue
        if row["score"] < cfg.min_score:
            rows.append(
                _base_order_row(row, cfg, target_exit_date, "skipped", "score_below_threshold")
            )
            continue
        if planned_count >= desired_count:
            rows.append(
                _base_order_row(
                    row,
                    cfg,
                    target_exit_date,
                    "skipped",
                    "rank_below_position_count",
                )
            )
            continue

        reference_price = row["adj_close"]
        shares = _shares_for_target(
            target_dollars=target_dollars,
            price=reference_price,
            allow_fractional=cfg.allow_fractional_shares,
        )
        estimated_entry_value = shares * reference_price
        estimated_cost = _trade_cost(
            estimated_entry_value,
            transaction_cost_bps=cfg.transaction_cost_bps,
            slippage_bps=cfg.slippage_bps,
        )
        required_cash = estimated_entry_value + estimated_cost
        if shares <= 0 or required_cash <= 0:
            rows.append(_base_order_row(row, cfg, target_exit_date, "skipped", "size_too_small"))
            continue
        if required_cash > remaining_cash:
            rows.append(_base_order_row(row, cfg, target_exit_date, "skipped", "insufficient_cash"))
            continue

        planned_count += 1
        remaining_cash -= required_cash
        order = _base_order_row(row, cfg, target_exit_date, "planned", "")
        order.update(
            {
                "shares": shares,
                "estimated_entry_value": estimated_entry_value,
                "estimated_cost": estimated_cost,
            }
        )
        rows.append(order)

    ledger = pd.DataFrame(rows).loc[:, PAPER_LEDGER_COLUMNS]
    ledger["order_id"] = _order_ids(ledger)
    return ledger


def reconcile_fills(
    ledger: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    config: PaperTradingConfig | None = None,
) -> pd.DataFrame:
    """Move planned orders to open when next-session open prices are available."""
    cfg = config or PaperTradingConfig()
    _validate_config(cfg)
    output = _normalize_ledger(ledger)
    output = _dedupe_active_orders(output)
    price_lookup = _price_lookup(prices)
    available_cash = _current_cash(output, initial_capital=cfg.initial_capital)
    active_symbols = set(output.loc[output["status"] == "open", "symbol"].str.upper())

    planned = output.loc[output["status"] == "planned"].sort_values(
        ["planned_date", "order_id"]
    )
    for index, row in planned.iterrows():
        symbol = row["symbol"].upper()
        if symbol in active_symbols:
            _skip_order_in_place(output, index, reason="duplicate_active_symbol")
            continue
        fill_date = _next_available_date(
            price_lookup,
            row["symbol"],
            pd.Timestamp(row["planned_date"]),
        )
        if fill_date is None:
            continue
        price_row = price_lookup[(row["symbol"], fill_date)]
        entry_price = _adjusted_open(price_row)
        entry_value = row["shares"] * entry_price
        entry_cost = _trade_cost(
            entry_value,
            transaction_cost_bps=cfg.transaction_cost_bps,
            slippage_bps=cfg.slippage_bps,
        )
        required_cash = entry_value + entry_cost
        if required_cash > available_cash:
            _skip_order_in_place(output, index, reason="insufficient_cash_at_fill")
            continue
        output.loc[index, ["status", "fill_date", "entry_price", "entry_value", "entry_cost"]] = [
            "open",
            fill_date,
            entry_price,
            entry_value,
            entry_cost,
        ]
        available_cash -= required_cash
        active_symbols.add(symbol)
    return output.loc[:, PAPER_LEDGER_COLUMNS]


def reconcile_exits(
    ledger: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    scores: pd.DataFrame | None = None,
    config: PaperTradingConfig | None = None,
    universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Close open paper positions when the first configured exit rule triggers."""
    cfg = config or PaperTradingConfig()
    _validate_config(cfg)
    output = _normalize_ledger(ledger)
    price_lookup = _price_lookup(prices)
    score_lookup = _score_lookup(scores)
    vol_lookup = (
        _volatility_lookup(
            prices,
            lookback=cfg.exit_rules.trailing_volatility_stop.volatility_lookback,
        )
        if cfg.exit_rules.trailing_volatility_stop.enabled
        else {}
    )
    sector_lookup = (
        _sector_lookup(
            prices,
            universe,
            lookback=cfg.exit_rules.sector_stop.lookback_days,
            min_records=cfg.exit_rules.sector_stop.min_sector_records,
        )
        if cfg.exit_rules.sector_stop.enabled and universe is not None
        else {}
    )

    for index, row in output.loc[output["status"] == "open"].iterrows():
        decision = _first_exit_decision(
            row,
            price_lookup,
            score_lookup,
            vol_lookup=vol_lookup,
            sector_lookup=sector_lookup,
            config=cfg,
        )
        if decision is None:
            state = _open_position_state(row, price_lookup, config=cfg)
            if state is not None:
                output.loc[
                    index,
                    ["highest_close_since_entry", "trailing_stop_activated"],
                ] = [
                    state["highest_close_since_entry"],
                    state["trailing_stop_activated"],
                ]
            continue
        exit_date = decision["exit_date"]
        exit_price = decision["exit_price"]
        exit_value = row["shares"] * exit_price
        exit_cost = _trade_cost(
            exit_value,
            transaction_cost_bps=cfg.transaction_cost_bps,
            slippage_bps=cfg.slippage_bps,
        )
        gross_pnl = exit_value - row["entry_value"]
        net_pnl = gross_pnl - row["entry_cost"] - exit_cost
        output.loc[
            index,
            [
                "status",
                "actual_exit_trigger_date",
                "exit_date",
                "exit_price",
                "exit_value",
                "exit_cost",
                "gross_pnl",
                "net_pnl",
                "return",
                "exit_reason",
                "exit_signal_value",
                "exit_rule_version",
                "highest_close_since_entry",
                "trailing_stop_activated",
            ],
        ] = [
            "closed",
            decision["actual_exit_trigger_date"],
            exit_date,
            exit_price,
            exit_value,
            exit_cost,
            gross_pnl,
            net_pnl,
            net_pnl / row["entry_value"] if row["entry_value"] else 0.0,
            decision["exit_reason"],
            decision["exit_signal_value"],
            EXIT_RULE_VERSION,
            decision["highest_close_since_entry"],
            decision["trailing_stop_activated"],
        ]
    return output.loc[:, PAPER_LEDGER_COLUMNS]


def summarize_paper_account(
    ledger: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    initial_capital: float = 2_000.0,
) -> dict:
    """Summarize current paper account state from persisted ledger and latest prices."""
    normalized = _normalize_ledger(ledger)
    price_lookup = _price_lookup(prices)
    filled = normalized.loc[normalized["status"].isin(["open", "closed"])]
    open_positions = normalized.loc[normalized["status"] == "open"]
    closed = normalized.loc[normalized["status"] == "closed"]

    realized_pnl = closed["net_pnl"].fillna(0.0).sum()
    committed_cash = (
        filled["entry_value"].fillna(0.0).sum() + filled["entry_cost"].fillna(0.0).sum()
    )
    unrealized_value = 0.0
    unrealized_pnl = 0.0
    for _, row in open_positions.iterrows():
        latest = _latest_price(price_lookup, row["symbol"])
        if latest is None:
            continue
        mark_value = row["shares"] * latest["adj_close"]
        unrealized_value += mark_value
        unrealized_pnl += mark_value - row["entry_value"] - row["entry_cost"]

    cash = initial_capital - committed_cash + closed["exit_value"].fillna(0.0).sum() - closed[
        "exit_cost"
    ].fillna(0.0).sum()
    equity = cash + unrealized_value
    return {
        "initial_capital": initial_capital,
        "cash": float(cash),
        "equity": float(equity),
        "realized_pnl": float(realized_pnl),
        "unrealized_pnl": float(unrealized_pnl),
        "open_positions": int(len(open_positions)),
        "closed_positions": int(len(closed)),
        "planned_orders": int((normalized["status"] == "planned").sum()),
        "skipped_orders": int((normalized["status"] == "skipped").sum()),
    }


def mark_paper_positions(
    ledger: pd.DataFrame,
    prices: pd.DataFrame,
    scores: pd.DataFrame | None = None,
    *,
    config: PaperTradingConfig | None = None,
) -> pd.DataFrame:
    """Mark open and planned paper positions using the latest available daily prices."""
    cfg = config or PaperTradingConfig()
    normalized = _normalize_ledger(ledger)
    price_lookup = _price_lookup(prices)
    score_lookup = _score_lookup(scores)
    rows = []
    for _, row in normalized.loc[normalized["status"].isin(["open", "planned"])].iterrows():
        latest = _latest_price(price_lookup, row["symbol"])
        latest_date = latest["date"] if latest is not None else pd.NaT
        latest_price = latest["adj_close"] if latest is not None else np.nan
        if row["status"] == "open":
            mark_value = row["shares"] * latest_price if latest is not None else np.nan
            unrealized_pnl = (
                mark_value - row["entry_value"] - row["entry_cost"]
                if latest is not None
                else np.nan
            )
            unrealized_return = (
                unrealized_pnl / row["entry_value"]
                if row["entry_value"] and latest is not None
                else 0.0
            )
            action = (
                "exit_pending"
                if _first_exit_decision(row, price_lookup, score_lookup, config=cfg) is not None
                else "hold"
            )
            state = _open_position_state(row, price_lookup, config=cfg) or {}
            current_score = _score_on_or_before(score_lookup, row["symbol"], latest_date)
        else:
            mark_value = 0.0
            unrealized_pnl = 0.0
            unrealized_return = 0.0
            action = "waiting_for_fill"
            state = {}
            current_score = np.nan

        rows.append(
            {
                "order_id": row["order_id"],
                "status": row["status"],
                "action": action,
                "symbol": row["symbol"],
                "sector": row["sector"],
                "planned_date": row["planned_date"],
                "fill_date": row["fill_date"],
                "target_exit_date": row["target_exit_date"],
                "latest_price_date": latest_date,
                "score": row["score"],
                "shares": row["shares"],
                "reference_price": row["reference_price"],
                "entry_price": row["entry_price"],
                "entry_value": row["entry_value"],
                "latest_price": latest_price,
                "mark_value": mark_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_return": unrealized_return,
                "days_open": _business_days_between(row["fill_date"], latest_date),
                "current_score": current_score,
                "entry_score": row["score"],
                "actual_exit_trigger_date": row["actual_exit_trigger_date"],
                "exit_reason": row["exit_reason"],
                "exit_signal_value": row["exit_signal_value"],
                "highest_close_since_entry": state.get(
                    "highest_close_since_entry", row["highest_close_since_entry"]
                ),
                "trailing_stop_activated": state.get(
                    "trailing_stop_activated", row["trailing_stop_activated"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _validate_config(config: PaperTradingConfig) -> None:
    if config.initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if not 0 < config.position_weight <= 1:
        raise ValueError("position_weight must be in (0, 1]")
    if not 0 < config.long_fraction <= 1:
        raise ValueError("long_fraction must be in (0, 1]")
    if config.horizon <= 0:
        raise ValueError("horizon must be positive")
    if config.exit_rules.horizon_days <= 0:
        raise ValueError("exit_rules.horizon_days must be positive")
    if config.transaction_cost_bps < 0 or config.slippage_bps < 0:
        raise ValueError("costs must be non-negative")
    if config.exit_rules.stop_loss.pct >= 0:
        raise ValueError("stop_loss.pct should be negative")
    if config.exit_rules.trailing_stop.trail_from_high_pct >= 0:
        raise ValueError("trailing stop trail_from_high_pct should be negative")
    if config.exit_rules.score_deterioration.min_days_held < 0:
        raise ValueError("score deterioration min_days_held must be non-negative")
    if config.exit_rules.rebalance.min_days_held < 0:
        raise ValueError("rebalance min_days_held must be non-negative")
    if config.exit_rules.rebalance.exit_below_score < 0:
        raise ValueError("rebalance exit_below_score must be non-negative")


def _business_days_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    if pd.isna(start) or pd.isna(end):
        return 0
    return len(pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))) - 1


def _base_order_row(
    row: pd.Series,
    config: PaperTradingConfig,
    target_exit_date: pd.Timestamp,
    status: str,
    skip_reason: str,
) -> dict:
    return {
        "order_id": "",
        "status": status,
        "planned_date": pd.Timestamp(row["date"]),
        "symbol": row["symbol"],
        "sector": row.get("sector", ""),
        "score": row["score"],
        "shares": 0.0,
        "reference_price": row["adj_close"],
        "estimated_entry_value": 0.0,
        "estimated_cost": 0.0,
        "target_exit_date": target_exit_date,
        "actual_exit_trigger_date": pd.NaT,
        "fill_date": pd.NaT,
        "entry_price": np.nan,
        "entry_value": 0.0,
        "entry_cost": 0.0,
        "exit_date": pd.NaT,
        "exit_price": np.nan,
        "exit_value": 0.0,
        "exit_cost": 0.0,
        "gross_pnl": 0.0,
        "net_pnl": 0.0,
        "return": 0.0,
        "exit_reason": "",
        "exit_signal_value": np.nan,
        "exit_rule_version": "",
        "highest_close_since_entry": np.nan,
        "trailing_stop_activated": False,
        "skip_reason": skip_reason,
    }


def _shares_for_target(*, target_dollars: float, price: float, allow_fractional: bool) -> float:
    if price <= 0:
        return 0.0
    shares = target_dollars / price
    return shares if allow_fractional else float(np.floor(shares))


def _trade_cost(value: float, *, transaction_cost_bps: float, slippage_bps: float) -> float:
    return value * (transaction_cost_bps + slippage_bps) / 10_000.0


def _target_exit_date(planned_date: pd.Timestamp, horizon: int) -> pd.Timestamp:
    return planned_date + pd.offsets.BDay(horizon)


def _configured_horizon_days(config: PaperTradingConfig) -> int:
    default_horizon = ExitRulesConfig().horizon_days
    if config.exit_rules.horizon_days != default_horizon:
        return config.exit_rules.horizon_days
    return config.horizon


def _order_ids(ledger: pd.DataFrame) -> list[str]:
    return [
        f"{pd.Timestamp(row.planned_date).date().isoformat()}-{row.symbol}-{index:03d}"
        for index, row in enumerate(ledger.itertuples(index=False), start=1)
    ]


def _normalize_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    output = ledger.copy()
    for column in PAPER_LEDGER_COLUMNS:
        if column not in output:
            output[column] = pd.NA
    for column in (
        "planned_date",
        "target_exit_date",
        "actual_exit_trigger_date",
        "fill_date",
        "exit_date",
    ):
        output[column] = pd.to_datetime(output[column])
    numeric_columns = [
        "score",
        "shares",
        "reference_price",
        "estimated_entry_value",
        "estimated_cost",
        "entry_price",
        "entry_value",
        "entry_cost",
        "exit_price",
        "exit_value",
        "exit_cost",
        "gross_pnl",
        "net_pnl",
        "return",
        "exit_signal_value",
        "highest_close_since_entry",
    ]
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    output["trailing_stop_activated"] = output["trailing_stop_activated"].fillna(False).astype(
        bool
    )
    for column in ("exit_reason", "exit_rule_version", "skip_reason"):
        output[column] = output[column].fillna("").astype(str)
    return output.loc[:, PAPER_LEDGER_COLUMNS]


def _dedupe_active_orders(ledger: pd.DataFrame) -> pd.DataFrame:
    output = ledger.copy()
    active = output.loc[output["status"].isin(["planned", "open"])].sort_values(
        ["planned_date", "fill_date", "order_id"],
        na_position="last",
    )
    seen: set[str] = set()
    for index, row in active.iterrows():
        symbol = row["symbol"].upper()
        if symbol in seen:
            _skip_order_in_place(output, index, reason="duplicate_active_symbol")
            continue
        seen.add(symbol)
    return output.loc[:, PAPER_LEDGER_COLUMNS]


def _skip_order_in_place(ledger: pd.DataFrame, index: int, *, reason: str) -> None:
    ledger.loc[
        index,
        [
            "status",
            "shares",
            "estimated_entry_value",
            "estimated_cost",
            "fill_date",
            "entry_price",
            "entry_value",
            "entry_cost",
            "exit_date",
            "exit_price",
            "exit_value",
            "exit_cost",
            "gross_pnl",
            "net_pnl",
            "return",
            "actual_exit_trigger_date",
            "exit_reason",
            "exit_signal_value",
            "exit_rule_version",
            "highest_close_since_entry",
            "trailing_stop_activated",
            "skip_reason",
        ],
    ] = [
        "skipped",
        0.0,
        0.0,
        0.0,
        pd.NaT,
        np.nan,
        0.0,
        0.0,
        pd.NaT,
        np.nan,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        pd.NaT,
        "",
        np.nan,
        "",
        np.nan,
        False,
        reason,
    ]


def _current_cash(ledger: pd.DataFrame, *, initial_capital: float) -> float:
    filled = ledger.loc[ledger["status"].isin(["open", "closed"])]
    closed = ledger.loc[ledger["status"] == "closed"]
    committed_cash = (
        filled["entry_value"].fillna(0.0).sum() + filled["entry_cost"].fillna(0.0).sum()
    )
    returned_cash = (
        closed["exit_value"].fillna(0.0).sum() - closed["exit_cost"].fillna(0.0).sum()
    )
    return float(initial_capital - committed_cash + returned_cash)


def _price_lookup(prices: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], pd.Series]:
    required = {"date", "symbol", "open", "close", "adj_close"}
    missing = required.difference(prices.columns)
    if missing:
        raise KeyError(f"prices are missing required columns: {sorted(missing)}")
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].str.upper()
    return {
        (row["symbol"], row["date"]): row
        for _, row in frame.sort_values(["symbol", "date"]).iterrows()
    }


def _volatility_lookup(
    prices: pd.DataFrame, *, lookback: int = 20
) -> dict[tuple[str, pd.Timestamp], float]:
    """Precompute rolling annualised volatility (adj_close) per symbol per date."""
    required = {"date", "symbol", "adj_close"}
    missing = required.difference(prices.columns)
    if missing:
        raise KeyError(f"prices are missing required columns: {sorted(missing)}")
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].str.upper()
    frame = frame.sort_values(["symbol", "date"])
    frame["daily_return"] = frame.groupby("symbol")["adj_close"].pct_change()
    vol = (
        frame.groupby("symbol")["daily_return"]
        .transform(lambda s: s.rolling(lookback, min_periods=lookback).std(ddof=1))
    )
    frame["volatility"] = vol
    return {
        (row["symbol"], row["date"]): row["volatility"]
        for _, row in frame.iterrows()
        if pd.notna(row["volatility"])
    }


def _sector_lookup(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    lookback: int = 5,
    min_records: int = 3,
    date_col: str = "date",
    symbol_col: str = "symbol",
    sector_col: str = "sector",
    price_col: str = "adj_close",
) -> dict[tuple[str, pd.Timestamp], float]:
    """Precompute rolling mean sector return for each (sector, date).

    Only sectors with at least ``min_records`` distinct symbols on a
    given date contribute to the lookup.
    """
    required_prices = {date_col, symbol_col, price_col}
    missing = required_prices.difference(prices.columns)
    if missing:
        raise KeyError(f"prices are missing required columns: {sorted(missing)}")
    required_universe = {symbol_col, sector_col}
    missing = required_universe.difference(universe.columns)
    if missing:
        raise KeyError(f"universe is missing required columns: {sorted(missing)}")

    frame = prices.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame[symbol_col] = frame[symbol_col].str.upper()
    univ = universe.copy()
    univ[symbol_col] = univ[symbol_col].str.upper()
    frame = frame.merge(univ[[symbol_col, sector_col]], on=symbol_col, how="inner")
    frame = frame.sort_values([symbol_col, date_col])
    frame["daily_return"] = frame.groupby(symbol_col)[price_col].pct_change()

    date_sector_counts = frame.groupby([date_col, sector_col])[symbol_col].nunique()
    valid_pairs = date_sector_counts[date_sector_counts >= min_records].reset_index()
    valid_pairs = valid_pairs[[date_col, sector_col]]
    frame = frame.merge(valid_pairs, on=[date_col, sector_col], how="inner")

    sector_daily = frame.groupby([date_col, sector_col])["daily_return"].mean().reset_index()
    sector_daily = sector_daily.sort_values([sector_col, date_col])
    sector_daily["sector_return"] = sector_daily.groupby(sector_col)[
        "daily_return"
    ].transform(lambda s: s.rolling(lookback, min_periods=lookback).mean())

    return {
        (row[sector_col], row[date_col]): row["sector_return"]
        for _, row in sector_daily.iterrows()
        if pd.notna(row["sector_return"])
    }


def _next_available_date(
    price_lookup: dict[tuple[str, pd.Timestamp], pd.Series],
    symbol: str,
    planned_date: pd.Timestamp,
) -> pd.Timestamp | None:
    dates = _symbol_dates(price_lookup, symbol)
    candidates = [date for date in dates if date > planned_date]
    return candidates[0] if candidates else None


def _first_available_on_or_after(
    price_lookup: dict[tuple[str, pd.Timestamp], pd.Series],
    symbol: str,
    target_date: pd.Timestamp,
) -> pd.Timestamp | None:
    dates = _symbol_dates(price_lookup, symbol)
    candidates = [date for date in dates if date >= target_date]
    return candidates[0] if candidates else None


def _latest_price(
    price_lookup: dict[tuple[str, pd.Timestamp], pd.Series],
    symbol: str,
) -> pd.Series | None:
    dates = _symbol_dates(price_lookup, symbol)
    return price_lookup[(symbol, dates[-1])] if dates else None


def _symbol_dates(
    price_lookup: dict[tuple[str, pd.Timestamp], pd.Series],
    symbol: str,
) -> list[pd.Timestamp]:
    return sorted(date for price_symbol, date in price_lookup if price_symbol == symbol)


def _first_exit_decision(
    row: pd.Series,
    price_lookup: dict[tuple[str, pd.Timestamp], pd.Series],
    score_lookup: dict[tuple[str, pd.Timestamp], float],
    *,
    vol_lookup: dict[tuple[str, pd.Timestamp], float] | None = None,
    sector_lookup: dict[tuple[str, pd.Timestamp], float] | None = None,
    config: PaperTradingConfig,
) -> dict | None:
    if pd.isna(row["fill_date"]):
        return None
    symbol = str(row["symbol"]).upper()
    dates = [
        date
        for date in _symbol_dates(price_lookup, symbol)
        if date >= pd.Timestamp(row["fill_date"])
    ]
    highest_close = (
        float(row["highest_close_since_entry"])
        if pd.notna(row["highest_close_since_entry"]) and row["highest_close_since_entry"] > 0
        else 0.0
    )
    trailing_active = bool(row["trailing_stop_activated"])
    for date in dates:
        price_row = price_lookup[(symbol, date)]
        current_close = float(price_row["adj_close"])
        highest_close = max(highest_close, current_close)
        entry_value = float(row["entry_value"])
        current_return = (
            (row["shares"] * current_close - entry_value - float(row["entry_cost"]))
            / entry_value
            if entry_value
            else 0.0
        )
        high_return = (
            (highest_close - float(row["entry_price"])) / float(row["entry_price"])
            if row["entry_price"]
            else 0.0
        )
        ts = config.exit_rules.trailing_stop
        tvs = config.exit_rules.trailing_volatility_stop
        if tvs.enabled or ts.enabled:
            activate_at = (
                tvs.activate_at_return if tvs.enabled
                else ts.activate_at_return
            )
            if high_return >= activate_at:
                trailing_active = True

        vol = (
            vol_lookup.get((symbol, date), np.nan)
            if vol_lookup
            else np.nan
        )
        sector = str(row.get("sector", ""))
        sector_return = (
            sector_lookup.get((sector, date), np.nan)
            if sector_lookup and sector
            else np.nan
        )
        decision = _exit_decision_for_date(
            row,
            date,
            current_close=current_close,
            current_return=current_return,
            highest_close=highest_close,
            trailing_active=trailing_active,
            current_score=_score_on_or_before(score_lookup, symbol, date),
            current_vol=vol,
            current_sector_return=sector_return,
            config=config,
        )
        if decision is not None:
            return decision
    return None


def _exit_decision_for_date(
    row: pd.Series,
    date: pd.Timestamp,
    *,
    current_close: float,
    current_return: float,
    highest_close: float,
    trailing_active: bool,
    current_score: float,
    current_vol: float = np.nan,
    current_sector_return: float = np.nan,
    config: PaperTradingConfig,
) -> dict | None:
    base = {
        "exit_date": date,
        "actual_exit_trigger_date": date,
        "exit_price": current_close,
        "highest_close_since_entry": highest_close,
        "trailing_stop_activated": trailing_active,
    }
    if (
        config.exit_rules.stop_loss.enabled
        and current_return <= config.exit_rules.stop_loss.pct
    ):
        return {**base, "exit_reason": "stop_loss", "exit_signal_value": current_return}

    if config.exit_rules.trailing_stop.enabled and trailing_active:
        drawdown_from_high = current_close / highest_close - 1.0 if highest_close else 0.0
        if drawdown_from_high <= config.exit_rules.trailing_stop.trail_from_high_pct:
            return {
                **base,
                "exit_reason": "trailing_stop",
                "exit_signal_value": drawdown_from_high,
            }

    vol_rule = config.exit_rules.trailing_volatility_stop
    if vol_rule.enabled and trailing_active and pd.notna(current_vol):
        raw = -(current_vol * vol_rule.volatility_multiple)
        trail_pct = min(raw, vol_rule.tightest_trail_pct)
        trail_pct = max(trail_pct, vol_rule.widest_trail_pct)
        drawdown_from_high = current_close / highest_close - 1.0 if highest_close else 0.0
        if drawdown_from_high <= trail_pct:
            return {
                **base,
                "exit_reason": "trailing_volatility_stop",
                "exit_signal_value": drawdown_from_high,
            }

    sector_rule = config.exit_rules.sector_stop
    if sector_rule.enabled and pd.notna(current_sector_return):
        if current_sector_return <= sector_rule.sector_decline_pct:
            return {
                **base,
                "exit_reason": "sector_stop",
                "exit_signal_value": current_sector_return,
            }

    score_rule = config.exit_rules.score_deterioration
    days_held = _business_days_between(row["fill_date"], date)
    if score_rule.enabled and days_held >= score_rule.min_days_held and pd.notna(current_score):
        entry_score = float(row["score"])
        score_decline = (
            (entry_score - current_score) / abs(entry_score)
            if entry_score
            else 0.0
        )
        if current_score <= score_rule.exit_below_score or score_decline >= (
            score_rule.exit_if_score_declines_pct
        ):
            return {
                **base,
                "exit_reason": "score_deterioration",
                "exit_signal_value": current_score,
            }

    if config.exit_rules.rebalance.enabled:
        rebalance_rule = config.exit_rules.rebalance
        if (
            days_held >= rebalance_rule.min_days_held
            and pd.notna(current_score)
            and current_score <= rebalance_rule.exit_below_score
        ):
            return {
                **base,
                "exit_reason": "rebalance",
                "exit_signal_value": current_score,
            }

    decay_rule = config.exit_rules.time_decay
    if decay_rule.enabled and days_held >= decay_rule.min_days_hold:
        entry_score = float(row["score"])
        if entry_score > 0:
            max_hold = int(
                decay_rule.half_life_days
                * (np.log(entry_score / decay_rule.min_score_for_decay) / np.log(2))
            )
            if days_held >= max_hold:
                remaining = 0.5 ** (days_held / decay_rule.half_life_days)
                return {
                    **base,
                    "exit_reason": "time_decay",
                    "exit_signal_value": remaining,
                }

    if date >= pd.Timestamp(row["target_exit_date"]):
        return {**base, "exit_reason": "horizon", "exit_signal_value": current_return}
    return None


def _open_position_state(
    row: pd.Series,
    price_lookup: dict[tuple[str, pd.Timestamp], pd.Series],
    *,
    config: PaperTradingConfig,
) -> dict | None:
    if pd.isna(row["fill_date"]):
        return None
    symbol = str(row["symbol"]).upper()
    dates = [
        date
        for date in _symbol_dates(price_lookup, symbol)
        if date >= pd.Timestamp(row["fill_date"])
    ]
    if not dates:
        return None
    highest_close = max(float(price_lookup[(symbol, date)]["adj_close"]) for date in dates)
    high_return = (
        (highest_close - float(row["entry_price"])) / float(row["entry_price"])
        if row["entry_price"]
        else 0.0
    )
    trailing_active = bool(row["trailing_stop_activated"])
    if not trailing_active:
        if config.exit_rules.trailing_volatility_stop.enabled:
            activate_at = config.exit_rules.trailing_volatility_stop.activate_at_return
        elif config.exit_rules.trailing_stop.enabled:
            activate_at = config.exit_rules.trailing_stop.activate_at_return
        else:
            activate_at = None
        if activate_at is not None and high_return >= activate_at:
            trailing_active = True
    return {
        "highest_close_since_entry": highest_close,
        "trailing_stop_activated": trailing_active,
    }


def _score_lookup(scores: pd.DataFrame | None) -> dict[tuple[str, pd.Timestamp], float]:
    if scores is None or scores.empty:
        return {}
    if "symbol" not in scores or "score" not in scores:
        return {}
    frame = scores.copy()
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"])
    else:
        frame["date"] = pd.Timestamp.max.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna(subset=["score"])
    return {
        (row["symbol"], row["date"]): float(row["score"])
        for _, row in frame.sort_values(["symbol", "date"]).iterrows()
    }


def _score_on_or_before(
    score_lookup: dict[tuple[str, pd.Timestamp], float],
    symbol: str,
    date: pd.Timestamp,
) -> float:
    if pd.isna(date):
        return np.nan
    symbol = str(symbol).upper()
    return score_lookup.get((symbol, pd.Timestamp(date)), np.nan)


def _adjusted_open(price_row: pd.Series) -> float:
    return float(price_row["open"] * price_row["adj_close"] / price_row["close"])
