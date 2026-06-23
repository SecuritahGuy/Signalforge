from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from signalforge.paper import (
    EXIT_RULE_VERSION,
    PAPER_LEDGER_COLUMNS,
    PaperTradingConfig,
    _normalize_ledger,
    _trade_cost,
)

INTRADAY_MARK_COLUMNS = (
    "timestamp",
    "symbol",
    "price",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
)

INTRADAY_EXIT_RULE_VERSION = f"{EXIT_RULE_VERSION}.intraday.v1"


def normalize_intraday_marks(frame: pd.DataFrame, *, source: str = "unknown") -> pd.DataFrame:
    """Normalize intraday mark rows into the local monitoring contract."""
    output = frame.copy()
    if "timestamp" not in output and "date" in output:
        output = output.rename(columns={"date": "timestamp"})
    if "close" not in output and "price" in output:
        output["close"] = output["price"]
    if "price" not in output and "close" in output:
        output["price"] = output["close"]
    for column in ("open", "high", "low"):
        if column not in output:
            output[column] = output["price"]
    if "volume" not in output:
        output["volume"] = 0.0
    if "source" not in output:
        output["source"] = source

    missing = {"timestamp", "symbol", "price"}.difference(output.columns)
    if missing:
        raise KeyError(f"intraday marks are missing required columns: {sorted(missing)}")

    output["timestamp"] = pd.to_datetime(output["timestamp"]).dt.tz_localize(None)
    output["symbol"] = output["symbol"].astype(str).str.upper()
    for column in ("price", "open", "high", "low", "close", "volume"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.dropna(subset=["timestamp", "symbol", "price"])
    output["close"] = output["close"].fillna(output["price"])
    output["open"] = output["open"].fillna(output["price"])
    output["high"] = output["high"].fillna(output["price"])
    output["low"] = output["low"].fillna(output["price"])
    output["volume"] = output["volume"].fillna(0.0)
    return output.loc[:, INTRADAY_MARK_COLUMNS].sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def latest_intraday_marks(marks: pd.DataFrame) -> pd.DataFrame:
    """Return the latest mark per symbol."""
    normalized = normalize_intraday_marks(marks)
    return (
        normalized.sort_values(["symbol", "timestamp"])
        .groupby("symbol", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def evaluate_intraday_risk_exits(
    ledger: pd.DataFrame,
    marks: pd.DataFrame,
    *,
    daily_prices: pd.DataFrame | None = None,
    config: PaperTradingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate stop-loss and trailing-stop exits from latest intraday marks."""
    cfg = config or PaperTradingConfig()
    output = _normalize_ledger(ledger)
    latest_marks = latest_intraday_marks(marks)
    mark_by_symbol = {
        str(row["symbol"]).upper(): row for _, row in latest_marks.iterrows()
    }
    daily_highs = _daily_highs_since_entry(output, daily_prices)
    decisions = []

    for index, row in output.loc[output["status"] == "open"].iterrows():
        symbol = str(row["symbol"]).upper()
        mark = mark_by_symbol.get(symbol)
        if mark is None:
            decisions.append(_decision_row(row, None, "no_mark", triggered=False))
            continue

        decision = _intraday_exit_decision(
            row,
            mark,
            daily_high=daily_highs.get(symbol, 0.0),
            config=cfg,
        )
        decisions.append(decision)
        if not decision["triggered"]:
            output.loc[
                index,
                ["highest_close_since_entry", "trailing_stop_activated"],
            ] = [
                decision["highest_price_since_entry"],
                decision["trailing_stop_activated"],
            ]
            continue

        exit_price = float(decision["price"])
        exit_value = float(row["shares"]) * exit_price
        exit_cost = _trade_cost(
            exit_value,
            transaction_cost_bps=cfg.transaction_cost_bps,
            slippage_bps=cfg.slippage_bps,
        )
        gross_pnl = exit_value - float(row["entry_value"])
        net_pnl = gross_pnl - float(row["entry_cost"]) - exit_cost
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
            decision["timestamp"],
            decision["timestamp"],
            exit_price,
            exit_value,
            exit_cost,
            gross_pnl,
            net_pnl,
            net_pnl / float(row["entry_value"]) if float(row["entry_value"]) else 0.0,
            decision["exit_reason"],
            decision["exit_signal_value"],
            INTRADAY_EXIT_RULE_VERSION,
            decision["highest_price_since_entry"],
            decision["trailing_stop_activated"],
        ]

    return output.loc[:, PAPER_LEDGER_COLUMNS], pd.DataFrame(decisions)


def _intraday_exit_decision(
    row: pd.Series,
    mark: pd.Series,
    *,
    daily_high: float,
    config: PaperTradingConfig,
) -> dict:
    price = float(mark["price"])
    high = max(float(mark.get("high", price)), price)
    existing_high = float(row["highest_close_since_entry"])
    highest_price = max(existing_high, daily_high, high, price)
    entry_value = float(row["entry_value"])
    current_return = (
        (float(row["shares"]) * price - entry_value - float(row["entry_cost"])) / entry_value
        if entry_value
        else 0.0
    )
    entry_price = float(row["entry_price"])
    high_return = highest_price / entry_price - 1.0 if entry_price else 0.0
    trailing_active = bool(row["trailing_stop_activated"]) or (
        config.exit_rules.trailing_stop.enabled
        and high_return >= config.exit_rules.trailing_stop.activate_at_return
    )

    base = _decision_row(row, mark, "hold", triggered=False)
    base.update(
        {
            "price": price,
            "current_return": current_return,
            "highest_price_since_entry": highest_price,
            "trailing_stop_activated": trailing_active,
        }
    )

    if (
        config.exit_rules.stop_loss.enabled
        and current_return <= config.exit_rules.stop_loss.pct
    ):
        return {
            **base,
            "triggered": True,
            "action": "exit",
            "exit_reason": "intraday_stop_loss",
            "exit_signal_value": current_return,
        }

    if config.exit_rules.trailing_stop.enabled and trailing_active:
        drawdown_from_high = price / highest_price - 1.0 if highest_price else 0.0
        if drawdown_from_high <= config.exit_rules.trailing_stop.trail_from_high_pct:
            return {
                **base,
                "triggered": True,
                "action": "exit",
                "exit_reason": "intraday_trailing_stop",
                "exit_signal_value": drawdown_from_high,
            }

    return base


def _decision_row(
    row: pd.Series,
    mark: pd.Series | None,
    action: str,
    *,
    triggered: bool,
) -> dict:
    timestamp = pd.NaT if mark is None else mark["timestamp"]
    price = float("nan") if mark is None else float(mark["price"])
    return {
        "timestamp": timestamp,
        "symbol": row["symbol"],
        "status": row["status"],
        "action": action,
        "triggered": triggered,
        "price": price,
        "current_return": float("nan"),
        "highest_price_since_entry": float(row.get("highest_close_since_entry", 0.0) or 0.0),
        "trailing_stop_activated": bool(row.get("trailing_stop_activated", False)),
        "exit_reason": "",
        "exit_signal_value": float("nan"),
    }


def _daily_highs_since_entry(
    ledger: pd.DataFrame,
    daily_prices: pd.DataFrame | None,
) -> dict[str, float]:
    if daily_prices is None or daily_prices.empty:
        return {}
    prices = daily_prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    prices["symbol"] = prices["symbol"].astype(str).str.upper()
    price_col = "high" if "high" in prices else "adj_close"
    highs = {}
    for _, row in ledger.loc[ledger["status"] == "open"].iterrows():
        if pd.isna(row["fill_date"]):
            continue
        symbol = str(row["symbol"]).upper()
        filled = pd.Timestamp(row["fill_date"]).normalize()
        matches = prices.loc[(prices["symbol"] == symbol) & (prices["date"] >= filled)]
        if not matches.empty:
            highs[symbol] = float(pd.to_numeric(matches[price_col], errors="coerce").max())
    return highs


def open_symbols(ledger: pd.DataFrame) -> list[str]:
    normalized = _normalize_ledger(ledger)
    return sorted(set(normalized.loc[normalized["status"] == "open", "symbol"].str.upper()))


def normalize_yfinance_intraday(raw: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    """Normalize a yfinance intraday download into intraday mark rows."""
    if raw.empty:
        raise ValueError("yfinance returned no intraday data")
    frames = []
    normalized_symbols = [symbol.upper() for symbol in symbols]
    if isinstance(raw.columns, pd.MultiIndex):
        for symbol in normalized_symbols:
            if symbol not in raw.columns.get_level_values(0):
                continue
            frames.append(_normalize_yfinance_symbol_intraday(raw[symbol].copy(), symbol))
    else:
        if len(normalized_symbols) != 1:
            raise ValueError("single-index yfinance data requires exactly one symbol")
        frames.append(_normalize_yfinance_symbol_intraday(raw.copy(), normalized_symbols[0]))
    if not frames:
        raise ValueError("no requested symbols were present in yfinance response")
    return normalize_intraday_marks(pd.concat(frames, ignore_index=True), source="yahoo_intraday")


def _normalize_yfinance_symbol_intraday(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "price",
        "Volume": "volume",
    }
    normalized = frame.rename(columns=rename_map).reset_index()
    timestamp_column = "Datetime" if "Datetime" in normalized.columns else normalized.columns[0]
    normalized = normalized.rename(columns={timestamp_column: "timestamp"})
    if "price" not in normalized and "close" in normalized:
        normalized["price"] = normalized["close"]
    normalized["symbol"] = symbol
    return normalized
