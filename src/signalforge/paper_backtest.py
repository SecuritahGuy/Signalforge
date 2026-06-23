from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from signalforge.metrics import max_drawdown, sharpe_ratio
from signalforge.paper import (
    PAPER_LEDGER_COLUMNS,
    PaperTradingConfig,
    build_planned_orders,
)


@dataclass(frozen=True)
class PaperBacktestResult:
    ledger: pd.DataFrame
    daily_equity: pd.DataFrame
    summary: dict
    config: PaperTradingConfig


def run_paper_style_backtest(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    config: PaperTradingConfig | None = None,
    score_col: str = "prediction",
    start_date: str | None = None,
    end_date: str | None = None,
) -> PaperBacktestResult:
    """Run a historical simulation through the same lifecycle as paper trading."""
    cfg = config or PaperTradingConfig()
    signal_frame = _normalize_signals(signals, score_col=score_col)
    price_frame = _normalize_prices(prices)
    if start_date is not None:
        signal_frame = signal_frame.loc[signal_frame["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        signal_frame = signal_frame.loc[signal_frame["date"] <= pd.Timestamp(end_date)]
        price_frame = price_frame.loc[price_frame["date"] <= pd.Timestamp(end_date)]

    ledger_rows: list[dict] = []
    planned_indices: list[int] = []
    open_indices: list[int] = []
    latest_prices: dict[str, pd.Series] = {}
    cash = cfg.initial_capital
    daily_rows = []
    prices_by_date = {
        date: group.set_index("symbol")
        for date, group in price_frame.groupby("date", sort=True)
    }
    all_dates = sorted(set(price_frame["date"]).union(signal_frame["date"]))
    signal_dates = set(signal_frame["date"])
    for current_date in all_dates:
        current_prices = prices_by_date.get(current_date)
        if current_prices is not None:
            for symbol, row in current_prices.iterrows():
                latest_prices[symbol] = row

        cash = _fill_planned_orders(
            ledger_rows,
            planned_indices,
            open_indices,
            current_prices,
            current_date=current_date,
            cash=cash,
            config=cfg,
        )
        cash = _close_due_positions(
            ledger_rows,
            open_indices,
            current_prices,
            current_date=current_date,
            cash=cash,
            config=cfg,
        )

        if current_date in signal_dates:
            excluded_symbols = _active_symbols_from_rows(ledger_rows)
            new_orders = build_planned_orders(
                signal_frame.loc[signal_frame["date"] == current_date],
                config=cfg,
                available_cash=cash,
                excluded_symbols=excluded_symbols,
            )
            for order in new_orders.to_dict(orient="records"):
                ledger_rows.append(order)
                if order["status"] == "planned":
                    planned_indices.append(len(ledger_rows) - 1)

        marks = _mark_open_rows(ledger_rows, open_indices, latest_prices)
        realized_pnl = sum(row["net_pnl"] for row in ledger_rows if row["status"] == "closed")
        equity = cash + sum(mark["mark_value"] for mark in marks)
        daily_rows.append(
            {
                "date": current_date,
                "initial_capital": cfg.initial_capital,
                "cash": cash,
                "equity": equity,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": equity - cash - sum(
                    row["entry_value"] + row["entry_cost"]
                    for row in ledger_rows
                    if row["status"] == "open"
                ),
                "open_positions": len(open_indices),
                "closed_positions": sum(row["status"] == "closed" for row in ledger_rows),
                "planned_orders": len(planned_indices),
                "skipped_orders": sum(row["status"] == "skipped" for row in ledger_rows),
                "gross_exposure": float(sum(mark["mark_value"] for mark in marks)),
                "open_symbols": len(
                    {row["symbol"] for row in ledger_rows if row["status"] == "open"}
                ),
            }
        )

    ledger = pd.DataFrame(ledger_rows, columns=PAPER_LEDGER_COLUMNS)
    daily_equity = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    if not daily_equity.empty:
        daily_equity["net_return"] = daily_equity["equity"].pct_change().fillna(
            daily_equity["equity"].div(cfg.initial_capital).sub(1.0)
        )
        daily_equity["drawdown"] = daily_equity["equity"].div(
            daily_equity["equity"].cummax()
        ).sub(1.0)
    summary = summarize_paper_backtest(ledger, daily_equity, initial_capital=cfg.initial_capital)
    return PaperBacktestResult(
        ledger=ledger.reset_index(drop=True),
        daily_equity=daily_equity,
        summary=summary,
        config=cfg,
    )


def summarize_paper_backtest(
    ledger: pd.DataFrame,
    daily_equity: pd.DataFrame,
    *,
    initial_capital: float,
) -> dict:
    filled = ledger.loc[ledger["status"].isin(["open", "closed"])] if not ledger.empty else ledger
    closed = ledger.loc[ledger["status"] == "closed"] if not ledger.empty else ledger
    skipped = ledger.loc[ledger["status"] == "skipped"] if not ledger.empty else ledger
    ending_equity = (
        float(daily_equity["equity"].iloc[-1]) if not daily_equity.empty else initial_capital
    )
    net_returns = (
        pd.to_numeric(daily_equity["net_return"], errors="coerce").fillna(0.0)
        if not daily_equity.empty
        else pd.Series(dtype="float64")
    )
    return {
        "initial_capital": float(initial_capital),
        "ending_equity": ending_equity,
        "total_return": ending_equity / initial_capital - 1.0,
        "sharpe": sharpe_ratio(net_returns),
        "max_drawdown": max_drawdown(net_returns),
        "planned_orders": int((ledger["status"] == "planned").sum()) if not ledger.empty else 0,
        "open_positions": int((ledger["status"] == "open").sum()) if not ledger.empty else 0,
        "closed_positions": int(len(closed)),
        "filled_positions": int(len(filled)),
        "skipped_orders": int(len(skipped)),
        "realized_pnl": float(closed["net_pnl"].sum()) if not closed.empty else 0.0,
        "closed_win_rate": float((closed["net_pnl"] > 0).mean()) if not closed.empty else 0.0,
        "avg_closed_pnl": float(closed["net_pnl"].mean()) if not closed.empty else 0.0,
    }


def _normalize_signals(signals: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    required = {"date", "symbol", "adj_close", score_col}
    missing = required.difference(signals.columns)
    if missing:
        raise KeyError(f"signals are missing required columns: {sorted(missing)}")
    frame = signals.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    if score_col != "score":
        frame = frame.rename(columns={score_col: "score"})
    return frame.sort_values(["date", "score"], ascending=[True, False]).reset_index(drop=True)


def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "open", "close", "adj_close"}
    missing = required.difference(prices.columns)
    if missing:
        raise KeyError(f"prices are missing required columns: {sorted(missing)}")
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def _fill_planned_orders(
    ledger_rows: list[dict],
    planned_indices: list[int],
    open_indices: list[int],
    current_prices: pd.DataFrame | None,
    *,
    current_date: pd.Timestamp,
    cash: float,
    config: PaperTradingConfig,
) -> float:
    if current_prices is None:
        return cash
    remaining_planned = []
    for index in planned_indices:
        row = ledger_rows[index]
        if (
            current_date <= pd.Timestamp(row["planned_date"])
            or row["symbol"] not in current_prices.index
        ):
            remaining_planned.append(index)
            continue
        price_row = current_prices.loc[row["symbol"]]
        entry_price = _adjusted_open(price_row)
        entry_value = row["shares"] * entry_price
        entry_cost = _trade_cost(entry_value, config=config)
        required_cash = entry_value + entry_cost
        if required_cash > cash:
            row.update(
                {
                    "status": "skipped",
                    "shares": 0.0,
                    "estimated_entry_value": 0.0,
                    "estimated_cost": 0.0,
                    "fill_date": pd.NaT,
                    "entry_price": 0.0,
                    "entry_value": 0.0,
                    "entry_cost": 0.0,
                    "skip_reason": "insufficient_cash_at_fill",
                }
            )
            continue
        row.update(
            {
                "status": "open",
                "fill_date": current_date,
                "entry_price": entry_price,
                "entry_value": entry_value,
                "entry_cost": entry_cost,
            }
        )
        cash -= required_cash
        open_indices.append(index)
    planned_indices[:] = remaining_planned
    return cash


def _close_due_positions(
    ledger_rows: list[dict],
    open_indices: list[int],
    current_prices: pd.DataFrame | None,
    *,
    current_date: pd.Timestamp,
    cash: float,
    config: PaperTradingConfig,
) -> float:
    if current_prices is None:
        return cash
    remaining_open = []
    for index in open_indices:
        row = ledger_rows[index]
        if (
            current_date < pd.Timestamp(row["target_exit_date"])
            or row["symbol"] not in current_prices.index
        ):
            remaining_open.append(index)
            continue
        price_row = current_prices.loc[row["symbol"]]
        exit_price = float(price_row["adj_close"])
        exit_value = row["shares"] * exit_price
        exit_cost = _trade_cost(exit_value, config=config)
        gross_pnl = exit_value - row["entry_value"]
        net_pnl = gross_pnl - row["entry_cost"] - exit_cost
        row.update(
            {
                "status": "closed",
                "actual_exit_trigger_date": current_date,
                "exit_date": current_date,
                "exit_price": exit_price,
                "exit_value": exit_value,
                "exit_cost": exit_cost,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "return": net_pnl / row["entry_value"] if row["entry_value"] else 0.0,
                "exit_reason": "horizon",
                "exit_signal_value": net_pnl / row["entry_value"] if row["entry_value"] else 0.0,
                "exit_rule_version": "paper_backtest.horizon",
                "highest_close_since_entry": max(
                    _safe_float(row.get("highest_close_since_entry")),
                    exit_price,
                ),
            }
        )
        cash += exit_value - exit_cost
    open_indices[:] = remaining_open
    return cash


def _mark_open_rows(
    ledger_rows: list[dict],
    open_indices: list[int],
    latest_prices: dict[str, pd.Series],
) -> list[dict]:
    marks = []
    for index in open_indices:
        row = ledger_rows[index]
        latest = latest_prices.get(row["symbol"])
        if latest is None:
            continue
        marks.append(
            {
                "symbol": row["symbol"],
                "mark_value": row["shares"] * float(latest["adj_close"]),
            }
        )
    return marks


def _safe_float(value: object) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0.0 if pd.isna(parsed) else float(parsed)


def _active_symbols_from_rows(ledger_rows: list[dict]) -> set[str]:
    return {
        row["symbol"].upper()
        for row in ledger_rows
        if row["status"] in {"planned", "open"}
    }


def _trade_cost(value: float, *, config: PaperTradingConfig) -> float:
    return value * (config.transaction_cost_bps + config.slippage_bps) / 10_000.0


def _adjusted_open(price_row: pd.Series) -> float:
    return float(price_row["open"] * price_row["adj_close"] / price_row["close"])
