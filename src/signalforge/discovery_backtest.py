from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from signalforge.discovery import (
    WATCHLIST_SCORE_COLUMNS,
    DiscoveryConfig,
    apply_discovery_filters,
    build_discovery_watchlists,
    score_discovery_candidates,
)
from signalforge.discovery_report import LANE_DISPLAY_NAMES, lane_candidate_reason


@dataclass(frozen=True)
class DiscoveryLaneBacktestConfig:
    rebalance: str = "monthly"
    top_n_per_lane: int = 25
    horizons: tuple[int, ...] = (5, 20, 60)
    price_col: str = "adj_close"
    discovery_config: DiscoveryConfig = DiscoveryConfig()


@dataclass(frozen=True)
class DiscoveryLaneBacktestResult:
    trades: pd.DataFrame
    summary: pd.DataFrame
    as_of_dates: tuple[pd.Timestamp, ...]
    config: DiscoveryLaneBacktestConfig


def run_discovery_lane_backtest(
    research_frame: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    existing_watchlist: pd.DataFrame | list[str] | tuple[str, ...] | set[str] | None = None,
    config: DiscoveryLaneBacktestConfig | None = None,
) -> DiscoveryLaneBacktestResult:
    """Run a lightweight historical signal-quality backtest for discovery lanes."""
    cfg = config or DiscoveryLaneBacktestConfig()
    _validate_backtest_config(cfg)
    frame = _normalize_research_frame(research_frame, price_col=cfg.price_col)
    as_of_dates = select_historical_as_of_dates(
        frame,
        start_date=start_date,
        end_date=end_date,
        rebalance=cfg.rebalance,
    )
    watchlist_symbols = _normalize_symbols(existing_watchlist)
    discovery_config = replace(cfg.discovery_config, top_n=cfg.top_n_per_lane)

    trade_rows = []
    for as_of_date in as_of_dates:
        snapshot = latest_rows_as_of(frame, as_of_date)
        if snapshot.empty:
            continue

        candidates, _ = apply_discovery_filters(
            snapshot,
            existing_watchlist=watchlist_symbols,
            config=discovery_config,
        )
        scored = score_discovery_candidates(candidates, config=discovery_config)
        watchlists = build_discovery_watchlists(scored, config=discovery_config)

        for lane, lane_frame in watchlists.items():
            if lane_frame.empty:
                continue
            score_col = WATCHLIST_SCORE_COLUMNS[lane]
            for rank, (_, row) in enumerate(lane_frame.iterrows(), start=1):
                trade = _base_trade_row(
                    row,
                    as_of_date=as_of_date,
                    lane=lane,
                    rank=rank,
                    score_col=score_col,
                    price_col=cfg.price_col,
                )
                trade.update(
                    forward_return_payload(
                        frame,
                        symbol=str(row["symbol"]),
                        as_of_date=as_of_date,
                        selection_price=row[cfg.price_col],
                        horizons=cfg.horizons,
                        price_col=cfg.price_col,
                    )
                )
                trade_rows.append(trade)

    trades = pd.DataFrame(trade_rows)
    if not trades.empty:
        trades = trades.sort_values(["as_of_date", "lane", "rank", "symbol"]).reset_index(
            drop=True
        )
    trades = _ordered_trade_frame(trades, horizons=cfg.horizons)
    summary = aggregate_lane_backtest(trades, horizons=cfg.horizons)
    return DiscoveryLaneBacktestResult(
        trades=trades,
        summary=summary,
        as_of_dates=tuple(as_of_dates),
        config=cfg,
    )


def select_historical_as_of_dates(
    frame: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    rebalance: str = "monthly",
    date_col: str = "date",
) -> tuple[pd.Timestamp, ...]:
    """Select deterministic historical rebalance dates from available rows."""
    if date_col not in frame.columns:
        raise KeyError(f"frame is missing required column: {date_col!r}")
    if rebalance not in {"monthly", "weekly"}:
        raise ValueError("rebalance must be 'monthly' or 'weekly'")

    dates = pd.Series(pd.to_datetime(frame[date_col].dropna()).sort_values().unique())
    if start_date is not None:
        dates = dates.loc[dates >= pd.Timestamp(start_date)]
    if end_date is not None:
        dates = dates.loc[dates <= pd.Timestamp(end_date)]
    if dates.empty:
        return tuple()

    date_frame = pd.DataFrame({"date": dates})
    if rebalance == "monthly":
        date_frame["period"] = date_frame["date"].dt.to_period("M")
    else:
        date_frame["period"] = date_frame["date"].dt.to_period("W-FRI")
    selected = date_frame.groupby("period", sort=True)["date"].max()
    return tuple(pd.Timestamp(date) for date in selected.tolist())


def latest_rows_as_of(
    frame: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Select each symbol's latest row available on or before the as-of date."""
    required = {date_col, symbol_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"frame is missing required columns: {sorted(missing)}")
    cutoff = pd.Timestamp(as_of_date)
    eligible = frame.loc[pd.to_datetime(frame[date_col]) <= cutoff].copy()
    if eligible.empty:
        return eligible
    eligible[date_col] = pd.to_datetime(eligible[date_col])
    eligible[symbol_col] = eligible[symbol_col].astype(str).str.upper()
    return (
        eligible.sort_values([symbol_col, date_col])
        .groupby(symbol_col, as_index=False, sort=True)
        .tail(1)
        .reset_index(drop=True)
    )


def forward_return_payload(
    frame: pd.DataFrame,
    *,
    symbol: str,
    as_of_date: str | pd.Timestamp,
    selection_price: float,
    horizons: tuple[int, ...],
    price_col: str = "adj_close",
) -> dict:
    """Build forward price and return fields for one selected symbol."""
    payload = {}
    symbol_frame = frame.loc[frame["symbol"].astype(str).str.upper() == symbol.upper()].sort_values(
        "date"
    )
    future = symbol_frame.loc[pd.to_datetime(symbol_frame["date"]) > pd.Timestamp(as_of_date)]
    for horizon in horizons:
        price_col_name = f"forward_price_{horizon}d"
        return_col_name = f"forward_return_{horizon}d"
        if len(future) < horizon or pd.isna(selection_price):
            payload[price_col_name] = pd.NA
            payload[return_col_name] = pd.NA
            continue
        forward_price = future.iloc[horizon - 1][price_col]
        payload[price_col_name] = forward_price
        payload[return_col_name] = (
            pd.NA if pd.isna(forward_price) else forward_price / selection_price - 1.0
        )
    return payload


def aggregate_lane_backtest(
    trades: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Aggregate forward-return metrics by lane and horizon."""
    rows = []
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "lane",
                "horizon",
                "selections",
                "avg_forward_return",
                "median_forward_return",
                "win_rate",
                "best_return",
                "worst_return",
            ]
        )

    for lane in WATCHLIST_SCORE_COLUMNS:
        lane_trades = trades.loc[trades["lane"] == lane]
        if lane_trades.empty:
            continue
        for horizon in horizons:
            return_col = f"forward_return_{horizon}d"
            returns = pd.to_numeric(lane_trades[return_col], errors="coerce").dropna()
            rows.append(
                {
                    "lane": lane,
                    "horizon": horizon,
                    "selections": int(len(returns)),
                    "avg_forward_return": returns.mean() if not returns.empty else pd.NA,
                    "median_forward_return": returns.median() if not returns.empty else pd.NA,
                    "win_rate": (returns > 0).mean() if not returns.empty else pd.NA,
                    "best_return": returns.max() if not returns.empty else pd.NA,
                    "worst_return": returns.min() if not returns.empty else pd.NA,
                }
            )
    return pd.DataFrame(rows)


def write_discovery_lane_backtest_outputs(
    result: DiscoveryLaneBacktestResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write discovery lane backtest CSV and markdown artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trades_path = output_path / "backtest_trades.csv"
    summary_path = output_path / "backtest_summary.csv"
    report_path = output_path / "backtest_report.md"

    result.trades.to_csv(trades_path, index=False)
    result.summary.to_csv(summary_path, index=False)
    report_path.write_text(render_discovery_lane_backtest_report(result))
    return {
        "trades": trades_path,
        "summary": summary_path,
        "report": report_path,
    }


def render_discovery_lane_backtest_report(result: DiscoveryLaneBacktestResult) -> str:
    """Render a concise markdown report for a discovery lane backtest."""
    date_range = _date_range_label(result.as_of_dates)
    horizon_label = ", ".join(f"{horizon}d" for horizon in result.config.horizons)
    lines = [
        "# Discovery Lane Backtest Report",
        "",
        f"- Date range: {date_range}",
        f"- Rebalance frequency: {result.config.rebalance}",
        f"- Horizons tested: {horizon_label}",
        f"- Total selections: {len(result.trades)}",
        "",
        "## Summary By Lane",
        "",
    ]
    if result.summary.empty:
        lines.append("No completed forward-return observations were available.")
    else:
        lines.extend(_summary_table(result.summary))

    best_worst = _best_worst_lane_lines(result.summary)
    if best_worst:
        lines.extend(["", "## Best/Worst Lanes", "", *best_worst])

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- No transaction costs.",
            "- No slippage.",
            "- Not a live trading strategy.",
            "- Depends on feature availability and point-in-time quality.",
            "",
        ]
    )
    return "\n".join(lines)


def _normalize_research_frame(frame: pd.DataFrame, *, price_col: str) -> pd.DataFrame:
    required = {"date", "symbol", price_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"research_frame is missing required columns: {sorted(missing)}")
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized["symbol"] = normalized["symbol"].astype(str).str.upper()
    normalized[price_col] = pd.to_numeric(normalized[price_col], errors="coerce")
    return normalized.sort_values(["symbol", "date"]).reset_index(drop=True)


def _normalize_symbols(
    symbols: pd.DataFrame | list[str] | tuple[str, ...] | set[str] | None,
) -> set[str]:
    if symbols is None:
        return set()
    if isinstance(symbols, pd.DataFrame):
        if "symbol" not in symbols.columns:
            raise KeyError("existing_watchlist is missing required column: 'symbol'")
        raw_symbols = symbols["symbol"]
    else:
        raw_symbols = symbols
    return {str(symbol).strip().upper() for symbol in raw_symbols if str(symbol).strip()}


def _base_trade_row(
    row: pd.Series,
    *,
    as_of_date: pd.Timestamp,
    lane: str,
    rank: int,
    score_col: str,
    price_col: str,
) -> dict:
    return {
        "as_of_date": as_of_date,
        "symbol": row["symbol"],
        "lane": lane,
        "rank": rank,
        "score": row.get(score_col, pd.NA),
        "selection_price": row.get(price_col, pd.NA),
        "sector": row.get("sector", ""),
        "lane_reason": lane_candidate_reason(lane, row),
    }


def _ordered_trade_frame(trades: pd.DataFrame, *, horizons: tuple[int, ...]) -> pd.DataFrame:
    columns = ["as_of_date", "symbol", "lane", "rank", "score", "selection_price"]
    for horizon in horizons:
        columns.extend([f"forward_price_{horizon}d", f"forward_return_{horizon}d"])
    columns.extend(["sector", "lane_reason"])
    if trades.empty:
        return pd.DataFrame(columns=columns)
    extra_columns = [column for column in trades.columns if column not in columns]
    ordered_columns = [
        column for column in [*columns, *extra_columns] if column in trades.columns
    ]
    return trades.loc[:, ordered_columns]


def _validate_backtest_config(config: DiscoveryLaneBacktestConfig) -> None:
    if config.rebalance not in {"monthly", "weekly"}:
        raise ValueError("rebalance must be 'monthly' or 'weekly'")
    if config.top_n_per_lane <= 0:
        raise ValueError("top_n_per_lane must be positive")
    if not config.horizons:
        raise ValueError("horizons must not be empty")
    if any(horizon <= 0 for horizon in config.horizons):
        raise ValueError("horizons must be positive")


def _date_range_label(as_of_dates: tuple[pd.Timestamp, ...]) -> str:
    if not as_of_dates:
        return "no rebalance dates"
    return f"{as_of_dates[0].date().isoformat()} to {as_of_dates[-1].date().isoformat()}"


def _summary_table(summary: pd.DataFrame) -> list[str]:
    lines = [
        "| Lane | Horizon | Selections | Avg Return | Median Return | Win Rate | Best | Worst |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.sort_values(["lane", "horizon"]).iterrows():
        lines.append(
            "| "
            f"{LANE_DISPLAY_NAMES.get(row['lane'], row['lane'])} | "
            f"{int(row['horizon'])} | "
            f"{int(row['selections'])} | "
            f"{_format_pct(row['avg_forward_return'])} | "
            f"{_format_pct(row['median_forward_return'])} | "
            f"{_format_pct(row['win_rate'])} | "
            f"{_format_pct(row['best_return'])} | "
            f"{_format_pct(row['worst_return'])} |"
        )
    return lines


def _best_worst_lane_lines(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return []
    lines = []
    for horizon, horizon_summary in summary.groupby("horizon", sort=True):
        valid = horizon_summary.dropna(subset=["avg_forward_return"])
        if valid.empty:
            continue
        best = valid.sort_values(["avg_forward_return", "lane"], ascending=[False, True]).iloc[0]
        worst = valid.sort_values(["avg_forward_return", "lane"], ascending=[True, True]).iloc[0]
        lines.append(
            f"- {int(horizon)}d best: {LANE_DISPLAY_NAMES.get(best['lane'], best['lane'])} "
            f"({_format_pct(best['avg_forward_return'])}); worst: "
            f"{LANE_DISPLAY_NAMES.get(worst['lane'], worst['lane'])} "
            f"({_format_pct(worst['avg_forward_return'])})."
        )
    return lines


def _format_pct(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2%}"
