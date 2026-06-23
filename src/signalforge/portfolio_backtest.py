from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LANE_ORDER = (
    "momentum_breakouts",
    "sector_leaders",
    "volume_anomalies",
    "quality_growth",
    "value_recoveries",
)


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    starting_capital: float = 100_000.0
    rebalance: str = "monthly"
    selected_lanes: tuple[str, ...] = ()
    max_positions: int = 25
    position_sizing_method: str = "equal_weight"
    cost_bps: float = 0.0
    price_col: str = "adj_close"
    max_position_weight: float | None = None


@dataclass(frozen=True)
class PortfolioBacktestResult:
    daily_returns: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame
    summary: dict
    config: PortfolioBacktestConfig


def run_portfolio_backtest(
    backtest_trades: pd.DataFrame,
    research_frame: pd.DataFrame,
    *,
    config: PortfolioBacktestConfig | None = None,
) -> PortfolioBacktestResult:
    """Run a simple long-only portfolio simulation from discovery lane selections."""
    cfg = config or PortfolioBacktestConfig()
    _validate_config(cfg)
    selections = _normalize_backtest_trades(backtest_trades)
    prices = _price_pivot(research_frame, price_col=cfg.price_col)

    if cfg.selected_lanes:
        selections = selections.loc[selections["lane"].isin(cfg.selected_lanes)]
    if selections.empty or prices.empty:
        return _empty_result(cfg)

    date_map = _align_rebalance_dates(selections["as_of_date"], prices.index)
    selections["rebalance_date"] = selections["as_of_date"].map(date_map)
    selections = selections.dropna(subset=["rebalance_date"])
    if selections.empty:
        return _empty_result(cfg)
    selections["rebalance_date"] = pd.to_datetime(selections["rebalance_date"])

    returns = prices.pct_change().dropna(how="all")

    targets_by_date = {
        pd.Timestamp(date): _target_holdings_for_date(
            group, prices.loc[pd.Timestamp(date)], cfg, returns=returns
        )
        for date, group in selections.groupby("rebalance_date", sort=True)
    }
    rebalance_dates = tuple(sorted(targets_by_date))
    simulation_dates = prices.loc[prices.index >= rebalance_dates[0]].index

    shares: dict[str, float] = {}
    cash = float(cfg.starting_capital)
    previous_value = float(cfg.starting_capital)
    daily_rows = []
    holding_rows = []
    trade_rows = []

    for date in simulation_dates:
        date_prices = prices.loc[date]
        gross_value = _portfolio_value(shares, cash, date_prices)
        gross_return = gross_value / previous_value - 1.0 if previous_value > 0 else 0.0
        transaction_cost = 0.0
        portfolio_value = gross_value

        if date in targets_by_date:
            targets = targets_by_date[date]
            holding_rows.extend(_holding_rows(date, targets))
            rebalance = _apply_rebalance(
                shares,
                cash,
                date_prices,
                targets,
                portfolio_value=gross_value,
                cost_bps=cfg.cost_bps,
            )
            shares = rebalance["shares"]
            cash = rebalance["cash"]
            transaction_cost = rebalance["transaction_cost"]
            portfolio_value = rebalance["portfolio_value"]
            trade_rows.extend(_dated_trade_rows(date, rebalance["trades"]))

        net_return = portfolio_value / previous_value - 1.0 if previous_value > 0 else 0.0
        daily_rows.append(
            {
                "date": date,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "portfolio_value": portfolio_value,
            }
        )
        previous_value = portfolio_value

    daily_returns = pd.DataFrame(daily_rows)
    holdings = _ordered_holdings_frame(pd.DataFrame(holding_rows))
    portfolio_trades = _ordered_trades_frame(pd.DataFrame(trade_rows))
    summary = calculate_summary_metrics(
        daily_returns,
        portfolio_trades,
        holdings,
        starting_capital=cfg.starting_capital,
    )
    return PortfolioBacktestResult(
        daily_returns=daily_returns,
        holdings=holdings,
        trades=portfolio_trades,
        summary=summary,
        config=cfg,
    )


def deduplicate_lane_selections(selections: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-lane rows to one ranked row per symbol."""
    if selections.empty:
        return _empty_deduped_frame()
    if "symbol" not in selections.columns:
        raise KeyError("backtest selections are missing required column: 'symbol'")

    frame = selections.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    if "lane" not in frame.columns:
        frame["lane"] = ""
    frame["lane"] = frame["lane"].astype(str)
    score_column = "composite_score" if "composite_score" in frame.columns else "score"
    if score_column in frame.columns:
        frame["_ranking_score"] = pd.to_numeric(frame[score_column], errors="coerce")
    else:
        frame["_ranking_score"] = np.nan
    frame["_rank"] = pd.to_numeric(frame["rank"], errors="coerce") if "rank" in frame else np.nan

    rows = []
    for symbol, group in frame.groupby("symbol", sort=True):
        lanes = _ordered_lanes(group["lane"].dropna().astype(str).tolist())
        best = group.sort_values(
            ["_ranking_score", "_rank", "symbol"],
            ascending=[False, True, True],
            na_position="last",
        ).iloc[0]
        rows.append(
            {
                "symbol": symbol,
                "lanes_matched": ", ".join(lanes),
                "lane_source": ", ".join(lanes),
                "lane_count": len(lanes),
                "score": best.get("_ranking_score", np.nan),
                "rank": best.get("_rank", np.nan),
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["score", "rank", "symbol"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def build_portfolio_targets(
    selections: pd.DataFrame,
    *,
    max_positions: int,
    position_sizing_method: str = "equal_weight",
    max_position_weight: float | None = None,
    returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build deterministic target portfolio weights from lane selections.

    Supported sizing methods:
      - equal_weight: uniform 1/N allocation.
      - inverse_volatility: weight proportional to 1/volatility from past returns.
    """
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")

    deduped = deduplicate_lane_selections(selections).head(max_positions).copy()
    if deduped.empty:
        return _empty_targets_frame()
    selected_symbols = set(deduped["symbol"])

    if position_sizing_method == "equal_weight":
        weight = 1.0 / len(deduped)
        if max_position_weight is not None:
            weight = min(weight, max_position_weight)
        deduped["weight"] = weight

    elif position_sizing_method == "inverse_volatility":
        if returns is None or returns.empty:
            raise ValueError("returns frame is required for inverse_volatility sizing")
        volatilities = (
            returns.loc[:, returns.columns.isin(selected_symbols)]
            .std()
            .dropna()
        )
        available = set(volatilities.index)
        eligible = deduped.loc[deduped["symbol"].isin(available)].copy()
        if eligible.empty:
            raise ValueError("no symbols have valid volatility estimates")
        inv_vol = 1.0 / volatilities[eligible["symbol"]]
        weights = inv_vol / inv_vol.sum()
        if max_position_weight is not None:
            for _ in range(20):
                clipped = weights.clip(upper=max_position_weight)
                clipped_sum = clipped.sum()
                if clipped_sum <= 0:
                    raise ValueError("max_position_weight too low for any position")
                new_weights = clipped / clipped_sum
                if new_weights.max() <= max_position_weight + 1e-10:
                    weights = new_weights
                    break
                weights = new_weights
            weights = weights.clip(upper=max_position_weight)
            weights = weights / weights.sum()
        eligible["weight"] = weights.values

        excluded = deduped.loc[~deduped["symbol"].isin(available)]
        if not excluded.empty:
            excluded = excluded.copy()
            excluded["weight"] = 0.0
            deduped = pd.concat([eligible, excluded], ignore_index=True)
        else:
            deduped = eligible

    else:
        raise ValueError(
            f"unsupported position_sizing_method: {position_sizing_method}; "
            "supported: equal_weight, inverse_volatility"
        )

    return deduped.loc[
        :, ["symbol", "weight", "lane_source", "lanes_matched", "lane_count", "score"]
    ].reset_index(drop=True)


def calculate_turnover(
    old_weights: Mapping[str, float],
    new_weights: Mapping[str, float],
) -> float:
    """Return one-way traded weight as the sum of absolute weight changes."""
    symbols = set(old_weights).union(new_weights)
    return float(
        sum(abs(new_weights.get(symbol, 0.0) - old_weights.get(symbol, 0.0)) for symbol in symbols)
    )


def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """Calculate max drawdown from a portfolio value series."""
    values = pd.to_numeric(portfolio_values, errors="coerce").dropna()
    if values.empty:
        return 0.0
    running_high = values.cummax()
    drawdowns = values.div(running_high).sub(1.0)
    return float(drawdowns.min())


def calculate_summary_metrics(
    daily_returns: pd.DataFrame,
    portfolio_trades: pd.DataFrame,
    holdings: pd.DataFrame,
    *,
    starting_capital: float,
) -> dict:
    """Summarize portfolio-level performance and trading activity."""
    if daily_returns.empty:
        ending_capital = float(starting_capital)
        net_returns = pd.Series(dtype="float64")
        portfolio_values = pd.Series([starting_capital])
    else:
        ending_capital = float(daily_returns["portfolio_value"].iloc[-1])
        net_returns = pd.to_numeric(daily_returns["net_return"], errors="coerce").fillna(0.0)
        portfolio_values = pd.to_numeric(daily_returns["portfolio_value"], errors="coerce")

    total_return = ending_capital / starting_capital - 1.0 if starting_capital > 0 else 0.0
    periods = max(len(daily_returns) - 1, 0)
    annualized_return = (
        (1.0 + total_return) ** (252.0 / periods) - 1.0
        if periods > 0 and total_return > -1.0
        else total_return
    )
    daily_std = float(net_returns.std(ddof=0)) if not net_returns.empty else 0.0
    volatility = daily_std * float(np.sqrt(252.0))
    mean_return = float(net_returns.mean()) if not net_returns.empty else 0.0
    sharpe_ratio = mean_return / daily_std * float(np.sqrt(252.0)) if daily_std > 0 else None

    turnover_by_date = (
        portfolio_trades.groupby("rebalance_date")["trade_weight"].sum()
        if not portfolio_trades.empty
        else pd.Series(dtype="float64")
    )
    positions_by_date = (
        holdings.groupby("rebalance_date")["symbol"].count()
        if not holdings.empty
        else pd.Series(dtype="float64")
    )
    total_transaction_cost = (
        float(daily_returns["transaction_cost"].sum()) if not daily_returns.empty else 0.0
    )
    rebalance_dates = set()
    if not holdings.empty:
        rebalance_dates.update(holdings["rebalance_date"].astype(str).tolist())
    if not portfolio_trades.empty:
        rebalance_dates.update(portfolio_trades["rebalance_date"].astype(str).tolist())

    return {
        "starting_capital": float(starting_capital),
        "ending_capital": ending_capital,
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": volatility,
        "sharpe_ratio": None if sharpe_ratio is None else float(sharpe_ratio),
        "max_drawdown": calculate_max_drawdown(portfolio_values),
        "average_turnover": float(turnover_by_date.mean()) if not turnover_by_date.empty else 0.0,
        "total_transaction_cost": total_transaction_cost,
        "number_of_rebalances": len(rebalance_dates),
        "number_of_positions_average": (
            float(positions_by_date.mean()) if not positions_by_date.empty else 0.0
        ),
    }


def write_portfolio_backtest_outputs(
    result: PortfolioBacktestResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write portfolio backtest CSV, JSON, and markdown artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    daily_path = output_path / "portfolio_daily_returns.csv"
    holdings_path = output_path / "portfolio_holdings.csv"
    trades_path = output_path / "portfolio_trades.csv"
    summary_path = output_path / "portfolio_summary.json"
    report_path = output_path / "portfolio_report.md"

    result.daily_returns.to_csv(daily_path, index=False)
    result.holdings.to_csv(holdings_path, index=False)
    result.trades.to_csv(trades_path, index=False)
    summary_path.write_text(json.dumps(result.summary, indent=2, default=str) + "\n")
    report_path.write_text(render_portfolio_report(result))

    return {
        "daily_returns": daily_path,
        "holdings": holdings_path,
        "trades": trades_path,
        "summary": summary_path,
        "report": report_path,
    }


def render_portfolio_report(result: PortfolioBacktestResult) -> str:
    """Render a concise markdown report for a portfolio backtest."""
    summary = result.summary
    lines = [
        "# Portfolio Backtest Report",
        "",
        f"- Date range: {_date_range_label(result.daily_returns)}",
        f"- Selected lanes: {_selected_lanes_label(result.config.selected_lanes)}",
        f"- Rebalance frequency: {result.config.rebalance}",
        f"- Starting capital: {_format_currency(summary['starting_capital'])}",
        f"- Ending capital: {_format_currency(summary['ending_capital'])}",
        f"- Total return: {_format_pct(summary['total_return'])}",
        f"- Annualized return: {_format_pct(summary['annualized_return'])}",
        f"- Annualized volatility: {_format_pct(summary['annualized_volatility'])}",
        f"- Sharpe ratio: {_format_number(summary['sharpe_ratio'])}",
        f"- Max drawdown: {_format_pct(summary['max_drawdown'])}",
        f"- Average turnover: {_format_pct(summary['average_turnover'])}",
        f"- Total transaction costs: {_format_currency(summary['total_transaction_cost'])}",
        "",
        "## Caveats",
        "",
        "- Simplified long-only simulation.",
        "- No slippage beyond configured transaction cost.",
        "- No taxes.",
        "- No shorting.",
        "- No intraday execution modeling.",
        "- Depends on point-in-time feature quality.",
        "",
    ]
    return "\n".join(lines)


def _target_holdings_for_date(
    selections: pd.DataFrame,
    date_prices: pd.Series,
    config: PortfolioBacktestConfig,
    returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    available = set(date_prices.dropna().index.astype(str))
    candidates = selections.loc[selections["symbol"].astype(str).str.upper().isin(available)]
    return build_portfolio_targets(
        candidates,
        max_positions=config.max_positions,
        position_sizing_method=config.position_sizing_method,
        max_position_weight=config.max_position_weight,
        returns=returns,
    )


def _apply_rebalance(
    shares: Mapping[str, float],
    cash: float,
    prices: pd.Series,
    targets: pd.DataFrame,
    *,
    portfolio_value: float,
    cost_bps: float,
) -> dict:
    old_values = {
        symbol: float(quantity) * float(prices.get(symbol, np.nan))
        for symbol, quantity in shares.items()
        if pd.notna(prices.get(symbol, np.nan))
    }
    old_weights = {
        symbol: value / portfolio_value
        for symbol, value in old_values.items()
        if portfolio_value > 0 and abs(value) > 1e-12
    }
    new_weights = {
        row["symbol"]: float(row["weight"])
        for _, row in targets.iterrows()
        if pd.notna(prices.get(row["symbol"], np.nan))
    }
    trade_rows = []
    cost_rate = cost_bps / 10_000.0
    total_transaction_cost = 0.0
    for symbol in sorted(set(old_weights).union(new_weights)):
        old_weight = old_weights.get(symbol, 0.0)
        new_weight = new_weights.get(symbol, 0.0)
        trade_weight = abs(new_weight - old_weight)
        if trade_weight <= 1e-12:
            continue
        traded_value = trade_weight * portfolio_value
        transaction_cost = traded_value * cost_rate
        total_transaction_cost += transaction_cost
        trade_rows.append(
            {
                "symbol": symbol,
                "old_weight": old_weight,
                "new_weight": new_weight,
                "trade_weight": trade_weight,
                "traded_value": traded_value,
                "transaction_cost": transaction_cost,
                "side": "buy" if new_weight > old_weight else "sell",
                "buy_value": traded_value if new_weight > old_weight else 0.0,
                "sell_value": traded_value if new_weight < old_weight else 0.0,
            }
        )

    new_shares = {}
    invested_value = 0.0
    for symbol, weight in new_weights.items():
        price = float(prices.get(symbol))
        target_value = portfolio_value * weight
        invested_value += target_value
        new_shares[symbol] = target_value / price

    new_cash = portfolio_value - total_transaction_cost - invested_value
    return {
        "shares": new_shares,
        "cash": new_cash,
        "trades": trade_rows,
        "transaction_cost": total_transaction_cost,
        "portfolio_value": portfolio_value - total_transaction_cost,
    }


def _portfolio_value(
    shares: Mapping[str, float],
    cash: float,
    prices: pd.Series,
) -> float:
    value = float(cash)
    for symbol, quantity in shares.items():
        price = prices.get(symbol, np.nan)
        if pd.notna(price):
            value += float(quantity) * float(price)
    return value


def _normalize_backtest_trades(backtest_trades: pd.DataFrame) -> pd.DataFrame:
    required = {"as_of_date", "symbol", "lane"}
    missing = required.difference(backtest_trades.columns)
    if missing:
        raise KeyError(f"backtest_trades is missing required columns: {sorted(missing)}")
    normalized = backtest_trades.copy()
    normalized["as_of_date"] = pd.to_datetime(normalized["as_of_date"])
    normalized["symbol"] = normalized["symbol"].astype(str).str.upper()
    normalized["lane"] = normalized["lane"].astype(str)
    if "score" in normalized.columns:
        normalized["score"] = pd.to_numeric(normalized["score"], errors="coerce")
    return normalized.sort_values(["as_of_date", "lane", "symbol"]).reset_index(drop=True)


def _price_pivot(research_frame: pd.DataFrame, *, price_col: str) -> pd.DataFrame:
    required = {"date", "symbol", price_col}
    missing = required.difference(research_frame.columns)
    if missing:
        raise KeyError(f"research_frame is missing required columns: {sorted(missing)}")
    frame = research_frame.loc[:, ["date", "symbol", price_col]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame[price_col] = pd.to_numeric(frame[price_col], errors="coerce")
    return (
        frame.pivot_table(index="date", columns="symbol", values=price_col, aggfunc="last")
        .sort_index()
        .ffill()
    )


def _align_rebalance_dates(
    raw_dates: pd.Series,
    price_dates: pd.Index,
) -> dict[pd.Timestamp, pd.Timestamp]:
    sorted_price_dates = pd.Series(pd.to_datetime(price_dates).sort_values().unique())
    mapping = {}
    for raw_date in sorted(pd.to_datetime(raw_dates).dropna().unique()):
        eligible = sorted_price_dates.loc[sorted_price_dates >= pd.Timestamp(raw_date)]
        if not eligible.empty:
            mapping[pd.Timestamp(raw_date)] = pd.Timestamp(eligible.iloc[0])
    return mapping


def _holding_rows(date: pd.Timestamp, targets: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in targets.iterrows():
        rows.append(
            {
                "rebalance_date": date,
                "symbol": row["symbol"],
                "weight": row["weight"],
                "lane_source": row["lane_source"],
                "lanes_matched": row["lanes_matched"],
                "score": row["score"],
            }
        )
    return rows


def _dated_trade_rows(date: pd.Timestamp, trades: list[dict]) -> list[dict]:
    return [{"rebalance_date": date, **trade} for trade in trades]


def _ordered_holdings_frame(holdings: pd.DataFrame) -> pd.DataFrame:
    columns = ["rebalance_date", "symbol", "weight", "lane_source", "lanes_matched", "score"]
    if holdings.empty:
        return pd.DataFrame(columns=columns)
    return holdings.loc[:, columns].sort_values(["rebalance_date", "symbol"]).reset_index(drop=True)


def _ordered_trades_frame(trades: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rebalance_date",
        "symbol",
        "old_weight",
        "new_weight",
        "trade_weight",
        "traded_value",
        "transaction_cost",
        "side",
        "buy_value",
        "sell_value",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    return trades.loc[:, columns].sort_values(["rebalance_date", "symbol"]).reset_index(drop=True)


def _empty_result(config: PortfolioBacktestConfig) -> PortfolioBacktestResult:
    daily_returns = pd.DataFrame(
        columns=["date", "gross_return", "transaction_cost", "net_return", "portfolio_value"]
    )
    holdings = _ordered_holdings_frame(pd.DataFrame())
    trades = _ordered_trades_frame(pd.DataFrame())
    summary = calculate_summary_metrics(
        daily_returns,
        trades,
        holdings,
        starting_capital=config.starting_capital,
    )
    return PortfolioBacktestResult(
        daily_returns=daily_returns,
        holdings=holdings,
        trades=trades,
        summary=summary,
        config=config,
    )


def _empty_deduped_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["symbol", "lanes_matched", "lane_source", "lane_count", "score", "rank"]
    )


def _empty_targets_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["symbol", "weight", "lane_source", "lanes_matched", "lane_count", "score"]
    )


def _ordered_lanes(lanes: list[str]) -> list[str]:
    lane_set = {lane for lane in lanes if lane}
    ordered = [lane for lane in LANE_ORDER if lane in lane_set]
    ordered.extend(sorted(lane_set.difference(ordered)))
    return ordered


def _date_range_label(daily_returns: pd.DataFrame) -> str:
    if daily_returns.empty:
        return "no simulation dates"
    dates = pd.to_datetime(daily_returns["date"])
    return f"{dates.min().date().isoformat()} to {dates.max().date().isoformat()}"


def _selected_lanes_label(lanes: tuple[str, ...]) -> str:
    return ", ".join(lanes) if lanes else "all lanes"


def _format_currency(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"${float(value):,.2f}"


def _format_pct(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _format_number(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def _validate_config(config: PortfolioBacktestConfig) -> None:
    if config.starting_capital <= 0:
        raise ValueError("starting_capital must be positive")
    if config.rebalance not in {"monthly", "weekly"}:
        raise ValueError("rebalance must be 'monthly' or 'weekly'")
    if config.max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if config.position_sizing_method not in {"equal_weight", "inverse_volatility"}:
        raise ValueError(
            f"unsupported position_sizing_method: {config.position_sizing_method}; "
            "supported: equal_weight, inverse_volatility"
        )
    if config.cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    if config.max_position_weight is not None and not 0 < config.max_position_weight <= 1:
        raise ValueError("max_position_weight must be between 0 and 1 when provided")
