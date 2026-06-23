from __future__ import annotations

import pandas as pd

from signalforge.backtest import BacktestConfig, build_daily_positions, long_short_daily_returns


def daily_portfolio_diagnostics(
    predictions: pd.DataFrame,
    *,
    realized_return_col: str,
    score_col: str = "prediction",
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Build per-day raw and risk-controlled portfolio returns from predictions."""
    required = {"date", "symbol", "split_id", score_col, realized_return_col}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"predictions is missing required columns: {sorted(missing)}")

    frames = []
    for split_id, split_predictions in predictions.groupby("split_id", sort=True):
        split_returns = long_short_daily_returns(
            split_predictions.rename(
                columns={score_col: "score", realized_return_col: "forward_return"}
            ),
            config=config,
        )
        split_returns["split_id"] = split_id
        frames.append(split_returns)

    daily = pd.concat(frames, ignore_index=True).sort_values(["split_id", "date"])
    daily["raw_equity"] = (1.0 + daily["net_return"]).cumprod()
    daily["risk_equity_full"] = (1.0 + daily["risk_net_return"]).cumprod()
    return daily.reset_index(drop=True)


def monthly_portfolio_returns(
    daily_returns: pd.DataFrame,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """Compound daily raw and risk-controlled returns by calendar month."""
    required = {date_col, "net_return", "risk_net_return"}
    missing = required.difference(daily_returns.columns)
    if missing:
        raise KeyError(f"daily_returns is missing required columns: {sorted(missing)}")

    frame = daily_returns.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame["month"] = frame[date_col].dt.to_period("M").astype(str)
    return (
        frame.groupby("month", as_index=False)
        .agg(
            raw_return=("net_return", lambda values: (1.0 + values).prod() - 1.0),
            risk_return=("risk_net_return", lambda values: (1.0 + values).prod() - 1.0),
            trading_days=("risk_trading_enabled", "sum"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )


def symbol_contribution_diagnostics(
    predictions: pd.DataFrame,
    *,
    realized_return_col: str,
    score_col: str = "prediction",
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Estimate symbol-level gross contribution from long/short selections."""
    cfg = config or BacktestConfig()
    required = {"date", "symbol", score_col, realized_return_col}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"predictions is missing required columns: {sorted(missing)}")

    rows = []
    group_cols = ["split_id", "date"] if "split_id" in predictions.columns else ["date"]
    grouped = predictions.dropna(subset=[score_col, realized_return_col]).groupby(
        group_cols,
        sort=True,
    )
    symbol_trade_counts_by_split: dict[object, dict[str, int]] = {}
    for group_key, day in grouped:
        split_id, date = group_key if isinstance(group_key, tuple) else (None, group_key)
        symbol_trade_counts = symbol_trade_counts_by_split.setdefault(split_id, {})
        positions = build_daily_positions(
            day.rename(columns={score_col: "score", realized_return_col: "forward_return"}),
            symbol_trade_counts=symbol_trade_counts,
            config=cfg,
        )
        positions["date"] = date
        if split_id is not None:
            positions["split_id"] = split_id
        rows.extend(positions.to_dict(orient="records"))

    positions = pd.DataFrame(rows)
    if positions.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "long_days",
                "short_days",
                "gross_contribution",
                "avg_daily_contribution",
                "avg_abs_weight",
            ]
        )

    return (
        positions.groupby("symbol", as_index=False)
        .agg(
            long_days=("side", lambda values: int((values == "long").sum())),
            short_days=("side", lambda values: int((values == "short").sum())),
            gross_contribution=("contribution", "sum"),
            avg_daily_contribution=("contribution", "mean"),
            avg_abs_weight=("weight", lambda values: values.abs().mean()),
        )
        .sort_values("gross_contribution", ascending=False)
        .reset_index(drop=True)
    )
