from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from signalforge.experiments import FAST_MODEL_SPECS, FEATURE_SETS
from signalforge.modeling import BaselineModelConfig, _build_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a paper-only daily watchlist and order ledger."
    )
    parser.add_argument("--research-frame", default="data/processed/research_frame.csv")
    parser.add_argument("--output-prefix", default="reports/paper_portfolio")
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--feature-set", default="volatility_liquidity")
    parser.add_argument("--model", default="rf_fast")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--target-kind", choices=("exec_excess",), default="exec_excess")
    parser.add_argument("--initial-capital", type=float, default=2_000.0)
    parser.add_argument("--position-weight", type=float, default=0.10)
    parser.add_argument("--long-fraction", type=float, default=0.10)
    parser.add_argument("--min-score", type=float, default=0.01)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--allow-fractional-shares", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    research_frame = pd.read_csv(args.research_frame)
    research_frame["date"] = pd.to_datetime(research_frame["date"])
    as_of_date = _resolve_as_of_date(research_frame, args.as_of_date)
    feature_columns = FEATURE_SETS[args.feature_set]
    target_col = f"fwd_{args.horizon}d_exec_excess_return"
    realized_col = f"fwd_{args.horizon}d_exec_return"
    model_spec = _model_spec(args.model)

    label_cutoff = as_of_date - pd.offsets.BDay(args.horizon)
    train = research_frame.dropna(subset=[*feature_columns, target_col, realized_col]).copy()
    train = train.loc[train["date"] <= label_cutoff]
    if train.empty:
        raise ValueError("no training rows available before as-of date")

    candidates = research_frame.loc[research_frame["date"] == as_of_date].dropna(
        subset=[*feature_columns, "adj_close", "avg_dollar_volume_20d"]
    )
    if candidates.empty:
        raise ValueError(f"no candidate rows available for {as_of_date.date()}")

    config = BaselineModelConfig(
        target_col=target_col,
        realized_return_col=realized_col,
        model_type=model_spec.model_type,
        alpha=model_spec.alpha,
        n_estimators=model_spec.n_estimators,
        max_depth=model_spec.max_depth,
        min_samples_leaf=model_spec.min_samples_leaf,
        n_jobs=args.n_jobs,
    )
    model = _build_model(config)
    model.fit(train.loc[:, feature_columns], train[target_col])

    scored = candidates.copy()
    scored["score"] = model.predict(scored.loc[:, feature_columns])
    watchlist = _build_watchlist(scored, feature_columns=feature_columns)
    ledger = _build_order_ledger(
        scored,
        initial_capital=args.initial_capital,
        position_weight=args.position_weight,
        long_fraction=args.long_fraction,
        min_score=args.min_score,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        allow_fractional_shares=args.allow_fractional_shares,
    )
    summary = _summary_payload(
        args=args,
        as_of_date=as_of_date,
        train=train,
        candidates=scored,
        ledger=ledger,
        feature_columns=feature_columns,
        target_col=target_col,
        label_cutoff=label_cutoff,
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    watchlist.to_csv(output_prefix.with_name(output_prefix.name + "_watchlist.csv"), index=False)
    ledger.to_csv(output_prefix.with_name(output_prefix.name + "_order_ledger.csv"), index=False)
    output_prefix.with_name(output_prefix.name + "_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(f"wrote paper portfolio artifacts with prefix {output_prefix}")


def _resolve_as_of_date(frame: pd.DataFrame, raw_date: str | None) -> pd.Timestamp:
    if raw_date:
        as_of_date = pd.Timestamp(raw_date)
        if as_of_date not in set(frame["date"]):
            raise ValueError(f"as-of date {raw_date!r} is not present in research frame")
        return as_of_date
    return frame["date"].max()


def _model_spec(name: str):
    for spec in FAST_MODEL_SPECS:
        if spec.name == name:
            return spec
    raise ValueError(f"unsupported paper model: {name!r}")


def _build_watchlist(scored: pd.DataFrame, *, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    columns = [
        "date",
        "symbol",
        "sector",
        "industry",
        "score",
        "adj_close",
        "avg_dollar_volume_20d",
        *feature_columns,
    ]
    return scored.loc[:, [column for column in columns if column in scored.columns]].sort_values(
        "score",
        ascending=False,
    )


def _build_order_ledger(
    scored: pd.DataFrame,
    *,
    initial_capital: float,
    position_weight: float,
    long_fraction: float,
    min_score: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    allow_fractional_shares: bool,
) -> pd.DataFrame:
    ranked = scored.sort_values("score", ascending=False).reset_index(drop=True)
    desired_count = max(1, int(len(ranked) * long_fraction))
    target_dollars = initial_capital * position_weight
    remaining_cash = initial_capital
    planned = 0
    rows = []
    for _, row in ranked.iterrows():
        if row["score"] < min_score:
            rows.append(_skipped_row(row, "score_below_threshold"))
            continue
        if planned >= desired_count:
            rows.append(_skipped_row(row, "rank_below_position_count"))
            continue

        reference_price = row["adj_close"]
        shares = target_dollars / reference_price
        if not allow_fractional_shares:
            shares = float(np.floor(shares))
        estimated_entry_value = shares * reference_price
        estimated_cost = estimated_entry_value * (
            transaction_cost_bps + slippage_bps
        ) / 10_000.0
        required_cash = estimated_entry_value + estimated_cost
        if shares <= 0 or required_cash <= 0:
            rows.append(_skipped_row(row, "size_too_small"))
            continue
        if required_cash > remaining_cash:
            rows.append(_skipped_row(row, "insufficient_cash"))
            continue

        remaining_cash -= required_cash
        planned += 1
        rows.append(
            {
                "status": "planned",
                "date": row["date"],
                "symbol": row["symbol"],
                "sector": row.get("sector", ""),
                "score": row["score"],
                "reference_price": reference_price,
                "shares": shares,
                "estimated_entry_value": estimated_entry_value,
                "estimated_cost": estimated_cost,
                "estimated_required_cash": required_cash,
                "remaining_cash_after_order": remaining_cash,
                "skip_reason": "",
            }
        )
    return pd.DataFrame(rows)


def _skipped_row(row: pd.Series, reason: str) -> dict:
    return {
        "status": "skipped",
        "date": row["date"],
        "symbol": row["symbol"],
        "sector": row.get("sector", ""),
        "score": row["score"],
        "reference_price": row["adj_close"],
        "shares": 0.0,
        "estimated_entry_value": 0.0,
        "estimated_cost": 0.0,
        "estimated_required_cash": 0.0,
        "remaining_cash_after_order": np.nan,
        "skip_reason": reason,
    }


def _summary_payload(
    *,
    args: argparse.Namespace,
    as_of_date: pd.Timestamp,
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    ledger: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_col: str,
    label_cutoff: pd.Timestamp,
) -> dict:
    planned = ledger.loc[ledger["status"] == "planned"]
    return {
        "mode": "paper_only_no_broker_execution",
        "as_of_date": as_of_date.date().isoformat(),
        "target_col": target_col,
        "feature_set": args.feature_set,
        "feature_columns": list(feature_columns),
        "model": args.model,
        "train_start": train["date"].min().date().isoformat(),
        "train_end": train["date"].max().date().isoformat(),
        "label_availability_cutoff": label_cutoff.date().isoformat(),
        "train_rows": int(len(train)),
        "candidate_count": int(len(candidates)),
        "planned_order_count": int(len(planned)),
        "planned_estimated_entry_value": float(planned["estimated_entry_value"].sum()),
        "initial_capital": args.initial_capital,
        "position_weight": args.position_weight,
        "long_fraction": args.long_fraction,
        "min_score": args.min_score,
        "reference_price_note": (
            "Uses latest adjusted close as a planning reference; actual next-open fill "
            "must be reconciled later."
        ),
    }


if __name__ == "__main__":
    main()
