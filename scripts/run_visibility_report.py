from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from signalforge.experiments import FAST_MODEL_SPECS, FEATURE_SETS
from signalforge.metrics import information_coefficient
from signalforge.modeling import BaselineModelConfig, _build_model, _feature_importance
from signalforge.validation import walk_forward_splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build model visibility artifacts for the selected SignalForge candidate."
    )
    parser.add_argument(
        "--predictions",
        default="reports/exec_top_experiment_weight10_score001_no_stop_predictions.csv",
    )
    parser.add_argument(
        "--ledger",
        default="reports/exec_top_experiment_weight10_score001_no_stop_trade_ledger.csv",
    )
    parser.add_argument("--research-frame", default="data/processed/research_frame.csv")
    parser.add_argument("--paper-ledger", default="reports/paper_portfolio_order_ledger.csv")
    parser.add_argument("--paper-watchlist", default="reports/paper_portfolio_watchlist.csv")
    parser.add_argument("--output-prefix", default="reports/model_visibility")
    parser.add_argument("--feature-set", default="volatility_liquidity")
    parser.add_argument("--model", default="rf_fast")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--target", default="fwd_20d_exec_excess_return")
    parser.add_argument("--realized-return", default="fwd_20d_exec_return")
    parser.add_argument("--score-col", default="prediction")
    parser.add_argument("--min-score", type=float, default=0.01)
    parser.add_argument("--first-train-start", default="2020-01-01")
    parser.add_argument("--first-validation-start", default="2022-01-01")
    parser.add_argument("--validation-months", type=int, default=6)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    ledger = pd.read_csv(args.ledger)
    research_frame = pd.read_csv(args.research_frame)
    paper_ledger = pd.read_csv(args.paper_ledger)
    paper_watchlist = pd.read_csv(args.paper_watchlist)

    for frame in (predictions, research_frame, paper_ledger, paper_watchlist):
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
    if "signal_date" in ledger.columns:
        ledger["signal_date"] = pd.to_datetime(ledger["signal_date"])

    feature_columns = FEATURE_SETS[args.feature_set]
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    enriched_predictions = _attach_reference_columns(predictions, research_frame)
    score_buckets = _score_bucket_report(
        enriched_predictions,
        score_col=args.score_col,
        target_col=args.target,
        realized_return_col=args.realized_return,
    )
    feature_by_split = _feature_importance_by_split(
        research_frame,
        feature_columns=feature_columns,
        args=args,
    )
    prediction_drift = _prediction_drift_report(
        enriched_predictions,
        score_col=args.score_col,
        min_score=args.min_score,
    )
    sector_contribution = _sector_contribution_report(ledger, research_frame)
    paper_explanations = _paper_explanation_report(
        paper_ledger,
        paper_watchlist,
        feature_by_split,
        feature_columns=feature_columns,
    )

    _write_csv(score_buckets, output_prefix, "score_buckets")
    _write_csv(feature_by_split, output_prefix, "feature_importance_by_split")
    _write_csv(prediction_drift, output_prefix, "prediction_drift")
    _write_csv(sector_contribution, output_prefix, "sector_contribution")
    _write_csv(paper_explanations, output_prefix, "paper_pick_explanations")
    _write_markdown_summary(
        output_prefix,
        score_buckets=score_buckets,
        feature_by_split=feature_by_split,
        prediction_drift=prediction_drift,
        sector_contribution=sector_contribution,
        paper_explanations=paper_explanations,
        min_score=args.min_score,
    )
    print(f"wrote model visibility artifacts with prefix {output_prefix}")


def _attach_reference_columns(
    predictions: pd.DataFrame,
    research_frame: pd.DataFrame,
) -> pd.DataFrame:
    reference_columns = [
        "date",
        "symbol",
        "sector",
        "industry",
        "market_return_20d",
        "market_return_60d",
        "volatility_20d",
        "volatility_60d",
        "log_avg_dollar_volume_20d",
        "sector_rank_volatility_20d",
    ]
    reference = research_frame.loc[
        :,
        [column for column in reference_columns if column in research_frame.columns],
    ].drop_duplicates(["date", "symbol"])
    return predictions.merge(reference, on=["date", "symbol"], how="left")


def _score_bucket_report(
    predictions: pd.DataFrame,
    *,
    score_col: str,
    target_col: str,
    realized_return_col: str,
) -> pd.DataFrame:
    required = {"split_id", score_col, target_col, realized_return_col}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"predictions are missing required columns: {sorted(missing)}")

    rows = []
    for split_id, split in predictions.dropna(
        subset=[score_col, target_col, realized_return_col]
    ).groupby("split_id", sort=True):
        scored = split.copy()
        scored["score_bucket"] = _quantile_bucket(scored[score_col], buckets=10)
        for bucket, bucket_frame in scored.groupby("score_bucket", sort=True):
            rows.append(
                {
                    "split_id": split_id,
                    "score_bucket": int(bucket),
                    "bucket_label": "highest" if bucket == 10 else "lowest" if bucket == 1 else "",
                    "rows": len(bucket_frame),
                    "score_min": bucket_frame[score_col].min(),
                    "score_mean": bucket_frame[score_col].mean(),
                    "score_max": bucket_frame[score_col].max(),
                    "mean_target": bucket_frame[target_col].mean(),
                    "mean_realized_return": bucket_frame[realized_return_col].mean(),
                    "win_rate": (bucket_frame[realized_return_col] > 0).mean(),
                    "ic_spearman": information_coefficient(
                        bucket_frame[score_col],
                        bucket_frame[target_col],
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["split_id", "score_bucket"])


def _quantile_bucket(values: pd.Series, *, buckets: int) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=buckets, labels=False, duplicates="drop").astype(int).add(1)


def _feature_importance_by_split(
    research_frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    args: argparse.Namespace,
) -> pd.DataFrame:
    model_spec = _model_spec(args.model)
    config = BaselineModelConfig(
        target_col=args.target,
        realized_return_col=args.realized_return,
        first_train_start=args.first_train_start,
        first_validation_start=args.first_validation_start,
        validation_months=args.validation_months,
        purge_days=args.horizon,
        model_type=model_spec.model_type,
        alpha=model_spec.alpha,
        n_estimators=model_spec.n_estimators,
        max_depth=model_spec.max_depth,
        min_samples_leaf=model_spec.min_samples_leaf,
        n_jobs=args.n_jobs,
    )
    frame = research_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=[*feature_columns, args.target, args.realized_return])
    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    rows = []
    for split_id, split in enumerate(
        walk_forward_splits(
            frame,
            first_train_start=config.first_train_start,
            first_validation_start=config.first_validation_start,
            validation_months=config.validation_months,
            purge_days=config.purge_days,
            embargo_days=config.embargo_days,
        ),
        start=1,
    ):
        train = frame.loc[split.train_index]
        validation = frame.loc[split.validation_index]
        if train.empty or validation.empty:
            continue
        model = _build_model(config)
        model.fit(train.loc[:, feature_columns], train[args.target])
        importance = _feature_importance(model, feature_columns)
        if importance is None:
            continue
        importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
        importance["split_id"] = split_id
        importance["rank"] = importance.index + 1
        importance["train_start"] = split.train_start
        importance["train_end"] = split.train_end
        importance["validation_start"] = split.validation_start
        importance["validation_end"] = split.validation_end
        rows.append(importance)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).loc[
        :,
        [
            "split_id",
            "rank",
            "feature",
            "importance",
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
        ],
    ]


def _model_spec(name: str):
    for spec in FAST_MODEL_SPECS:
        if spec.name == name:
            return spec
    raise ValueError(f"unsupported model: {name!r}")


def _prediction_drift_report(
    predictions: pd.DataFrame,
    *,
    score_col: str,
    min_score: float,
) -> pd.DataFrame:
    frame = predictions.dropna(subset=[score_col]).copy()
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    return (
        frame.groupby("month", as_index=False)
        .agg(
            rows=(score_col, "size"),
            symbols=("symbol", "nunique"),
            score_mean=(score_col, "mean"),
            score_std=(score_col, "std"),
            score_p10=(score_col, lambda values: values.quantile(0.10)),
            score_p50=(score_col, "median"),
            score_p90=(score_col, lambda values: values.quantile(0.90)),
            above_threshold=(score_col, lambda values: int((values >= min_score).sum())),
            market_return_20d=("market_return_20d", "mean"),
            market_return_60d=("market_return_60d", "mean"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )


def _sector_contribution_report(
    ledger: pd.DataFrame,
    research_frame: pd.DataFrame,
) -> pd.DataFrame:
    filled = ledger.loc[ledger["status"] == "filled"].copy()
    if filled.empty:
        return pd.DataFrame()
    reference_columns = [
        column for column in ("date", "symbol", "sector", "industry") if column in research_frame
    ]
    reference = research_frame.loc[:, reference_columns].drop_duplicates(["date", "symbol"])
    filled = filled.merge(
        reference,
        left_on=["signal_date", "symbol"],
        right_on=["date", "symbol"],
        how="left",
    )
    return (
        filled.groupby("sector", dropna=False, as_index=False)
        .agg(
            trades=("symbol", "size"),
            symbols=("symbol", "nunique"),
            net_pnl=("net_pnl", "sum"),
            avg_net_pnl=("net_pnl", "mean"),
            win_rate=("net_pnl", lambda values: (values > 0).mean()),
            gross_entry_value=("entry_value", "sum"),
        )
        .sort_values("net_pnl", ascending=False)
        .reset_index(drop=True)
    )


def _paper_explanation_report(
    paper_ledger: pd.DataFrame,
    paper_watchlist: pd.DataFrame,
    feature_by_split: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    planned = paper_ledger.loc[paper_ledger["status"] == "planned"].copy()
    if planned.empty:
        return pd.DataFrame()
    top_features = _stable_top_features(feature_by_split, limit=5)
    explanation_columns = [
        "symbol",
        "score",
        "sector",
        "reference_price",
        "shares",
        "estimated_entry_value",
        "estimated_required_cash",
    ]
    merged = planned.loc[:, explanation_columns].merge(
        paper_watchlist.loc[:, ["symbol", *feature_columns]],
        on="symbol",
        how="left",
    )
    percentile_frame = paper_watchlist.loc[:, ["symbol", *feature_columns]].copy()
    for feature in feature_columns:
        percentile_frame[f"{feature}_percentile"] = percentile_frame[feature].rank(pct=True)
    merged = merged.merge(
        percentile_frame.loc[
            :,
            ["symbol", *[f"{feature}_percentile" for feature in top_features]],
        ],
        on="symbol",
        how="left",
    )
    merged["top_model_features"] = ", ".join(top_features)
    for feature in top_features:
        merged[f"{feature}_value"] = merged[feature]
    ordered_columns = [
        *explanation_columns,
        "top_model_features",
        *[f"{feature}_value" for feature in top_features],
        *[f"{feature}_percentile" for feature in top_features],
    ]
    return merged.loc[:, ordered_columns].sort_values("score", ascending=False)


def _stable_top_features(feature_by_split: pd.DataFrame, *, limit: int) -> list[str]:
    if feature_by_split.empty:
        return []
    return (
        feature_by_split.groupby("feature", as_index=False)
        .agg(mean_importance=("importance", "mean"), median_rank=("rank", "median"))
        .sort_values(["mean_importance", "median_rank"], ascending=[False, True])
        .head(limit)["feature"]
        .tolist()
    )


def _write_csv(frame: pd.DataFrame, output_prefix: Path, suffix: str) -> None:
    frame.to_csv(output_prefix.with_name(output_prefix.name + f"_{suffix}.csv"), index=False)


def _write_markdown_summary(
    output_prefix: Path,
    *,
    score_buckets: pd.DataFrame,
    feature_by_split: pd.DataFrame,
    prediction_drift: pd.DataFrame,
    sector_contribution: pd.DataFrame,
    paper_explanations: pd.DataFrame,
    min_score: float,
) -> None:
    bucket_summary = (
        score_buckets.groupby("score_bucket", as_index=False)
        .agg(
            mean_realized_return=("mean_realized_return", "mean"),
            win_rate=("win_rate", "mean"),
            rows=("rows", "sum"),
        )
        .sort_values("score_bucket", ascending=False)
    )
    top_features = _stable_top_features(feature_by_split, limit=5)
    recent_drift = prediction_drift.tail(6)
    lines = [
        "# SignalForge Model Visibility",
        "",
        f"Minimum paper-trade score threshold: `{min_score}`.",
        "",
        "## Stable Top Features",
        "",
        *[f"- `{feature}`" for feature in top_features],
        "",
        "## Score Bucket Check",
        "",
        _markdown_table(bucket_summary.head(5)),
        "",
        "## Recent Prediction Drift",
        "",
        _markdown_table(recent_drift),
        "",
        "## Sector Contribution",
        "",
        _markdown_table(sector_contribution.head(10)),
        "",
        "## Current Paper Picks",
        "",
        _markdown_table(paper_explanations.head(15)),
        "",
    ]
    output_prefix.with_name(output_prefix.name + "_summary.md").write_text("\n".join(lines))


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    rows = [list(display.columns), ["---"] * len(display.columns)]
    rows.extend(display.astype(str).values.tolist())
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


if __name__ == "__main__":
    main()
