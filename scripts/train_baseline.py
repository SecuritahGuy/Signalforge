from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from signalforge.modeling import BaselineModelConfig, train_baseline_walkforward


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the first walk-forward baseline model.")
    parser.add_argument("--research-frame", default="data/processed/research_frame.csv")
    parser.add_argument("--predictions-output", default="data/processed/model_predictions.csv")
    parser.add_argument("--summary-output", default="reports/baseline_walkforward_summary.csv")
    parser.add_argument("--metadata-output", default="reports/baseline_model_metadata.json")
    parser.add_argument("--target", default="fwd_5d_excess_return")
    parser.add_argument("--realized-return", default="fwd_5d_return")
    parser.add_argument("--benchmark-score", default="momentum_20d")
    parser.add_argument("--first-train-start", default="2020-01-01")
    parser.add_argument("--first-validation-start", default="2022-01-01")
    parser.add_argument("--validation-months", type=int, default=6)
    parser.add_argument("--purge-days", type=int, default=5)
    parser.add_argument("--embargo-days", type=int, default=0)
    parser.add_argument(
        "--model-type",
        choices=["ridge", "elasticnet", "random_forest"],
        default="ridge",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--feature-importance-output", default=None)
    args = parser.parse_args()

    research_frame = pd.read_csv(args.research_frame)
    config = BaselineModelConfig(
        target_col=args.target,
        realized_return_col=args.realized_return,
        benchmark_score_col=args.benchmark_score,
        first_train_start=args.first_train_start,
        first_validation_start=args.first_validation_start,
        validation_months=args.validation_months,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
        model_type=args.model_type,
        alpha=args.alpha,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    predictions, summary, metadata = train_baseline_walkforward(research_frame, config=config)

    _write_csv(predictions, args.predictions_output)
    _write_csv(summary, args.summary_output)
    _write_json(metadata, args.metadata_output)
    if args.feature_importance_output and "feature_importance" in metadata:
        _write_csv(pd.DataFrame(metadata["feature_importance"]), args.feature_importance_output)
    print(
        "wrote "
        f"{len(predictions):,} predictions, {len(summary):,} split summaries, "
        f"and metadata for {metadata['symbol_count']} symbols"
    )


def _write_csv(frame: pd.DataFrame, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def _write_json(payload: dict, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
