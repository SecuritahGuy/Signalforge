from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from signalforge.backtest import BacktestConfig
from signalforge.experiments import (
    DEFAULT_MODEL_SPECS,
    FAST_FEATURE_SETS,
    FAST_MODEL_SPECS,
    run_experiment_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SignalForge baseline experiment grid.")
    parser.add_argument("--research-frame", default="data/processed/research_frame.csv")
    parser.add_argument("--horizons", default="5,20")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--target-kind",
        choices=("close_excess", "exec_excess"),
        default="close_excess",
    )
    parser.add_argument("--first-train-start", default="2020-01-01")
    parser.add_argument("--first-validation-start", default="2022-01-01")
    parser.add_argument("--validation-months", type=int, default=6)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--target-volatility", type=float, default=None)
    parser.add_argument("--volatility-lookback", type=int, default=20)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--max-drawdown-stop", type=float, default=None)
    parser.add_argument("--cooldown-days", type=int, default=20)
    parser.add_argument("--max-symbol-trades", type=int, default=None)
    parser.add_argument("--leaderboard-output", default="reports/experiment_leaderboard.csv")
    parser.add_argument("--split-summary-output", default="reports/experiment_split_summaries.csv")
    args = parser.parse_args()

    research_frame = pd.read_csv(args.research_frame)
    horizons = (20,) if args.fast else _parse_horizons(args.horizons)
    leaderboard, split_summaries = run_experiment_grid(
        research_frame,
        horizons=horizons,
        feature_sets=FAST_FEATURE_SETS if args.fast else None,
        model_specs=FAST_MODEL_SPECS if args.fast else DEFAULT_MODEL_SPECS,
        first_train_start=args.first_train_start,
        first_validation_start=args.first_validation_start,
        validation_months=args.validation_months,
        n_jobs=args.n_jobs,
        target_kind=args.target_kind,
        backtest_config=BacktestConfig(
            target_volatility=args.target_volatility,
            volatility_lookback=args.volatility_lookback,
            max_leverage=args.max_leverage,
            max_drawdown_stop=args.max_drawdown_stop,
            cooldown_days=args.cooldown_days,
            max_symbol_trades=args.max_symbol_trades,
        ),
    )

    _write_csv(leaderboard, args.leaderboard_output)
    if split_summaries:
        all_split_summaries = pd.concat(split_summaries.values(), ignore_index=True)
        _write_csv(all_split_summaries, args.split_summary_output)
    print(f"wrote {len(leaderboard):,} experiment rows to {args.leaderboard_output}")


def _parse_horizons(value: str) -> tuple[int, ...]:
    horizons = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not horizons:
        raise ValueError("--horizons must include at least one integer")
    return horizons


def _write_csv(frame: pd.DataFrame, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
