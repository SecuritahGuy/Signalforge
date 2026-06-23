from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SUMMARY_COLUMNS = [
    "ic_spearman",
    "directional_hit_rate",
    "backtest_mean_daily_return",
    "backtest_sharpe",
    "backtest_max_drawdown",
    "benchmark_ic_spearman",
    "benchmark_backtest_mean_daily_return",
    "benchmark_backtest_sharpe",
    "benchmark_backtest_max_drawdown",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare walk-forward model summary files.")
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Model name and CSV path as name=path. Repeat for each model.",
    )
    parser.add_argument("--output", default="reports/model_comparison.csv")
    args = parser.parse_args()

    rows = []
    for item in args.summary:
        model_name, path = _parse_summary_arg(item)
        summary = pd.read_csv(path)
        missing = set(SUMMARY_COLUMNS).difference(summary.columns)
        if missing:
            raise KeyError(f"{path} is missing columns: {sorted(missing)}")
        rows.append({"model": model_name, **summary[SUMMARY_COLUMNS].mean().to_dict()})

    comparison = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    print(f"wrote {len(comparison)} model rows to {output_path}")


def _parse_summary_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError("--summary must be formatted as name=path")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("--summary must include both name and path")
    return name, path


if __name__ == "__main__":
    main()
