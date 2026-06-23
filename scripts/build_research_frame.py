from __future__ import annotations

import argparse
from pathlib import Path

from signalforge.data import load_price_csv, load_universe_csv
from signalforge.research import build_research_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a processed model-ready research frame.")
    parser.add_argument("--prices", default="data/raw/yahoo_prices.csv")
    parser.add_argument("--universe", default="data/reference/tracked_universe.csv")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--horizons",
        default=None,
        help="Comma-separated label horizons. Overrides --horizon when provided.",
    )
    parser.add_argument("--output", default="data/processed/research_frame.csv")
    args = parser.parse_args()

    prices = load_price_csv(args.prices)
    universe = load_universe_csv(args.universe)
    research_frame = build_research_frame(
        prices,
        universe,
        benchmark_symbol=args.benchmark,
        horizon=args.horizon,
        horizons=_parse_horizons(args.horizons),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    research_frame.to_csv(output_path, index=False)
    row_count = len(research_frame)
    column_count = len(research_frame.columns)
    print(f"wrote {row_count:,} rows with {column_count} columns to {output_path}")


def _parse_horizons(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    horizons = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not horizons:
        raise ValueError("--horizons must include at least one integer")
    return horizons


if __name__ == "__main__":
    main()
