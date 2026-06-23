from __future__ import annotations

import argparse
from pathlib import Path

from signalforge.universe import BroadUniverseConfig, build_broad_universe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a generated broad universe CSV for stock discovery."
    )
    parser.add_argument("--source", choices=("sp500", "us_listed"), default="sp500")
    parser.add_argument("--output", default="data/reference/sp500_universe.csv")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    universe = build_broad_universe(
        BroadUniverseConfig(
            source=args.source,
            include_benchmark=not args.no_benchmark,
            benchmark_symbol=args.benchmark,
            limit=args.limit,
        )
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(output_path, index=False)
    print(
        f"wrote {len(universe):,} rows for {universe['symbol'].nunique():,} symbols "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
