from __future__ import annotations

import argparse
from pathlib import Path

from signalforge.data import load_universe_csv
from signalforge.providers.yahoo import download_yahoo_prices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download normalized daily prices from Yahoo Finance."
    )
    parser.add_argument("--universe", default="data/reference/tracked_universe.csv")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default="data/raw/yahoo_prices.csv")
    parser.add_argument("--include-benchmark", action="store_true", default=True)
    args = parser.parse_args()

    universe = load_universe_csv(args.universe)
    symbols = universe["symbol"].tolist()
    if not args.include_benchmark:
        symbols = universe.loc[universe["category"] != "benchmark", "symbol"].tolist()

    prices = download_yahoo_prices(symbols, start=args.start, end=args.end)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path, index=False)
    print(f"wrote {len(prices):,} rows for {prices['symbol'].nunique()} symbols to {output_path}")


if __name__ == "__main__":
    main()
