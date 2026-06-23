from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from signalforge.discovery import DiscoveryConfig, run_stock_discovery
from signalforge.discovery_report import write_discovery_outputs
from signalforge.run_manifest import write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate explainable stock discovery watchlists from a broad feature frame."
    )
    parser.add_argument("--research-frame", default="data/processed/research_frame.csv")
    parser.add_argument("--universe", default=None)
    parser.add_argument("--existing-watchlist", default=None)
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--output-dir", default="reports/discovery")
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-avg-dollar-volume-20d", type=float, default=5_000_000.0)
    parser.add_argument("--min-market-cap", type=float, default=300_000_000.0)
    parser.add_argument("--no-market-cap-filter", action="store_true")
    parser.add_argument("--earnings-blackout-days", type=int, default=1)
    args = parser.parse_args()

    research_frame = pd.read_csv(args.research_frame)
    universe = _read_optional_csv(args.universe)
    existing_watchlist = _read_optional_csv(args.existing_watchlist)
    config = DiscoveryConfig(
        top_n=args.top_n,
        min_price=args.min_price,
        min_avg_dollar_volume_20d=args.min_avg_dollar_volume_20d,
        min_market_cap=None if args.no_market_cap_filter else args.min_market_cap,
        earnings_blackout_days=args.earnings_blackout_days,
    )

    try:
        result = run_stock_discovery(
            research_frame,
            universe=universe,
            as_of_date=args.as_of_date,
            existing_watchlist=existing_watchlist,
            config=config,
        )
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = Path(args.output_dir)
    artifacts = write_discovery_outputs(result, output_dir, source_universe=universe)
    write_run_manifest(
        output_dir,
        run_type="discovery",
        as_of_date=result.as_of_date.date().isoformat(),
        parameters=_manifest_parameters(args),
        inputs={
            "research_frame": args.research_frame,
            "universe": args.universe,
            "watchlist": args.existing_watchlist,
        },
        outputs=artifacts,
        code_cwd=Path.cwd(),
    )
    print(f"wrote discovery artifacts to {output_dir}")


def _read_optional_csv(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return pd.read_csv(path)


def _manifest_parameters(args: argparse.Namespace) -> dict:
    return {
        "top_n": args.top_n,
        "top_n_per_lane": args.top_n,
        "as_of_date": args.as_of_date,
        "min_price": args.min_price,
        "min_avg_dollar_volume_20d": args.min_avg_dollar_volume_20d,
        "min_market_cap": None if args.no_market_cap_filter else args.min_market_cap,
        "market_cap_min": None if args.no_market_cap_filter else args.min_market_cap,
        "market_cap_max": None,
        "no_market_cap_filter": args.no_market_cap_filter,
        "earnings_blackout_days": args.earnings_blackout_days,
        "output_dir": args.output_dir,
    }


if __name__ == "__main__":
    main()
