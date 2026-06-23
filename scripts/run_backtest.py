from __future__ import annotations

import argparse

import pandas as pd

from signalforge.discovery import DiscoveryConfig
from signalforge.discovery_backtest import (
    DiscoveryLaneBacktestConfig,
    run_discovery_lane_backtest,
    write_discovery_lane_backtest_outputs,
)
from signalforge.run_manifest import write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest SignalForge discovery lane selections over historical dates."
    )
    parser.add_argument("--research-frame", default="data/processed/research_frame.csv")
    parser.add_argument("--output", default="reports/discovery_backtest")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--rebalance", choices=("monthly", "weekly"), default="monthly")
    parser.add_argument("--top-n-per-lane", type=int, default=25)
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20, 60])
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--price-col", default="adj_close")
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-avg-dollar-volume-20d", type=float, default=5_000_000.0)
    parser.add_argument("--min-market-cap", type=float, default=300_000_000.0)
    parser.add_argument("--no-market-cap-filter", action="store_true")
    args = parser.parse_args()

    research_frame = pd.read_csv(args.research_frame)
    existing_watchlist = _read_optional_csv(args.exclude)
    discovery_config = DiscoveryConfig(
        top_n=args.top_n_per_lane,
        price_col=args.price_col,
        min_price=args.min_price,
        min_avg_dollar_volume_20d=args.min_avg_dollar_volume_20d,
        min_market_cap=None if args.no_market_cap_filter else args.min_market_cap,
    )
    backtest_config = DiscoveryLaneBacktestConfig(
        rebalance=args.rebalance,
        top_n_per_lane=args.top_n_per_lane,
        horizons=tuple(args.horizons),
        price_col=args.price_col,
        discovery_config=discovery_config,
    )

    try:
        result = run_discovery_lane_backtest(
            research_frame,
            start_date=args.start_date,
            end_date=args.end_date,
            existing_watchlist=existing_watchlist,
            config=backtest_config,
        )
    except ValueError as exc:
        parser.error(str(exc))

    artifacts = write_discovery_lane_backtest_outputs(result, args.output)
    write_run_manifest(
        args.output,
        run_type="backtest",
        start_date=args.start_date,
        end_date=args.end_date,
        parameters=_manifest_parameters(args),
        inputs={
            "research_frame": args.research_frame,
            "watchlist": args.exclude,
        },
        outputs=artifacts,
        code_cwd=".",
    )
    print(f"wrote discovery lane backtest artifacts to {args.output}")


def _read_optional_csv(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return pd.read_csv(path)


def _manifest_parameters(args: argparse.Namespace) -> dict:
    return {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "rebalance": args.rebalance,
        "top_n_per_lane": args.top_n_per_lane,
        "horizons": args.horizons,
        "exclude": args.exclude,
        "price_col": args.price_col,
        "min_price": args.min_price,
        "min_avg_dollar_volume_20d": args.min_avg_dollar_volume_20d,
        "min_market_cap": None if args.no_market_cap_filter else args.min_market_cap,
        "market_cap_min": None if args.no_market_cap_filter else args.min_market_cap,
        "market_cap_max": None,
        "no_market_cap_filter": args.no_market_cap_filter,
        "output": args.output,
    }


if __name__ == "__main__":
    main()
