from __future__ import annotations

import argparse

from signalforge.workflow import WorkflowConfig, run_workflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the SignalForge discovery, lane backtest, and portfolio workflow."
    )
    parser.add_argument("--research-frame", required=True)
    parser.add_argument("--output-root", default="reports/workflow")
    parser.add_argument("--universe-source", default="sp500")
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--fundamentals", default=None)
    parser.add_argument("--backtest-trades", default=None)
    parser.add_argument("--rebalance", choices=("monthly", "weekly"), default="monthly")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20, 60])
    parser.add_argument("--top-n-per-lane", type=int, default=25)
    parser.add_argument("--max-positions", type=int, default=25)
    parser.add_argument("--starting-capital", type=float, default=100_000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--lanes", nargs="*", default=None)
    parser.add_argument("--position-sizing-method", default="equal_weight")
    parser.add_argument("--price-col", default="adj_close")
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-avg-dollar-volume-20d", type=float, default=5_000_000.0)
    parser.add_argument("--min-market-cap", type=float, default=300_000_000.0)
    parser.add_argument("--earnings-blackout-days", type=int, default=1)
    parser.add_argument("--no-market-cap-filter", action="store_true")
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--skip-lane-backtest", action="store_true")
    parser.add_argument("--skip-portfolio-backtest", action="store_true")
    args = parser.parse_args()

    config = WorkflowConfig(
        research_frame=args.research_frame,
        universe_source=args.universe_source,
        output_root=args.output_root,
        as_of_date=args.as_of_date,
        start_date=args.start_date,
        end_date=args.end_date,
        rebalance=args.rebalance,
        horizons=tuple(args.horizons),
        top_n_per_lane=args.top_n_per_lane,
        max_positions=args.max_positions,
        starting_capital=args.starting_capital,
        cost_bps=args.cost_bps,
        exclude_watchlist=args.exclude,
        no_market_cap_filter=args.no_market_cap_filter,
        fundamentals=args.fundamentals,
        run_discovery=not args.skip_discovery,
        run_lane_backtest=not args.skip_lane_backtest,
        run_portfolio_backtest=not args.skip_portfolio_backtest,
        backtest_trades=args.backtest_trades,
        selected_lanes=tuple(args.lanes or ()),
        position_sizing_method=args.position_sizing_method,
        price_col=args.price_col,
        min_price=args.min_price,
        min_avg_dollar_volume_20d=args.min_avg_dollar_volume_20d,
        min_market_cap=args.min_market_cap,
        earnings_blackout_days=args.earnings_blackout_days,
    )

    try:
        result = run_workflow(config)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"wrote workflow artifacts to {result.output_root}")


if __name__ == "__main__":
    main()
