from __future__ import annotations

import argparse

import pandas as pd

from signalforge.portfolio_backtest import (
    PortfolioBacktestConfig,
    run_portfolio_backtest,
    write_portfolio_backtest_outputs,
)
from signalforge.run_manifest import write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate a simple long-only portfolio from discovery backtest selections."
    )
    parser.add_argument("--backtest-trades", required=True)
    parser.add_argument("--research-frame", required=True)
    parser.add_argument("--output", default="reports/portfolio_backtest")
    parser.add_argument("--starting-capital", type=float, default=100_000.0)
    parser.add_argument("--rebalance", choices=("monthly", "weekly"), default="monthly")
    parser.add_argument("--max-positions", type=int, default=25)
    parser.add_argument("--cost-bps", type=float, default=0.0)
    parser.add_argument("--lanes", nargs="*", default=None)
    parser.add_argument("--position-sizing-method", default="equal_weight")
    parser.add_argument("--price-col", default="adj_close")
    args = parser.parse_args()

    backtest_trades = pd.read_csv(args.backtest_trades)
    research_frame = pd.read_csv(args.research_frame)
    config = PortfolioBacktestConfig(
        starting_capital=args.starting_capital,
        rebalance=args.rebalance,
        selected_lanes=tuple(args.lanes or ()),
        max_positions=args.max_positions,
        position_sizing_method=args.position_sizing_method,
        cost_bps=args.cost_bps,
        price_col=args.price_col,
    )

    try:
        result = run_portfolio_backtest(backtest_trades, research_frame, config=config)
    except ValueError as exc:
        parser.error(str(exc))

    artifacts = write_portfolio_backtest_outputs(result, args.output)
    write_run_manifest(
        args.output,
        run_type="portfolio_backtest",
        parameters=_manifest_parameters(args),
        inputs={
            "backtest_trades": args.backtest_trades,
            "research_frame": args.research_frame,
        },
        outputs=artifacts,
        code_cwd=".",
    )
    print(f"wrote portfolio backtest artifacts to {args.output}")


def _manifest_parameters(args: argparse.Namespace) -> dict:
    return {
        "starting_capital": args.starting_capital,
        "rebalance": args.rebalance,
        "max_positions": args.max_positions,
        "cost_bps": args.cost_bps,
        "selected_lanes": args.lanes or [],
        "position_sizing_method": args.position_sizing_method,
        "price_col": args.price_col,
        "output": args.output,
    }


if __name__ == "__main__":
    main()
