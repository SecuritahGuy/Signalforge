from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from signalforge.data import load_price_csv
from signalforge.paper import PaperTradingConfig
from signalforge.paper_backtest import run_paper_style_backtest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a historical backtest through the paper-trading ledger lifecycle."
    )
    parser.add_argument(
        "--signals",
        default="reports/exec_top_experiment_weight10_score001_no_stop_predictions.csv",
    )
    parser.add_argument("--prices", default="data/raw/yahoo_prices.csv")
    parser.add_argument("--output-prefix", default="reports/paper_style_backtest")
    parser.add_argument("--score-col", default="prediction")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--initial-capital", type=float, default=2_000.0)
    parser.add_argument("--position-weight", type=float, default=0.10)
    parser.add_argument("--long-fraction", type=float, default=0.10)
    parser.add_argument("--min-score", type=float, default=0.01)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--allow-fractional-shares", action="store_true")
    args = parser.parse_args()

    signals = pd.read_csv(args.signals)
    prices = load_price_csv(args.prices)
    config = PaperTradingConfig(
        initial_capital=args.initial_capital,
        position_weight=args.position_weight,
        long_fraction=args.long_fraction,
        min_score=args.min_score,
        horizon=args.horizon,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        allow_fractional_shares=args.allow_fractional_shares,
    )
    result = run_paper_style_backtest(
        signals,
        prices,
        config=config,
        score_col=args.score_col,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    result.ledger.to_csv(output_prefix.with_name(output_prefix.name + "_ledger.csv"), index=False)
    result.daily_equity.to_csv(
        output_prefix.with_name(output_prefix.name + "_daily_equity.csv"),
        index=False,
    )
    output_prefix.with_name(output_prefix.name + "_summary.json").write_text(
        json.dumps(result.summary, indent=2, default=str) + "\n"
    )
    output_prefix.with_name(output_prefix.name + "_report.md").write_text(
        _markdown_report(result.summary)
    )
    print(f"wrote paper-style backtest artifacts with prefix {output_prefix}")


def _markdown_report(summary: dict) -> str:
    return "\n".join(
        [
            "# Paper-Style Backtest",
            "",
            f"- Initial capital: `${summary['initial_capital']:.2f}`",
            f"- Ending equity: `${summary['ending_equity']:.2f}`",
            f"- Total return: `{summary['total_return']:.2%}`",
            f"- Sharpe: `{summary['sharpe']:.4f}`",
            f"- Max drawdown: `{summary['max_drawdown']:.2%}`",
            f"- Filled positions: `{summary['filled_positions']}`",
            f"- Closed positions: `{summary['closed_positions']}`",
            f"- Open positions: `{summary['open_positions']}`",
            f"- Skipped orders: `{summary['skipped_orders']}`",
            f"- Closed win rate: `{summary['closed_win_rate']:.2%}`",
            f"- Avg closed PnL: `${summary['avg_closed_pnl']:.2f}`",
            "",
        ]
    )


if __name__ == "__main__":
    main()
