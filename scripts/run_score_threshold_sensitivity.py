from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from signalforge.backtest import BacktestConfig, event_based_long_only_backtest
from signalforge.metrics import max_drawdown, sharpe_ratio


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run strict $2k event-ledger sensitivity across model score thresholds."
    )
    parser.add_argument("--predictions", default="reports/top_experiment_predictions.csv")
    parser.add_argument("--exit-price-col", default="exit_close_20d")
    parser.add_argument("--output", default="reports/score_threshold_sensitivity.csv")
    parser.add_argument("--initial-capital", type=float, default=2_000.0)
    parser.add_argument("--position-weight", type=float, default=0.20)
    parser.add_argument("--long-fraction", type=float, default=0.10)
    parser.add_argument("--rebalance-interval-days", type=int, default=20)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--max-symbol-trades", type=int, default=None)
    parser.add_argument("--allow-fractional-shares", action="store_true")
    parser.add_argument("--thresholds", default="none,0,0.001,0.002,0.005,0.01,0.02")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    signals = predictions.rename(
        columns={"prediction": "score", args.exit_price_col: "exit_close"}
    )
    rows = []
    for threshold in _parse_thresholds(args.thresholds):
        config = BacktestConfig(
            long_fraction=args.long_fraction,
            max_position_weight=args.position_weight,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            max_symbol_trades=args.max_symbol_trades,
            initial_capital=args.initial_capital,
            allow_fractional_shares=args.allow_fractional_shares,
            rebalance_interval_days=args.rebalance_interval_days,
            min_score=threshold,
        )
        equity, ledger = event_based_long_only_backtest(
            signals,
            exit_price_col="exit_close",
            config=config,
        )
        rows.append(_summary_row(threshold, equity, ledger, initial_capital=args.initial_capital))

    output = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"wrote {len(output):,} threshold rows to {output_path}")


def _parse_thresholds(raw: str) -> list[float | None]:
    thresholds: list[float | None] = []
    for item in raw.split(","):
        value = item.strip().lower()
        if not value:
            continue
        thresholds.append(None if value in {"none", "null"} else float(value))
    if not thresholds:
        raise ValueError("--thresholds must include at least one threshold")
    return thresholds


def _summary_row(
    threshold: float | None,
    equity: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    initial_capital: float,
) -> dict:
    filled = ledger.loc[ledger["status"] == "filled"] if not ledger.empty else ledger
    skips = ledger.loc[ledger["status"] == "skipped"] if not ledger.empty else ledger
    end_capital = equity["capital"].iloc[-1] if not equity.empty else initial_capital
    skip_counts = skips["skip_reason"].value_counts().to_dict() if not skips.empty else {}
    return {
        "min_score": "none" if threshold is None else threshold,
        "end_capital": end_capital,
        "total_return": end_capital / initial_capital - 1.0,
        "sharpe": sharpe_ratio(equity["net_return"]) if not equity.empty else 0.0,
        "max_drawdown": max_drawdown(equity["net_return"]) if not equity.empty else 0.0,
        "filled_trades": len(filled),
        "skipped_trades": len(skips),
        "win_rate": (filled["net_pnl"] > 0).mean() if not filled.empty else 0.0,
        "avg_net_pnl": filled["net_pnl"].mean() if not filled.empty else 0.0,
        "score_below_threshold": skip_counts.get("score_below_threshold", 0),
        "size_too_small": skip_counts.get("size_too_small", 0),
        "insufficient_cash": skip_counts.get("insufficient_cash", 0),
        "symbol_trade_cap": skip_counts.get("symbol_trade_cap", 0),
    }


if __name__ == "__main__":
    main()
