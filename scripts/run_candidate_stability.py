from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from signalforge.backtest import BacktestConfig, event_based_long_only_backtest
from signalforge.metrics import max_drawdown, sharpe_ratio


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a selected executable candidate by split and stress settings."
    )
    parser.add_argument(
        "--predictions",
        default="reports/exec_top_experiment_weight10_score001_no_stop_predictions.csv",
    )
    parser.add_argument("--exit-price-col", default="exit_close_20d")
    parser.add_argument("--output-prefix", default="reports/exec_candidate_stability")
    parser.add_argument("--initial-capital", type=float, default=2_000.0)
    parser.add_argument("--position-weight", type=float, default=0.10)
    parser.add_argument("--long-fraction", type=float, default=0.10)
    parser.add_argument("--min-score", type=float, default=0.01)
    parser.add_argument("--rebalance-interval-days", type=int, default=20)
    parser.add_argument("--cost-bps", default="5,10,25,50")
    parser.add_argument("--slippage-bps", default="5,10,25,50")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    signals = predictions.rename(
        columns={"prediction": "score", args.exit_price_col: "exit_close"}
    )

    split_rows = [
        _run_one(
            split_frame,
            scenario=f"split_{int(split_id)}",
            initial_capital=args.initial_capital,
            position_weight=args.position_weight,
            long_fraction=args.long_fraction,
            min_score=args.min_score,
            rebalance_interval_days=args.rebalance_interval_days,
            transaction_cost_bps=5.0,
            slippage_bps=5.0,
        )
        for split_id, split_frame in signals.groupby("split_id", sort=True)
    ]

    stress_rows = []
    for cost_bps in _parse_float_list(args.cost_bps):
        for slippage_bps in _parse_float_list(args.slippage_bps):
            stress_rows.append(
                _run_one(
                    signals,
                    scenario=f"cost_{cost_bps:g}_slippage_{slippage_bps:g}",
                    initial_capital=args.initial_capital,
                    position_weight=args.position_weight,
                    long_fraction=args.long_fraction,
                    min_score=args.min_score,
                    rebalance_interval_days=args.rebalance_interval_days,
                    transaction_cost_bps=cost_bps,
                    slippage_bps=slippage_bps,
                )
            )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(split_rows).to_csv(
        output_prefix.with_name(output_prefix.name + "_by_split.csv"),
        index=False,
    )
    pd.DataFrame(stress_rows).sort_values(
        ["transaction_cost_bps", "slippage_bps"]
    ).to_csv(
        output_prefix.with_name(output_prefix.name + "_cost_stress.csv"),
        index=False,
    )
    print(f"wrote candidate stability artifacts with prefix {output_prefix}")


def _run_one(
    signals: pd.DataFrame,
    *,
    scenario: str,
    initial_capital: float,
    position_weight: float,
    long_fraction: float,
    min_score: float,
    rebalance_interval_days: int,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> dict:
    config = BacktestConfig(
        long_fraction=long_fraction,
        max_position_weight=position_weight,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        initial_capital=initial_capital,
        rebalance_interval_days=rebalance_interval_days,
        min_score=min_score,
    )
    equity, ledger = event_based_long_only_backtest(
        signals,
        exit_price_col="exit_close",
        config=config,
    )
    filled = ledger.loc[ledger["status"] == "filled"] if not ledger.empty else ledger
    end_capital = equity["capital"].iloc[-1] if not equity.empty else initial_capital
    return {
        "scenario": scenario,
        "split_id": _single_value(signals, "split_id"),
        "validation_start": _single_value(signals, "validation_start"),
        "validation_end": _single_value(signals, "validation_end"),
        "transaction_cost_bps": transaction_cost_bps,
        "slippage_bps": slippage_bps,
        "end_capital": end_capital,
        "total_return": end_capital / initial_capital - 1.0,
        "sharpe": sharpe_ratio(equity["net_return"]) if not equity.empty else 0.0,
        "max_drawdown": max_drawdown(equity["net_return"]) if not equity.empty else 0.0,
        "filled_trades": len(filled),
        "win_rate": (filled["net_pnl"] > 0).mean() if not filled.empty else 0.0,
        "avg_net_pnl": filled["net_pnl"].mean() if not filled.empty else 0.0,
        "total_net_pnl": filled["net_pnl"].sum() if not filled.empty else 0.0,
    }


def _single_value(frame: pd.DataFrame, column: str) -> object:
    if column not in frame.columns:
        return pd.NA
    values = frame[column].dropna().unique()
    return values[0] if len(values) == 1 else pd.NA


def _parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("list argument must include at least one value")
    return values


if __name__ == "__main__":
    main()
