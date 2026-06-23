from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from signalforge.backtest import BacktestConfig, event_based_long_only_backtest
from signalforge.metrics import sharpe_ratio


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict event-ledger bankroll sensitivity.")
    parser.add_argument("--predictions", default="reports/top_experiment_predictions.csv")
    parser.add_argument("--exit-price-col", default="exit_close_20d")
    parser.add_argument("--capitals", default="2000,5000,10000,25000,50000,100000")
    parser.add_argument("--capital-position-weight", type=float, default=0.20)
    parser.add_argument("--capital-long-fraction", type=float, default=0.10)
    parser.add_argument("--max-symbol-trades", type=int, default=25)
    parser.add_argument("--rebalance-interval-days", type=int, default=20)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--max-adv-fraction", type=float, default=0.01)
    parser.add_argument("--allow-fractional-shares", action="store_true")
    parser.add_argument("--output", default="reports/bankroll_sensitivity.csv")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions).rename(
        columns={"prediction": "score", args.exit_price_col: "exit_close"}
    )
    rows = []
    ledgers = []
    for capital in _parse_capitals(args.capitals):
        config = BacktestConfig(
            long_fraction=args.capital_long_fraction,
            max_position_weight=args.capital_position_weight,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            max_symbol_trades=args.max_symbol_trades,
            initial_capital=capital,
            allow_fractional_shares=args.allow_fractional_shares,
            rebalance_interval_days=args.rebalance_interval_days,
            max_adv_fraction=args.max_adv_fraction,
        )
        equity, ledger = event_based_long_only_backtest(
            predictions,
            exit_price_col="exit_close",
            config=config,
        )
        rows.append(_summary_row(capital=capital, equity=equity, ledger=ledger))
        ledgers.append(ledger.assign(initial_capital=capital))

    sensitivity = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(output_path, index=False)

    ledger_path = output_path.with_name(output_path.stem + "_ledger.csv")
    pd.concat(ledgers, ignore_index=True).to_csv(ledger_path, index=False)
    print(f"wrote {len(sensitivity)} bankroll sensitivity rows to {output_path}")


def _parse_capitals(value: str) -> list[float]:
    capitals = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not capitals:
        raise ValueError("--capitals must include at least one value")
    return capitals


def _summary_row(
    *,
    capital: float,
    equity: pd.DataFrame,
    ledger: pd.DataFrame,
) -> dict:
    filled = ledger.loc[ledger["status"] == "filled"]
    skipped = ledger.loc[ledger["status"] == "skipped"]
    end_capital = equity["capital"].iloc[-1]
    return {
        "initial_capital": capital,
        "end_capital": end_capital,
        "total_return": end_capital / capital - 1.0,
        "sharpe": sharpe_ratio(equity["net_return"]),
        "max_drawdown": equity["drawdown"].min(),
        "trade_days": int((equity["positions"] > 0).sum()),
        "avg_positions": equity["positions"].mean(),
        "avg_invested": equity["invested"].mean(),
        "filled_trades": int(len(filled)),
        "skipped_trades": int(len(skipped)),
        "skip_size_too_small": int((skipped["skip_reason"] == "size_too_small").sum()),
        "skip_insufficient_cash": int((skipped["skip_reason"] == "insufficient_cash").sum()),
        "skip_symbol_cap": int((skipped["skip_reason"] == "symbol_trade_cap").sum()),
        "win_rate": float((filled["net_pnl"] > 0).mean()) if not filled.empty else pd.NA,
        "avg_trade_net_pnl": filled["net_pnl"].mean() if not filled.empty else pd.NA,
    }


if __name__ == "__main__":
    main()
