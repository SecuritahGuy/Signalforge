from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from signalforge.backtest import BacktestConfig, event_based_long_only_backtest
from signalforge.metrics import max_drawdown, sharpe_ratio


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run strict event-ledger sensitivity for $2k risk controls."
    )
    parser.add_argument("--predictions", default="reports/exec_top_experiment_predictions.csv")
    parser.add_argument("--research-frame", default="data/processed/research_frame.csv")
    parser.add_argument("--exit-price-col", default="exit_close_20d")
    parser.add_argument("--output", default="reports/event_risk_sensitivity.csv")
    parser.add_argument("--initial-capital", type=float, default=2_000.0)
    parser.add_argument("--position-weights", default="0.10,0.15,0.20")
    parser.add_argument("--long-fractions", default="0.03,0.05,0.10")
    parser.add_argument("--min-scores", default="0.005,0.01,0.02")
    parser.add_argument("--drawdown-stops", default="none,0.15,0.20")
    parser.add_argument("--cooldown-days", type=int, default=20)
    parser.add_argument("--rebalance-interval-days", type=int, default=20)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument(
        "--regimes",
        default="none,market_20d_positive,market_60d_positive,market_20d_and_60d_positive",
    )
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    signals = predictions.rename(
        columns={"prediction": "score", args.exit_price_col: "exit_close"}
    )
    signals = _attach_regime_columns(signals, args.research_frame)

    rows = []
    for position_weight, long_fraction, min_score, drawdown_stop, regime in product(
        _parse_float_list(args.position_weights),
        _parse_float_list(args.long_fractions),
        _parse_nullable_float_list(args.min_scores),
        _parse_nullable_float_list(args.drawdown_stops),
        _parse_str_list(args.regimes),
    ):
        regime_signals = _apply_regime(signals, regime)
        config = BacktestConfig(
            long_fraction=long_fraction,
            max_position_weight=position_weight,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            max_drawdown_stop=drawdown_stop,
            cooldown_days=args.cooldown_days,
            initial_capital=args.initial_capital,
            rebalance_interval_days=args.rebalance_interval_days,
            min_score=min_score,
        )
        equity, ledger = event_based_long_only_backtest(
            regime_signals,
            exit_price_col="exit_close",
            config=config,
        )
        rows.append(
            _summary_row(
                equity,
                ledger,
                initial_capital=args.initial_capital,
                position_weight=position_weight,
                long_fraction=long_fraction,
                min_score=min_score,
                drawdown_stop=drawdown_stop,
                regime=regime,
            )
        )

    output = pd.DataFrame(rows).sort_values(
        ["max_drawdown", "total_return"],
        ascending=[False, False],
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"wrote {len(output):,} event risk rows to {output_path}")


def _attach_regime_columns(signals: pd.DataFrame, research_frame_path: str) -> pd.DataFrame:
    research_frame = pd.read_csv(
        research_frame_path,
        usecols=["date", "market_return_20d", "market_return_60d"],
    )
    regimes = research_frame.drop_duplicates("date")
    output = signals.copy()
    output["date"] = pd.to_datetime(output["date"])
    regimes["date"] = pd.to_datetime(regimes["date"])
    return output.merge(regimes, on="date", how="left")


def _apply_regime(signals: pd.DataFrame, regime: str) -> pd.DataFrame:
    if regime == "none":
        return signals
    if regime == "market_20d_positive":
        return signals.loc[signals["market_return_20d"] > 0]
    if regime == "market_60d_positive":
        return signals.loc[signals["market_return_60d"] > 0]
    if regime == "market_20d_and_60d_positive":
        return signals.loc[
            (signals["market_return_20d"] > 0) & (signals["market_return_60d"] > 0)
        ]
    raise ValueError(f"unsupported regime: {regime}")


def _summary_row(
    equity: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    initial_capital: float,
    position_weight: float,
    long_fraction: float,
    min_score: float | None,
    drawdown_stop: float | None,
    regime: str,
) -> dict:
    filled = ledger.loc[ledger["status"] == "filled"] if not ledger.empty else ledger
    skipped = ledger.loc[ledger["status"] == "skipped"] if not ledger.empty else ledger
    end_capital = equity["capital"].iloc[-1] if not equity.empty else initial_capital
    return {
        "position_weight": position_weight,
        "long_fraction": long_fraction,
        "min_score": "none" if min_score is None else min_score,
        "drawdown_stop": "none" if drawdown_stop is None else drawdown_stop,
        "regime": regime,
        "end_capital": end_capital,
        "total_return": end_capital / initial_capital - 1.0,
        "sharpe": sharpe_ratio(equity["net_return"]) if not equity.empty else 0.0,
        "max_drawdown": max_drawdown(equity["net_return"]) if not equity.empty else 0.0,
        "filled_trades": len(filled),
        "skipped_trades": len(skipped),
        "win_rate": (filled["net_pnl"] > 0).mean() if not filled.empty else 0.0,
        "avg_net_pnl": filled["net_pnl"].mean() if not filled.empty else 0.0,
        "total_net_pnl": filled["net_pnl"].sum() if not filled.empty else 0.0,
    }


def _parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("list argument must include at least one value")
    return values


def _parse_nullable_float_list(raw: str) -> list[float | None]:
    values: list[float | None] = []
    for item in raw.split(","):
        value = item.strip().lower()
        if not value:
            continue
        values.append(None if value in {"none", "null"} else float(value))
    if not values:
        raise ValueError("list argument must include at least one value")
    return values


def _parse_str_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("list argument must include at least one value")
    return values


if __name__ == "__main__":
    main()
