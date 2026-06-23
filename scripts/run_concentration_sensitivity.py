from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from signalforge.backtest import BacktestConfig, long_only_capital_backtest
from signalforge.diagnostics import daily_portfolio_diagnostics, symbol_contribution_diagnostics
from signalforge.metrics import max_drawdown, sharpe_ratio


def main() -> None:
    parser = argparse.ArgumentParser(description="Run symbol concentration sensitivity tests.")
    parser.add_argument("--predictions", default="reports/top_experiment_predictions.csv")
    parser.add_argument("--realized-return", default="fwd_20d_return")
    parser.add_argument("--rebalance-interval-days", type=int, default=20)
    parser.add_argument("--caps", default="none,300,200,125,75")
    parser.add_argument("--target-volatility", type=float, default=0.12)
    parser.add_argument("--volatility-lookback", type=int, default=20)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--max-drawdown-stop", type=float, default=0.12)
    parser.add_argument("--cooldown-days", type=int, default=20)
    parser.add_argument("--initial-capital", type=float, default=2_000.0)
    parser.add_argument("--capital-position-weight", type=float, default=0.20)
    parser.add_argument("--capital-long-fraction", type=float, default=0.10)
    parser.add_argument("--allow-fractional-shares", action="store_true")
    parser.add_argument("--output", default="reports/concentration_sensitivity.csv")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    rows = []
    for cap in _parse_caps(args.caps):
        config = BacktestConfig(
            target_volatility=args.target_volatility,
            volatility_lookback=args.volatility_lookback,
            max_leverage=args.max_leverage,
            max_drawdown_stop=args.max_drawdown_stop,
            cooldown_days=args.cooldown_days,
            max_symbol_trades=cap,
            initial_capital=args.initial_capital,
            allow_fractional_shares=args.allow_fractional_shares,
            rebalance_interval_days=args.rebalance_interval_days,
        )
        daily = daily_portfolio_diagnostics(
            predictions,
            realized_return_col=args.realized_return,
            config=config,
        )
        symbols = symbol_contribution_diagnostics(
            predictions,
            realized_return_col=args.realized_return,
            config=config,
        )
        capital_config = BacktestConfig(
            long_fraction=args.capital_long_fraction,
            max_position_weight=args.capital_position_weight,
            transaction_cost_bps=config.transaction_cost_bps,
            slippage_bps=config.slippage_bps,
            target_volatility=args.target_volatility,
            volatility_lookback=args.volatility_lookback,
            max_leverage=args.max_leverage,
            max_drawdown_stop=args.max_drawdown_stop,
            cooldown_days=args.cooldown_days,
            max_symbol_trades=cap,
            initial_capital=args.initial_capital,
            allow_fractional_shares=args.allow_fractional_shares,
            rebalance_interval_days=args.rebalance_interval_days,
        )
        capital = long_only_capital_backtest(
            predictions.rename(columns={"prediction": "score"}),
            return_col=args.realized_return,
            config=capital_config,
        )
        rows.append(
            _summary_row(
                cap=cap,
                daily=daily,
                symbols=symbols,
                capital=capital,
                initial_capital=args.initial_capital,
            )
        )

    sensitivity = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(output_path, index=False)
    print(f"wrote {len(sensitivity)} concentration sensitivity rows to {output_path}")


def _parse_caps(value: str) -> list[int | None]:
    caps = []
    for item in value.split(","):
        normalized = item.strip().lower()
        if normalized in {"", "none", "null"}:
            caps.append(None)
        else:
            caps.append(int(normalized))
    return caps


def _summary_row(
    *,
    cap: int | None,
    daily: pd.DataFrame,
    symbols: pd.DataFrame,
    capital: pd.DataFrame,
    initial_capital: float,
) -> dict:
    gross_contribution = symbols["gross_contribution"].abs().sum()
    top_symbol_share = (
        symbols["gross_contribution"].abs().max() / gross_contribution
        if gross_contribution
        else pd.NA
    )
    return {
        "max_symbol_trades": cap if cap is not None else "none",
        "daily_rows": len(daily),
        "raw_mean_daily_return": daily["net_return"].mean(),
        "raw_sharpe": sharpe_ratio(daily["net_return"]),
        "raw_max_drawdown": max_drawdown(daily["net_return"]),
        "risk_mean_daily_return": daily["risk_net_return"].mean(),
        "risk_sharpe": sharpe_ratio(daily["risk_net_return"]),
        "risk_max_drawdown": max_drawdown(daily["risk_net_return"]),
        "avg_gross_exposure": daily["gross_exposure"].mean(),
        "trading_days": int(daily["risk_trading_enabled"].sum()),
        "symbols_used": int(len(symbols)),
        "top_symbol_share_abs_contribution": top_symbol_share,
        "capital_start": initial_capital,
        "capital_end": capital["capital"].iloc[-1],
        "capital_total_return": capital["capital"].iloc[-1] / initial_capital - 1.0,
        "capital_sharpe": sharpe_ratio(capital["net_return"]),
        "capital_max_drawdown": max_drawdown(capital["net_return"]),
        "capital_avg_positions": capital["positions"].mean(),
        "capital_avg_invested": capital["invested"].mean(),
    }


if __name__ == "__main__":
    main()
