from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from signalforge.backtest import (
    BacktestConfig,
    event_based_long_only_backtest,
    long_only_capital_backtest,
)
from signalforge.diagnostics import (
    daily_portfolio_diagnostics,
    monthly_portfolio_returns,
    symbol_contribution_diagnostics,
)
from signalforge.experiments import DEFAULT_MODEL_SPECS, FAST_MODEL_SPECS, FEATURE_SETS
from signalforge.modeling import BaselineModelConfig, train_baseline_walkforward


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the best experiment from a leaderboard.")
    parser.add_argument("--research-frame", default="data/processed/research_frame.csv")
    parser.add_argument("--leaderboard", default="reports/risk_experiment_leaderboard.csv")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--output-prefix", default="reports/top_experiment")
    parser.add_argument("--target-kind", choices=("close_excess", "exec_excess"), default=None)
    parser.add_argument("--target-volatility", type=float, default=0.12)
    parser.add_argument("--volatility-lookback", type=int, default=20)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--max-drawdown-stop", type=float, default=0.12)
    parser.add_argument("--disable-drawdown-stop", action="store_true")
    parser.add_argument("--cooldown-days", type=int, default=20)
    parser.add_argument("--max-symbol-trades", type=int, default=None)
    parser.add_argument("--initial-capital", type=float, default=2_000.0)
    parser.add_argument("--capital-position-weight", type=float, default=0.20)
    parser.add_argument("--capital-long-fraction", type=float, default=0.10)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--allow-fractional-shares", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    leaderboard = pd.read_csv(args.leaderboard)
    selected = _select_experiment(leaderboard, args.experiment)
    research_frame = pd.read_csv(args.research_frame)
    model_spec = _model_spec(selected["model"])
    feature_columns = FEATURE_SETS[selected["feature_set"]]
    horizon = int(selected["horizon"])
    target_kind = args.target_kind or selected.get("target_kind", "close_excess")
    target_col, realized_return_col = _target_columns(horizon, target_kind=target_kind)
    max_drawdown_stop = None if args.disable_drawdown_stop else args.max_drawdown_stop
    backtest_config = BacktestConfig(
        target_volatility=args.target_volatility,
        volatility_lookback=args.volatility_lookback,
        max_leverage=args.max_leverage,
        max_drawdown_stop=max_drawdown_stop,
        cooldown_days=args.cooldown_days,
        max_symbol_trades=args.max_symbol_trades,
    )
    capital_config = BacktestConfig(
        long_fraction=args.capital_long_fraction,
        max_position_weight=args.capital_position_weight,
        transaction_cost_bps=backtest_config.transaction_cost_bps,
        slippage_bps=backtest_config.slippage_bps,
        target_volatility=args.target_volatility,
        volatility_lookback=args.volatility_lookback,
        max_leverage=args.max_leverage,
        max_drawdown_stop=max_drawdown_stop,
        cooldown_days=args.cooldown_days,
        max_symbol_trades=args.max_symbol_trades,
        initial_capital=args.initial_capital,
        allow_fractional_shares=args.allow_fractional_shares,
        min_score=args.min_score,
        rebalance_interval_days=horizon,
    )
    model_config = BaselineModelConfig(
        target_col=target_col,
        realized_return_col=realized_return_col,
        first_train_start="2020-01-01",
        first_validation_start="2022-01-01",
        validation_months=6,
        purge_days=horizon,
        model_type=model_spec.model_type,
        alpha=model_spec.alpha,
        n_estimators=model_spec.n_estimators,
        max_depth=model_spec.max_depth,
        min_samples_leaf=model_spec.min_samples_leaf,
        n_jobs=args.n_jobs,
    )

    predictions, split_summary, metadata = train_baseline_walkforward(
        research_frame,
        feature_columns=feature_columns,
        config=model_config,
        backtest_config=backtest_config,
    )
    daily = daily_portfolio_diagnostics(
        predictions,
        realized_return_col=model_config.realized_return_col,
        config=backtest_config,
    )
    monthly = monthly_portfolio_returns(daily)
    symbols = symbol_contribution_diagnostics(
        predictions,
        realized_return_col=model_config.realized_return_col,
        config=backtest_config,
    )
    capital = long_only_capital_backtest(
        predictions.rename(columns={"prediction": "score"}),
        return_col=model_config.realized_return_col,
        config=capital_config,
    )
    event_equity, event_ledger = event_based_long_only_backtest(
        predictions.rename(
            columns={
                "prediction": "score",
                f"exit_close_{horizon}d": "exit_close",
            }
        ),
        exit_price_col="exit_close",
        config=capital_config,
    )

    prefix = Path(args.output_prefix)
    _write_csv(predictions, prefix.with_name(prefix.name + "_predictions.csv"))
    _write_csv(split_summary, prefix.with_name(prefix.name + "_split_summary.csv"))
    _write_csv(daily, prefix.with_name(prefix.name + "_daily_returns.csv"))
    _write_csv(monthly, prefix.with_name(prefix.name + "_monthly_returns.csv"))
    _write_csv(symbols, prefix.with_name(prefix.name + "_symbol_contributions.csv"))
    _write_csv(capital, prefix.with_name(prefix.name + "_capital_backtest.csv"))
    _write_csv(event_equity, prefix.with_name(prefix.name + "_event_equity.csv"))
    _write_csv(event_ledger, prefix.with_name(prefix.name + "_trade_ledger.csv"))
    if "feature_importance" in metadata:
        _write_csv(
            pd.DataFrame(metadata["feature_importance"]),
            prefix.with_name(prefix.name + "_feature_importance.csv"),
        )
    _write_json(
        {
            "selected_experiment": selected.to_dict(),
            "model_metadata": metadata,
            "target_kind": target_kind,
            "backtest_config": backtest_config.__dict__,
            "capital_backtest_config": capital_config.__dict__,
        },
        prefix.with_name(prefix.name + "_metadata.json"),
    )
    print(f"wrote top-experiment audit artifacts with prefix {prefix}")


def _select_experiment(leaderboard: pd.DataFrame, experiment: str | None) -> pd.Series:
    if experiment is None:
        return leaderboard.iloc[0]
    matches = leaderboard.loc[leaderboard["experiment"] == experiment]
    if matches.empty:
        raise ValueError(f"experiment {experiment!r} not found in leaderboard")
    return matches.iloc[0]


def _target_columns(horizon: int, *, target_kind: str) -> tuple[str, str]:
    if target_kind == "close_excess":
        return f"fwd_{horizon}d_excess_return", f"fwd_{horizon}d_return"
    if target_kind == "exec_excess":
        return f"fwd_{horizon}d_exec_excess_return", f"fwd_{horizon}d_exec_return"
    raise ValueError(f"unsupported target_kind: {target_kind}")


def _model_spec(name: str):
    for spec in (*DEFAULT_MODEL_SPECS, *FAST_MODEL_SPECS):
        if spec.name == name:
            return spec
    raise ValueError(f"model spec {name!r} not found")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
