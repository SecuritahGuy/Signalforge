from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from signalforge.backtest import BacktestConfig, event_based_long_only_backtest
from signalforge.experiments import (
    FAST_MODEL_SPECS,
    FEATURE_SETS,
    run_experiment_grid,
)
from signalforge.metrics import max_drawdown, sharpe_ratio


@dataclass(frozen=True)
class PortfolioRuleSpec:
    name: str
    long_fraction: float
    max_position_weight: float
    min_score: float | None
    max_symbol_trades: int | None = None
    max_drawdown_stop: float | None = None
    cooldown_days: int = 20
    rebalance_interval_days: int = 20
    allow_fractional_shares: bool = False


DEFAULT_PORTFOLIO_RULES = (
    PortfolioRuleSpec(
        name="current_10pct_weight_10pct_universe",
        long_fraction=0.10,
        max_position_weight=0.10,
        min_score=0.01,
    ),
    PortfolioRuleSpec(
        name="conservative_5pct_weight",
        long_fraction=0.10,
        max_position_weight=0.05,
        min_score=0.01,
    ),
    PortfolioRuleSpec(
        name="higher_score_threshold_2pct",
        long_fraction=0.10,
        max_position_weight=0.10,
        min_score=0.02,
    ),
    PortfolioRuleSpec(
        name="fewer_names_5pct_universe",
        long_fraction=0.05,
        max_position_weight=0.10,
        min_score=0.01,
    ),
    PortfolioRuleSpec(
        name="symbol_cap_75",
        long_fraction=0.10,
        max_position_weight=0.10,
        min_score=0.01,
        max_symbol_trades=75,
    ),
    PortfolioRuleSpec(
        name="drawdown_stop_12pct",
        long_fraction=0.10,
        max_position_weight=0.10,
        min_score=0.01,
        max_drawdown_stop=0.12,
        cooldown_days=20,
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SignalForge R&D feature-ablation and portfolio-rule experiments."
    )
    parser.add_argument("--research-frame", default="data/processed/research_frame.csv")
    parser.add_argument(
        "--predictions",
        default="reports/exec_top_experiment_min_score_001_predictions.csv",
    )
    parser.add_argument("--output-prefix", default="reports/rd")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--target-kind",
        choices=("close_excess", "exec_excess"),
        default="exec_excess",
    )
    parser.add_argument("--first-train-start", default="2020-01-01")
    parser.add_argument("--first-validation-start", default="2022-01-01")
    parser.add_argument("--validation-months", type=int, default=6)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--initial-capital", type=float, default=2_000.0)
    args = parser.parse_args()

    research_frame = pd.read_csv(args.research_frame)
    predictions = pd.read_csv(args.predictions)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    ablation, split_summaries = run_feature_ablation(
        research_frame,
        horizon=args.horizon,
        target_kind=args.target_kind,
        first_train_start=args.first_train_start,
        first_validation_start=args.first_validation_start,
        validation_months=args.validation_months,
        n_jobs=args.n_jobs,
    )
    portfolio_rules = run_portfolio_rule_experiments(
        predictions,
        initial_capital=args.initial_capital,
        rules=DEFAULT_PORTFOLIO_RULES,
    )
    ablation_path = output_prefix.with_name(output_prefix.name + "_feature_ablation.csv")
    split_path = output_prefix.with_name(output_prefix.name + "_feature_ablation_splits.csv")
    rules_path = output_prefix.with_name(output_prefix.name + "_portfolio_rules.csv")
    report_path = output_prefix.with_name(output_prefix.name + "_summary.md")

    ablation.to_csv(ablation_path, index=False)
    if split_summaries:
        pd.concat(split_summaries.values(), ignore_index=True).to_csv(split_path, index=False)
    portfolio_rules.to_csv(rules_path, index=False)
    report_path.write_text(render_rd_summary(ablation, portfolio_rules))
    print(f"wrote R&D outputs with prefix {output_prefix}")


def run_feature_ablation(
    research_frame: pd.DataFrame,
    *,
    horizon: int,
    target_kind: str,
    first_train_start: str,
    first_validation_start: str,
    validation_months: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    leaderboard, split_summaries = run_experiment_grid(
        research_frame,
        horizons=(horizon,),
        feature_sets=FEATURE_SETS,
        model_specs=FAST_MODEL_SPECS,
        first_train_start=first_train_start,
        first_validation_start=first_validation_start,
        validation_months=validation_months,
        n_jobs=n_jobs,
        target_kind=target_kind,
        backtest_config=BacktestConfig(
            initial_capital=2_000.0,
            max_position_weight=0.10,
            long_fraction=0.10,
            min_score=0.01,
            rebalance_interval_days=horizon,
        ),
    )
    return leaderboard, split_summaries


def run_portfolio_rule_experiments(
    predictions: pd.DataFrame,
    *,
    initial_capital: float,
    rules: tuple[PortfolioRuleSpec, ...],
) -> pd.DataFrame:
    rows = []
    for rule in rules:
        config = BacktestConfig(
            initial_capital=initial_capital,
            long_fraction=rule.long_fraction,
            max_position_weight=rule.max_position_weight,
            min_score=rule.min_score,
            max_symbol_trades=rule.max_symbol_trades,
            max_drawdown_stop=rule.max_drawdown_stop,
            cooldown_days=rule.cooldown_days,
            rebalance_interval_days=rule.rebalance_interval_days,
            allow_fractional_shares=rule.allow_fractional_shares,
        )
        equity, ledger = event_based_long_only_backtest(
            predictions.rename(columns={"prediction": "score"}),
            exit_price_col="exit_close_20d",
            config=config,
        )
        rows.append(
            _portfolio_rule_row(
                rule,
                equity=equity,
                ledger=ledger,
                initial_capital=initial_capital,
            )
        )
    return pd.DataFrame(rows).sort_values("ending_capital", ascending=False).reset_index(drop=True)


def render_rd_summary(ablation: pd.DataFrame, portfolio_rules: pd.DataFrame) -> str:
    top_ablation = ablation.head(8)
    top_rules = portfolio_rules.head(8)
    return "\n".join(
        [
            "# SignalForge R&D Summary",
            "",
            "## Feature Ablation",
            "",
            _markdown_table(
                top_ablation[
                    [
                        "experiment",
                        "feature_set",
                        "model",
                        "risk_backtest_sharpe",
                        "risk_backtest_max_drawdown",
                        "ic_mean",
                        "positive_ic_splits",
                    ]
                ]
            ),
            "",
            "## Portfolio Rules",
            "",
            _markdown_table(
                top_rules[
                    [
                        "rule",
                        "ending_capital",
                        "total_return",
                        "sharpe",
                        "max_drawdown",
                        "filled_trades",
                        "skipped_trades",
                    ]
                ]
            ),
            "",
        ]
    )


def _portfolio_rule_row(
    rule: PortfolioRuleSpec,
    *,
    equity: pd.DataFrame,
    ledger: pd.DataFrame,
    initial_capital: float,
) -> dict:
    ending_capital = float(equity["capital"].iloc[-1]) if not equity.empty else initial_capital
    net_returns = (
        pd.to_numeric(equity["net_return"], errors="coerce").fillna(0.0)
        if not equity.empty
        else pd.Series(dtype="float64")
    )
    filled = ledger.loc[ledger["status"] == "filled"] if not ledger.empty else ledger
    skipped = ledger.loc[ledger["status"] == "skipped"] if not ledger.empty else ledger
    return {
        "rule": rule.name,
        "initial_capital": initial_capital,
        "ending_capital": ending_capital,
        "total_return": ending_capital / initial_capital - 1.0,
        "sharpe": sharpe_ratio(net_returns),
        "max_drawdown": max_drawdown(net_returns),
        "filled_trades": int(len(filled)),
        "skipped_trades": int(len(skipped)),
        "win_rate": float((filled["net_pnl"] > 0).mean()) if not filled.empty else 0.0,
        "avg_net_pnl": float(filled["net_pnl"].mean()) if not filled.empty else 0.0,
        "avg_positions": float(equity["positions"].mean()) if not equity.empty else 0.0,
        "avg_invested": float(equity["invested"].mean()) if not equity.empty else 0.0,
        "long_fraction": rule.long_fraction,
        "max_position_weight": rule.max_position_weight,
        "min_score": rule.min_score,
        "max_symbol_trades": rule.max_symbol_trades,
        "max_drawdown_stop": rule.max_drawdown_stop,
        "rebalance_interval_days": rule.rebalance_interval_days,
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    rows = [list(display.columns), ["---"] * len(display.columns)]
    rows.extend(display.astype(str).values.tolist())
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


if __name__ == "__main__":
    main()
