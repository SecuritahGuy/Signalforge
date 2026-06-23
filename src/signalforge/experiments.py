from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from signalforge.backtest import BacktestConfig
from signalforge.modeling import BaselineModelConfig, train_baseline_walkforward

FEATURE_SETS = {
    "momentum": (
        "return_5d",
        "return_20d",
        "return_60d",
        "return_20d_skip_5d",
        "return_60d_skip_5d",
        "sector_rank_momentum_20d",
    ),
    "volatility_liquidity": (
        "volatility_5d",
        "volatility_20d",
        "volatility_60d",
        "volatility_ratio_5d_20d",
        "volatility_ratio_20d_60d",
        "volume_change_5d",
        "relative_volume_20d",
        "log_dollar_volume",
        "log_avg_dollar_volume_20d",
        "sector_rank_volatility_20d",
    ),
    "relative_strength": (
        "stock_minus_market_return_5d",
        "stock_minus_market_return_20d",
        "stock_minus_market_return_60d",
        "sector_return_20d",
        "sector_return_60d",
        "stock_minus_sector_return_20d",
        "stock_minus_sector_return_60d",
        "sector_rank_return_20d",
        "sector_rank_return_60d",
        "beta_60d",
        "correlation_to_market_60d",
    ),
    "trend_range": (
        "price_above_sma_20",
        "price_above_sma_50",
        "distance_from_20d_high",
        "distance_from_60d_high",
        "drawdown_20d",
        "drawdown_60d",
        "high_low_range_20d",
        "high_low_range_60d",
    ),
    "all_technical": (
        "return_1d",
        "return_5d",
        "volatility_5d",
        "return_20d",
        "return_20d_skip_5d",
        "volatility_20d",
        "return_60d",
        "return_60d_skip_5d",
        "volatility_60d",
        "volatility_ratio_5d_20d",
        "volatility_ratio_20d_60d",
        "volume_change_5d",
        "relative_volume_20d",
        "log_dollar_volume",
        "log_avg_dollar_volume_20d",
        "stock_minus_market_return_20d",
        "stock_minus_market_return_60d",
        "stock_minus_sector_return_20d",
        "stock_minus_sector_return_60d",
        "beta_60d",
        "correlation_to_market_60d",
        "price_above_sma_20",
        "price_above_sma_50",
        "distance_from_20d_high",
        "drawdown_60d",
        "sector_rank_momentum_20d",
        "sector_rank_volatility_20d",
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_type: str
    alpha: float = 1.0
    n_estimators: int = 300
    max_depth: int | None = 6
    min_samples_leaf: int = 25


DEFAULT_MODEL_SPECS = (
    ModelSpec(name="ridge", model_type="ridge"),
    ModelSpec(name="elasticnet", model_type="elasticnet", alpha=0.0005),
    ModelSpec(
        name="rf_shallow",
        model_type="random_forest",
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=40,
    ),
    ModelSpec(
        name="rf_balanced",
        model_type="random_forest",
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=25,
    ),
    ModelSpec(
        name="rf_large_leaf",
        model_type="random_forest",
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=60,
    ),
)

FAST_MODEL_SPECS = (
    ModelSpec(name="ridge", model_type="ridge"),
    ModelSpec(
        name="rf_fast",
        model_type="random_forest",
        n_estimators=180,
        max_depth=6,
        min_samples_leaf=30,
    ),
)

FAST_FEATURE_SETS = {
    "volatility_liquidity": FEATURE_SETS["volatility_liquidity"],
    "relative_strength": FEATURE_SETS["relative_strength"],
    "all_technical": FEATURE_SETS["all_technical"],
}


def run_experiment_grid(
    research_frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (5, 20),
    feature_sets: dict[str, tuple[str, ...]] | None = None,
    model_specs: tuple[ModelSpec, ...] = DEFAULT_MODEL_SPECS,
    first_train_start: str = "2020-01-01",
    first_validation_start: str = "2022-01-01",
    validation_months: int = 6,
    random_state: int = 42,
    n_jobs: int = -1,
    target_kind: Literal["close_excess", "exec_excess"] = "close_excess",
    backtest_config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run a leakage-aware experiment grid and return leaderboard plus summaries."""
    selected_feature_sets = feature_sets or FEATURE_SETS
    leaderboard_rows = []
    split_summaries = {}

    for horizon in horizons:
        for feature_set_name, feature_columns in selected_feature_sets.items():
            for model_spec in model_specs:
                target_suffix = "" if target_kind == "close_excess" else f"_{target_kind}"
                experiment_name = f"{model_spec.name}_{feature_set_name}_{horizon}d{target_suffix}"
                purge_days = horizon
                config = _config_from_spec(
                    model_spec,
                    horizon=horizon,
                    first_train_start=first_train_start,
                    first_validation_start=first_validation_start,
                    validation_months=validation_months,
                    purge_days=purge_days,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    target_kind=target_kind,
                )
                try:
                    _, summary, metadata = train_baseline_walkforward(
                        research_frame,
                        feature_columns=feature_columns,
                        config=config,
                        backtest_config=backtest_config,
                    )
                except ValueError as exc:
                    leaderboard_rows.append(
                        _failed_row(
                            experiment_name=experiment_name,
                            horizon=horizon,
                            feature_set_name=feature_set_name,
                            model_spec=model_spec,
                            target_kind=target_kind,
                            error=str(exc),
                        )
                    )
                    continue

                split_summaries[experiment_name] = summary.assign(
                    experiment=experiment_name,
                    horizon=horizon,
                    feature_set=feature_set_name,
                    model=model_spec.name,
                )
                leaderboard_rows.append(
                    _leaderboard_row(
                        experiment_name=experiment_name,
                        horizon=horizon,
                        feature_set_name=feature_set_name,
                        model_spec=model_spec,
                        target_kind=target_kind,
                        summary=summary,
                        metadata=metadata,
                    )
                )

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["risk_backtest_sharpe", "ic_mean"],
        ascending=[False, False],
    )
    return leaderboard.reset_index(drop=True), split_summaries


def _config_from_spec(
    spec: ModelSpec,
    *,
    horizon: int,
    first_train_start: str,
    first_validation_start: str,
    validation_months: int,
    purge_days: int,
    random_state: int,
    n_jobs: int,
    target_kind: Literal["close_excess", "exec_excess"],
) -> BaselineModelConfig:
    target_col, realized_return_col = _target_columns(horizon, target_kind=target_kind)
    return BaselineModelConfig(
        target_col=target_col,
        realized_return_col=realized_return_col,
        first_train_start=first_train_start,
        first_validation_start=first_validation_start,
        validation_months=validation_months,
        purge_days=purge_days,
        model_type=spec.model_type,
        alpha=spec.alpha,
        n_estimators=spec.n_estimators,
        max_depth=spec.max_depth,
        min_samples_leaf=spec.min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def _leaderboard_row(
    *,
    experiment_name: str,
    horizon: int,
    feature_set_name: str,
    model_spec: ModelSpec,
    target_kind: str,
    summary: pd.DataFrame,
    metadata: dict,
) -> dict:
    return {
        "experiment": experiment_name,
        "horizon": horizon,
        "feature_set": feature_set_name,
        "model": model_spec.name,
        "model_type": model_spec.model_type,
        "target_kind": target_kind,
        "split_count": metadata["split_count"],
        "prediction_count": metadata["prediction_count"],
        "ic_mean": summary["ic_spearman"].mean(),
        "ic_std": summary["ic_spearman"].std(),
        "positive_ic_splits": int((summary["ic_spearman"] > 0).sum()),
        "directional_hit_rate": summary["directional_hit_rate"].mean(),
        "backtest_mean_daily_return": summary["backtest_mean_daily_return"].mean(),
        "backtest_sharpe": summary["backtest_sharpe"].mean(),
        "backtest_max_drawdown": summary["backtest_max_drawdown"].min(),
        "risk_backtest_mean_daily_return": summary["risk_backtest_mean_daily_return"].mean(),
        "risk_backtest_sharpe": summary["risk_backtest_sharpe"].mean(),
        "risk_backtest_max_drawdown": summary["risk_backtest_max_drawdown"].min(),
        "risk_backtest_avg_trading_days": summary["risk_backtest_trading_days"].mean(),
        "benchmark_sharpe": summary["benchmark_backtest_sharpe"].mean(),
        "benchmark_risk_sharpe": summary["benchmark_risk_backtest_sharpe"].mean(),
        "error": "",
    }


def _failed_row(
    *,
    experiment_name: str,
    horizon: int,
    feature_set_name: str,
    model_spec: ModelSpec,
    target_kind: str,
    error: str,
) -> dict:
    return {
        "experiment": experiment_name,
        "horizon": horizon,
        "feature_set": feature_set_name,
        "model": model_spec.name,
        "model_type": model_spec.model_type,
        "target_kind": target_kind,
        "split_count": 0,
        "prediction_count": 0,
        "ic_mean": pd.NA,
        "ic_std": pd.NA,
        "positive_ic_splits": 0,
        "directional_hit_rate": pd.NA,
        "backtest_mean_daily_return": pd.NA,
        "backtest_sharpe": pd.NA,
        "backtest_max_drawdown": pd.NA,
        "risk_backtest_mean_daily_return": pd.NA,
        "risk_backtest_sharpe": pd.NA,
        "risk_backtest_max_drawdown": pd.NA,
        "risk_backtest_avg_trading_days": pd.NA,
        "benchmark_sharpe": pd.NA,
        "benchmark_risk_sharpe": pd.NA,
        "error": error,
    }


def _target_columns(
    horizon: int,
    *,
    target_kind: Literal["close_excess", "exec_excess"],
) -> tuple[str, str]:
    if target_kind == "close_excess":
        return f"fwd_{horizon}d_excess_return", f"fwd_{horizon}d_return"
    if target_kind == "exec_excess":
        return f"fwd_{horizon}d_exec_excess_return", f"fwd_{horizon}d_exec_return"
    raise ValueError(f"unsupported target_kind: {target_kind}")
