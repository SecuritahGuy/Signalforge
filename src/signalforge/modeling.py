from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from signalforge.exceptions import ModelError
from signalforge.logging_config import get_logger

logger = get_logger(__name__)

try:
    import lightgbm as lgb

    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

try:
    import xgboost as xgb

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from signalforge.backtest import BacktestConfig, long_short_daily_returns
from signalforge.metrics import hit_rate, information_coefficient, max_drawdown, sharpe_ratio
from signalforge.validation import walk_forward_splits

DEFAULT_FEATURE_COLUMNS = (
    "return_1d",
    "return_5d",
    "momentum_5d",
    "volatility_5d",
    "return_20d",
    "momentum_20d",
    "volatility_20d",
    "return_60d",
    "momentum_60d",
    "volatility_60d",
    "volume_change_5d",
    "dollar_volume",
    "avg_dollar_volume_20d",
    "sector_rank_momentum_20d",
    "sector_rank_volatility_20d",
    # New expanded features
    "return_20d_lag_1",
    "volatility_20d_lag_1",
    "rsi_14",
    "macd_histogram_12_26_9",
    "bollinger_pct_b_20_2",
    "atr_14",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "zscore_return_20d",
    "zscore_volatility_20d",
    "return_20d_x_volatility_20d",
    "momentum_factor",
    "low_vol_factor",
)


@dataclass(frozen=True)
class BaselineModelConfig:
    target_col: str = "fwd_5d_excess_return"
    realized_return_col: str = "fwd_5d_return"
    benchmark_score_col: str = "momentum_20d"
    first_train_start: str = "2020-01-01"
    first_validation_start: str = "2022-01-01"
    validation_months: int = 6
    purge_days: int = 5
    embargo_days: int = 0
    model_type: Literal[
        "ridge", "elasticnet", "random_forest", "lgbm", "xgboost"
    ] = "ridge"
    alpha: float = 1.0
    l1_ratio: float = 0.25
    n_estimators: int = 600
    max_depth: int | None = 6
    min_samples_leaf: int = 25
    random_state: int = 42
    n_jobs: int = -1
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    num_leaves: int = 31
    min_child_samples: int = 20
    min_child_weight: float = 1.0


def train_baseline_walkforward(
    research_frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    config: BaselineModelConfig | None = None,
    backtest_config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Train a linear baseline over leakage-aware walk-forward splits."""
    cfg = config or BaselineModelConfig()
    missing = _required_columns(feature_columns, cfg).difference(research_frame.columns)
    if missing:
        raise ModelError(f"research_frame is missing required columns: {sorted(missing)}")

    frame = research_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=[*feature_columns, cfg.target_col, cfg.realized_return_col])
    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    if frame.empty:
        raise ModelError("no model-ready rows remain after dropping null features and targets")

    logger.info(
        "walkforward training: model=%s, %d features, %d rows, %d symbols",
        cfg.model_type, len(feature_columns), len(frame), frame["symbol"].nunique(),
    )

    prediction_frames = []
    summary_rows = []
    importance_frames = []
    for split_id, split in enumerate(
        walk_forward_splits(
            frame,
            first_train_start=cfg.first_train_start,
            first_validation_start=cfg.first_validation_start,
            validation_months=cfg.validation_months,
            purge_days=cfg.purge_days,
            embargo_days=cfg.embargo_days,
        ),
        start=1,
    ):
        train = frame.loc[split.train_index]
        validation = frame.loc[split.validation_index]
        if train.empty or validation.empty:
            continue

        model = _build_model(cfg)
        model.fit(train.loc[:, feature_columns], train[cfg.target_col])
        importance = _feature_importance(model, feature_columns)
        if importance is not None:
            importance["split_id"] = split_id
            importance_frames.append(importance)
        prediction_columns = [
            "date",
            "symbol",
            cfg.target_col,
            cfg.realized_return_col,
            cfg.benchmark_score_col,
        ]
        prediction_columns.extend(
            column
            for column in (
                "adj_close",
                "close",
                "next_open",
                "exit_close_5d",
                "exit_close_20d",
                "avg_dollar_volume_20d",
            )
            if column in validation.columns
        )
        predictions = validation.loc[:, prediction_columns].copy()
        predictions["prediction"] = model.predict(validation.loc[:, feature_columns])
        predictions["split_id"] = split_id
        predictions["train_start"] = split.train_start
        predictions["train_end"] = split.train_end
        predictions["validation_start"] = split.validation_start
        predictions["validation_end"] = split.validation_end
        prediction_frames.append(predictions)

        split_backtest = long_short_daily_returns(
            predictions.rename(
                columns={"prediction": "score", cfg.realized_return_col: "forward_return"}
            ),
            config=backtest_config,
        )
        summary_rows.append(
            _summarize_split(
                split_id=split_id,
                train=train,
                validation=validation,
                predictions=predictions,
                split_backtest=split_backtest,
                target_col=cfg.target_col,
                realized_return_col=cfg.realized_return_col,
                benchmark_score_col=cfg.benchmark_score_col,
                backtest_config=backtest_config,
            )
        )

    if not prediction_frames:
        raise ModelError("walk-forward configuration produced no usable validation splits")

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    logger.info(
        "walkforward complete: %d splits, %d predictions, mean sharpe=%.3f",
        len(summary), len(all_predictions),
        summary["backtest_sharpe"].mean() if "backtest_sharpe" in summary.columns else float("nan"),
    )
    feature_importance = _aggregate_feature_importance(importance_frames)
    metadata = {
        "config": asdict(cfg),
        "feature_columns": list(feature_columns),
        "row_count": int(len(frame)),
        "symbol_count": int(frame["symbol"].nunique()),
        "prediction_count": int(len(all_predictions)),
        "split_count": int(len(summary)),
    }
    if not feature_importance.empty:
        metadata["feature_importance"] = feature_importance.to_dict(orient="records")
    return all_predictions, summary, metadata


def _build_model(config: BaselineModelConfig) -> Pipeline:
    if config.model_type == "ridge":
        estimator = Ridge(alpha=config.alpha)
    elif config.model_type == "elasticnet":
        estimator = ElasticNet(alpha=config.alpha, l1_ratio=config.l1_ratio, max_iter=10_000)
    elif config.model_type == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )
    elif config.model_type == "lgbm":
        if not _LGBM_AVAILABLE:
            raise ImportError(
                "lightgbm is not installed. Install it with: pip install signalforge[modeling]"
            )
        estimator = lgb.LGBMRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            num_leaves=config.num_leaves,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            min_child_samples=config.min_child_samples,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            verbose=-1,
        )
    elif config.model_type == "xgboost":
        if not _XGB_AVAILABLE:
            raise ImportError(
                "xgboost is not installed. Install it with: pip install signalforge[modeling]"
            )
        estimator = xgb.XGBRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            min_child_weight=config.min_child_weight,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            verbosity=0,
        )
    else:
        raise ModelError(f"unsupported model_type: {config.model_type}")

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if config.model_type in {"ridge", "elasticnet"}:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def _summarize_split(
    *,
    split_id: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    predictions: pd.DataFrame,
    split_backtest: pd.DataFrame,
    target_col: str,
    realized_return_col: str,
    benchmark_score_col: str,
    backtest_config: BacktestConfig | None,
) -> dict:
    net_returns = (
        split_backtest["net_return"] if "net_return" in split_backtest else pd.Series(dtype=float)
    )
    risk_net_returns = (
        split_backtest["risk_net_return"]
        if "risk_net_return" in split_backtest
        else pd.Series(dtype=float)
    )
    benchmark_backtest = long_short_daily_returns(
        predictions.rename(
            columns={benchmark_score_col: "score", realized_return_col: "forward_return"}
        ),
        config=backtest_config,
    )
    benchmark_net_returns = benchmark_backtest["net_return"]
    benchmark_risk_net_returns = benchmark_backtest["risk_net_return"]
    directional_hits = (
        (predictions["prediction"] > 0) == (predictions[target_col] > 0)
    ).astype(float)
    return {
        "split_id": split_id,
        "train_start": train["date"].min(),
        "train_end": train["date"].max(),
        "validation_start": validation["date"].min(),
        "validation_end": validation["date"].max(),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "symbols": validation["symbol"].nunique(),
        "ic_spearman": information_coefficient(predictions["prediction"], predictions[target_col]),
        "directional_hit_rate": hit_rate(directional_hits),
        "backtest_mean_daily_return": net_returns.mean(),
        "backtest_sharpe": sharpe_ratio(net_returns),
        "backtest_max_drawdown": max_drawdown(net_returns),
        "risk_backtest_mean_daily_return": risk_net_returns.mean(),
        "risk_backtest_sharpe": sharpe_ratio(risk_net_returns),
        "risk_backtest_max_drawdown": max_drawdown(risk_net_returns),
        "risk_backtest_trading_days": int(split_backtest["risk_trading_enabled"].sum()),
        "benchmark_score_col": benchmark_score_col,
        "benchmark_ic_spearman": information_coefficient(
            predictions[benchmark_score_col],
            predictions[target_col],
        ),
        "benchmark_backtest_mean_daily_return": benchmark_net_returns.mean(),
        "benchmark_backtest_sharpe": sharpe_ratio(benchmark_net_returns),
        "benchmark_backtest_max_drawdown": max_drawdown(benchmark_net_returns),
        "benchmark_risk_backtest_mean_daily_return": benchmark_risk_net_returns.mean(),
        "benchmark_risk_backtest_sharpe": sharpe_ratio(benchmark_risk_net_returns),
        "benchmark_risk_backtest_max_drawdown": max_drawdown(benchmark_risk_net_returns),
    }


def _required_columns(
    feature_columns: tuple[str, ...],
    config: BaselineModelConfig,
) -> set[str]:
    return {
        "date",
        "symbol",
        config.target_col,
        config.realized_return_col,
        config.benchmark_score_col,
        *feature_columns,
    }


def _feature_importance(
    model: Pipeline,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame | None:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = abs(estimator.coef_)
    else:
        return None
    return pd.DataFrame({"feature": feature_columns, "importance": values})


def _aggregate_feature_importance(importance_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not importance_frames:
        return pd.DataFrame(columns=["feature", "mean_importance", "std_importance"])
    combined = pd.concat(importance_frames, ignore_index=True)
    return (
        combined.groupby("feature", as_index=False)["importance"]
        .agg(mean_importance="mean", std_importance="std")
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )
