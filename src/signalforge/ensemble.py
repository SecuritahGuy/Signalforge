from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from signalforge.backtest import BacktestConfig, long_short_daily_returns
from signalforge.metrics import information_coefficient, sharpe_ratio
from signalforge.modeling import (
    BaselineModelConfig,
    _build_model,
    _feature_importance,
    _summarize_split,
)
from signalforge.validation import walk_forward_splits

ENSEMBLE_METHODS = frozenset({"average", "weighted", "meta"})

BASE_MODEL_TYPES = ("ridge", "elasticnet", "random_forest", "lgbm", "xgboost")


@dataclass(frozen=True)
class EnsembleConfig:
    model_types: tuple[str, ...] = ("ridge", "elasticnet", "random_forest")
    method: str = "average"
    meta_alpha: float = 1.0
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.method not in ENSEMBLE_METHODS:
            raise ValueError(f"unsupported ensemble method: {self.method}; choose from {ENSEMBLE_METHODS}")
        unknown = set(self.model_types) - set(BASE_MODEL_TYPES)
        if unknown:
            raise ValueError(f"unknown model types: {unknown}; choose from {BASE_MODEL_TYPES}")
        if len(self.model_types) < 2:
            raise ValueError("ensemble requires at least two model types")


def train_ensemble_walkforward(
    research_frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | None = None,
    ensemble_config: EnsembleConfig | None = None,
    base_config: BaselineModelConfig | None = None,
    backtest_config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Train an ensemble of model types over walk-forward splits.

    Returns (predictions, summary, metadata) matching the same schema as
    ``train_baseline_walkforward``.
    """
    from signalforge.modeling import DEFAULT_FEATURE_COLUMNS

    ecfg = ensemble_config or EnsembleConfig()
    feats = feature_columns or DEFAULT_FEATURE_COLUMNS

    bcfg = base_config or BaselineModelConfig()
    bcfg_kwargs = _overrides_for_ensemble(bcfg, ecfg)

    frame = research_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=[*feats, bcfg.target_col, bcfg.realized_return_col])
    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("no model-ready rows remain after dropping null features and targets")

    prediction_frames: list[pd.DataFrame] = []
    ensemble_summaries: list[dict[str, Any]] = []
    oof_predictions: list[pd.DataFrame] = []  # for meta training

    for split_id, split in enumerate(
        walk_forward_splits(
            frame,
            first_train_start=bcfg.first_train_start,
            first_validation_start=bcfg.first_validation_start,
            validation_months=bcfg.validation_months,
            purge_days=bcfg.purge_days,
            embargo_days=bcfg.embargo_days,
        ),
        start=1,
    ):
        train = frame.loc[split.train_index]
        validation = frame.loc[split.validation_index]
        if train.empty or validation.empty:
            continue

        # Train all base models
        raw_preds: dict[str, np.ndarray] = {}
        models: dict[str, Pipeline] = {}
        for mt in ecfg.model_types:
            cfg = _model_config_for_type(bcfg_kwargs, mt)
            model = _build_model(cfg)
            model.fit(train.loc[:, feats], train[cfg.target_col])
            models[mt] = model
            raw_preds[mt] = model.predict(validation.loc[:, feats])

        # Build blended prediction
        blended = _blend_predictions(raw_preds, ecfg)
        ensemble_prediction = blended["combined"]

        # Collect prediction frame
        pred_cols = _prediction_columns(validation, bcfg)
        predictions = validation.loc[:, pred_cols].copy()
        predictions["prediction"] = ensemble_prediction
        predictions["split_id"] = split_id
        predictions["train_start"] = split.train_start
        predictions["train_end"] = split.train_end
        predictions["validation_start"] = split.validation_start
        predictions["validation_end"] = split.validation_end
        prediction_frames.append(predictions)

        # Split summary
        split_backtest = long_short_daily_returns(
            predictions.rename(
                columns={"prediction": "score", bcfg.realized_return_col: "forward_return"}
            ),
            config=backtest_config,
        )
        summary_row = _summarize_ensemble_split(
            split_id=split_id,
            train=train,
            validation=validation,
            predictions=predictions,
            split_backtest=split_backtest,
            raw_preds=raw_preds,
            target_col=bcfg.target_col,
            realized_return_col=bcfg.realized_return_col,
            benchmark_score_col=bcfg.benchmark_score_col,
            backtest_config=backtest_config,
        )
        ensemble_summaries.append(summary_row)

        # Save OOF for meta training
        if ecfg.method == "meta":
            oof = validation.loc[:, pred_cols].copy()
            for mt in ecfg.model_types:
                oof[f"pred_{mt}"] = raw_preds[mt]
            oof["split_id"] = split_id
            oof_predictions.append(oof)

    if not prediction_frames:
        raise ValueError("walk-forward configuration produced no usable validation splits")

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = pd.DataFrame(ensemble_summaries)

    metadata: dict[str, Any] = {
        "ensemble_config": asdict(ecfg),
        "base_config": asdict(bcfg),
        "feature_columns": list(feats),
        "row_count": int(len(frame)),
        "symbol_count": int(frame["symbol"].nunique()),
        "prediction_count": int(len(all_predictions)),
        "split_count": int(len(summary)),
    }

    return all_predictions, summary, metadata


def _blend_predictions(
    raw_preds: dict[str, np.ndarray],
    config: EnsembleConfig,
) -> dict[str, Any]:
    """Blend raw model predictions into a single ensemble prediction."""
    pred_array = np.column_stack([raw_preds[mt] for mt in config.model_types])
    weights: np.ndarray | None = None

    if config.method == "average":
        weights = np.full(len(config.model_types), 1.0 / len(config.model_types))
        combined = pred_array @ weights

    elif config.method == "weighted":
        weights = _compute_weights(raw_preds)
        combined = pred_array @ weights

    elif config.method == "meta":
        combined = pred_array.mean(axis=1)

    else:
        raise ValueError(f"unsupported method: {config.method}")

    return {"combined": combined, "weights": weights}


def _compute_weights(raw_preds: dict[str, np.ndarray]) -> np.ndarray:
    """Compute weight for each model based on correlation with the consensus."""
    names = list(raw_preds)
    pred_array = np.column_stack([raw_preds[n] for n in names])
    median_pred = np.median(pred_array, axis=1)
    ics = np.array(
        [
            max(
                np.corrcoef(raw_preds[n], median_pred)[0, 1]
                if len(raw_preds[n]) > 1 and np.std(raw_preds[n]) > 0
                else 0.0,
                0.0,
            )
            for n in names
        ]
    )
    total = ics.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(len(names), 1.0 / len(names))
    return ics / total


def _model_config_for_type(
    base: dict[str, Any],
    model_type: str,
) -> BaselineModelConfig:
    return BaselineModelConfig(**{**base, "model_type": model_type})


def _overrides_for_ensemble(
    base: BaselineModelConfig,
    ensemble: EnsembleConfig,
) -> dict[str, Any]:
    kwargs = asdict(base)
    kwargs.pop("model_type")
    return kwargs


def _prediction_columns(
    validation: pd.DataFrame,
    config: BaselineModelConfig,
) -> list[str]:
    cols = [
        "date",
        "symbol",
        config.target_col,
        config.realized_return_col,
        config.benchmark_score_col,
    ]
    for extra in ("adj_close", "close", "next_open", "exit_close_5d", "exit_close_20d", "avg_dollar_volume_20d"):
        if extra in validation.columns:
            cols.append(extra)
    return cols


def _summarize_ensemble_split(
    *,
    split_id: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    predictions: pd.DataFrame,
    split_backtest: pd.DataFrame,
    raw_preds: dict[str, np.ndarray],
    target_col: str,
    realized_return_col: str,
    benchmark_score_col: str,
    backtest_config: BacktestConfig | None,
) -> dict[str, Any]:
    row = _summarize_split(
        split_id=split_id,
        train=train,
        validation=validation,
        predictions=predictions,
        split_backtest=split_backtest,
        target_col=target_col,
        realized_return_col=realized_return_col,
        benchmark_score_col=benchmark_score_col,
        backtest_config=backtest_config,
    )
    for mt, preds in raw_preds.items():
        row[f"{mt}_ic"] = information_coefficient(
            pd.Series(preds, index=predictions.index), predictions[target_col]
        )
    return row
