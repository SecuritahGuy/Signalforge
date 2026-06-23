from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalforge.ensemble import (
    BASE_MODEL_TYPES,
    EnsembleConfig,
    train_ensemble_walkforward,
)
from signalforge.exceptions import ModelError
from signalforge.modeling import BaselineModelConfig


def _research_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=130, freq="D")
    rows = []
    for sym_idx, symbol in enumerate(["AAPL", "MSFT", "NVDA", "AMZN"]):
        for day_idx, date in enumerate(dates):
            base = day_idx / 100
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "return_1d": 0.001 * (sym_idx + 1),
                    "return_5d": base,
                    "momentum_5d": base,
                    "volatility_5d": 0.01 + sym_idx / 1_000,
                    "return_20d": base * 2,
                    "momentum_20d": base * 2,
                    "volatility_20d": 0.02 + sym_idx / 1_000,
                    "return_60d": base * 3,
                    "momentum_60d": base * 3,
                    "volatility_60d": 0.03 + sym_idx / 1_000,
                    "volume_change_5d": 0.05,
                    "dollar_volume": 100_000_000 + sym_idx,
                    "avg_dollar_volume_20d": 100_000_000 + sym_idx,
                    "sector_rank_momentum_20d": (sym_idx + 1) / 4,
                    "sector_rank_volatility_20d": (4 - sym_idx) / 4,
                    "return_20d_lag_1": base * 2,
                    "volatility_20d_lag_1": 0.02 + sym_idx / 1_000,
                    "rsi_14": 50.0 + sym_idx,
                    "macd_histogram_12_26_9": 0.01 * sym_idx,
                    "bollinger_pct_b_20_2": 0.5 + sym_idx / 10,
                    "atr_14": 1.0 + sym_idx,
                    "day_of_week_sin": 0.0,
                    "day_of_week_cos": 1.0,
                    "month_sin": 0.0,
                    "month_cos": 1.0,
                    "zscore_return_20d": 0.0,
                    "zscore_volatility_20d": 0.0,
                    "return_20d_x_volatility_20d": base * 2 * 0.02,
                    "momentum_factor": 0.0,
                    "low_vol_factor": 0.5,
                    "fwd_5d_return": base / 10 + sym_idx / 1_000,
                    "fwd_5d_excess_return": base / 20 + sym_idx / 1_000,
                }
            )
    return pd.DataFrame(rows)


def test_ensemble_average_returns_predictions_summary_and_metadata():
    frame = _research_frame()
    config = EnsembleConfig(
        model_types=("ridge", "elasticnet"),
        method="average",
    )
    base_cfg = BaselineModelConfig(
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        purge_days=5,
        n_estimators=20,
        max_depth=3,
        min_samples_leaf=5,
    )

    predictions, summary, metadata = train_ensemble_walkforward(
        frame, ensemble_config=config, base_config=base_cfg
    )

    assert not predictions.empty
    assert {"date", "symbol", "prediction", "split_id"}.issubset(predictions.columns)
    assert not summary.empty
    assert {"ic_spearman", "directional_hit_rate", "backtest_sharpe"}.issubset(summary.columns)
    assert metadata["split_count"] == len(summary)
    assert metadata["ensemble_config"]["method"] == "average"
    assert len(metadata["ensemble_config"]["model_types"]) == 2


def test_ensemble_weighted_returns_predictions():
    frame = _research_frame()
    config = EnsembleConfig(
        model_types=("ridge", "elasticnet", "random_forest"),
        method="weighted",
    )
    base_cfg = BaselineModelConfig(
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        purge_days=5,
        n_estimators=20,
        max_depth=3,
        min_samples_leaf=5,
    )

    predictions, summary, metadata = train_ensemble_walkforward(
        frame, ensemble_config=config, base_config=base_cfg
    )

    assert not predictions.empty
    assert not summary.empty
    assert metadata["ensemble_config"]["method"] == "weighted"
    # Each model type should have a per-split IC column in summary
    for mt in ("ridge", "elasticnet", "random_forest"):
        assert f"{mt}_ic" in summary.columns


def test_ensemble_meta_returns_predictions():
    frame = _research_frame()
    config = EnsembleConfig(
        model_types=("ridge", "elasticnet"),
        method="meta",
    )
    base_cfg = BaselineModelConfig(
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        purge_days=5,
        n_estimators=20,
        max_depth=3,
        min_samples_leaf=5,
    )

    predictions, summary, metadata = train_ensemble_walkforward(
        frame, ensemble_config=config, base_config=base_cfg
    )

    assert not predictions.empty
    assert metadata["ensemble_config"]["method"] == "meta"


def test_ensemble_raises_with_single_model():
    with pytest.raises(ModelError, match="at least two model types"):
        EnsembleConfig(model_types=("ridge",))


def test_ensemble_raises_with_unknown_model():
    with pytest.raises(ModelError, match="unknown model types"):
        EnsembleConfig(model_types=("ridge", "fake_model"))


def test_ensemble_raises_with_unknown_method():
    with pytest.raises(ModelError, match="unsupported ensemble method"):
        EnsembleConfig(model_types=("ridge", "elasticnet"), method="unknown")


def test_ensemble_works_with_lgbm_xgboost():
    # Will be skipped if deps are missing, but should still construct config
    config = EnsembleConfig(model_types=("lgbm", "xgboost"))
    assert len(config.model_types) == 2


def test_ensemble_methods_are_all_defined():
    from signalforge.ensemble import ENSEMBLE_METHODS
    assert "average" in ENSEMBLE_METHODS
    assert "weighted" in ENSEMBLE_METHODS
    assert "meta" in ENSEMBLE_METHODS


def test_base_model_types_are_exported():
    assert "ridge" in BASE_MODEL_TYPES
    assert "elasticnet" in BASE_MODEL_TYPES
    assert "random_forest" in BASE_MODEL_TYPES
    assert "lgbm" in BASE_MODEL_TYPES
    assert "xgboost" in BASE_MODEL_TYPES
