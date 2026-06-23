from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from signalforge.model_registry import (
    ModelRegistry,
    save_walkforward_model,
)
from signalforge.modeling import BaselineModelConfig


def _stub_pipeline() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])


def _stub_config() -> BaselineModelConfig:
    return BaselineModelConfig(
        model_type="ridge",
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
    )


@pytest.fixture
def registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(
        db_path=str(tmp_path / "test.db"),
        artifact_dir=str(tmp_path / "models"),
    )


def test_save_and_load(registry: ModelRegistry):
    model = _stub_pipeline()
    config = _stub_config()
    features = ("return_1d", "momentum_20d")

    model_id = registry.save(model, feature_columns=features, config=config)

    assert len(model_id) == 16
    artifact = registry.load(model_id)
    assert artifact is not None
    assert artifact.model_id == model_id
    assert artifact.feature_columns == features
    assert artifact.config.model_type == "ridge"


def test_load_returns_none_for_missing(registry: ModelRegistry):
    assert registry.load("nonexistent") is None


def test_list_models_returns_dataframe(registry: ModelRegistry):
    model = _stub_pipeline()
    config = _stub_config()

    registry.save(model, feature_columns=("return_1d",), config=config)
    registry.save(model, feature_columns=("return_1d",), config=config)

    df = registry.list_models()
    assert len(df) == 2
    assert "model_id" in df.columns
    assert "model_type" in df.columns


def test_list_models_filters_by_type(registry: ModelRegistry):
    model = _stub_pipeline()

    cfg1 = BaselineModelConfig(model_type="ridge")
    cfg2 = BaselineModelConfig(model_type="elasticnet")

    registry.save(model, feature_columns=("return_1d",), config=cfg1)
    registry.save(model, feature_columns=("return_1d",), config=cfg2)

    ridges = registry.list_models(model_type="ridge")
    assert len(ridges) == 1
    assert ridges.iloc[0]["model_type"] == "ridge"


def test_save_with_metrics(registry: ModelRegistry):
    model = _stub_pipeline()
    config = _stub_config()

    model_id = registry.save(
        model,
        feature_columns=("return_1d",),
        config=config,
        metrics={"sharpe": 1.5, "ic": 0.05},
    )

    artifact = registry.load(model_id)
    assert artifact is not None
    assert artifact.metrics["sharpe"] == 1.5
    assert artifact.metrics["ic"] == 0.05


def test_save_with_metadata(registry: ModelRegistry):
    model = _stub_pipeline()
    config = _stub_config()

    model_id = registry.save(
        model,
        feature_columns=("return_1d",),
        config=config,
        metrics={"sharpe": 1.0},
        metadata={"training_rows": 5000, "training_symbols": 100},
    )

    row = registry.db.get_model(model_id)
    assert row is not None
    assert row["training_rows"] == 5000
    assert row["training_symbols"] == 100


def test_load_best_by_metric(registry: ModelRegistry):
    model = _stub_pipeline()
    config = _stub_config()

    registry.save(
        model, feature_columns=("return_1d",), config=config,
        metrics={"sharpe": 0.5},
    )
    mid2 = registry.save(
        model, feature_columns=("return_1d",), config=config,
        metrics={"sharpe": 2.0},
    )
    registry.save(
        model, feature_columns=("return_1d",), config=config,
        metrics={"sharpe": 1.0},
    )

    best = registry.load_best(metric="sharpe")
    assert best is not None
    assert best.model_id == mid2


def test_load_best_filters_by_type(registry: ModelRegistry):
    model = _stub_pipeline()

    registry.save(
        model, feature_columns=("return_1d",),
        config=BaselineModelConfig(model_type="ridge"),
        metrics={"sharpe": 1.0},
    )
    mid_best = registry.save(
        model, feature_columns=("return_1d",),
        config=BaselineModelConfig(model_type="elasticnet"),
        metrics={"sharpe": 3.0},
    )
    registry.save(
        model, feature_columns=("return_1d",),
        config=BaselineModelConfig(model_type="elasticnet"),
        metrics={"sharpe": 2.0},
    )

    best = registry.load_best(model_type="elasticnet", metric="sharpe")
    assert best is not None
    assert best.model_id == mid_best


def test_compare_models(registry: ModelRegistry):
    model = _stub_pipeline()
    config = _stub_config()

    mid1 = registry.save(
        model, feature_columns=("return_1d",), config=config,
        metrics={"sharpe": 0.5},
    )
    mid2 = registry.save(
        model, feature_columns=("return_1d",), config=config,
        metrics={"sharpe": 1.5},
    )

    comparison = registry.compare_models([mid1, mid2])
    assert len(comparison) == 2
    assert "metric_sharpe" in comparison.columns
    assert comparison["metric_sharpe"].iloc[0] != comparison["metric_sharpe"].iloc[1]


def test_delete_removes_artifact_and_record(registry: ModelRegistry):
    model = _stub_pipeline()
    config = _stub_config()
    mid = registry.save(model, feature_columns=("return_1d",), config=config)

    assert registry.load(mid) is not None
    assert registry.delete(mid)
    assert registry.load(mid) is None


def test_artifact_file_exists_on_disk(registry: ModelRegistry):
    model = _stub_pipeline()
    config = _stub_config()
    mid = registry.save(model, feature_columns=("return_1d",), config=config)

    row = registry.db.get_model(mid)
    assert row is not None
    assert Path(row["artifact_path"]).exists()


def test_close_then_reopen_persists(registry: ModelRegistry):
    model = _stub_pipeline()
    config = _stub_config()
    mid = registry.save(model, feature_columns=("return_1d",), config=config)

    artifact_dir = registry._artifact_dir
    db_path = registry.db.path
    registry.close()

    registry2 = ModelRegistry(db_path=db_path, artifact_dir=artifact_dir)
    artifact = registry2.load(mid)
    assert artifact is not None
    assert artifact.model_id == mid
    registry2.close()


def test_save_walkforward_model_convenience(tmp_path: Path):
    reg = ModelRegistry(
        db_path=str(tmp_path / "test.db"),
        artifact_dir=str(tmp_path / "models"),
    )
    model = _stub_pipeline()
    config = _stub_config()
    predictions = pd.DataFrame({"date": ["2024-01-01"], "symbol": ["AAPL"], "prediction": [0.1]})
    summary = pd.DataFrame({
        "risk_backtest_sharpe": [1.5, 2.0],
        "ic_spearman": [0.03, 0.05],
        "validation_end": ["2024-02-01", "2024-03-01"],
    })
    metadata = {
        "row_count": 5000,
        "symbol_count": 100,
        "split_count": 2,
        "prediction_count": 100,
    }

    model_id = save_walkforward_model(
        predictions, summary, metadata, model,
        feature_columns=("return_1d", "momentum_20d"),
        config=config,
        registry=reg,
    )

    artifact = reg.load(model_id)
    assert artifact is not None
    assert artifact.metrics["sharpe"] == pytest.approx(1.75)
    assert artifact.metrics["split_count"] == 2
    assert artifact.metrics["ic"] == pytest.approx(0.04)
