from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from signalforge.db import DatabaseConfig, SignalForgeDB
from signalforge.modeling import BaselineModelConfig

_MODEL_AVAILABLE: bool
try:
    import joblib

    _MODEL_AVAILABLE = True
except ImportError:
    _MODEL_AVAILABLE = False


@dataclass
class ModelArtifact:
    """Deserialized model artifact with metadata."""
    model: Pipeline
    feature_columns: tuple[str, ...]
    config: BaselineModelConfig
    metrics: dict[str, float]
    model_id: str
    version: int
    model_type: str
    created_at: str


class ModelRegistry:
    """Persistent model registry backed by SQLite and joblib serialization.

    Usage::

        registry = ModelRegistry()
        model_id = registry.save(pipeline, feature_cols, config, metrics)
        artifact = registry.load(model_id)
        best = registry.load_best(metric="sharpe")
        df = registry.list_models()
    """

    def __init__(
        self,
        db_path: str | None = None,
        artifact_dir: str | Path = "data/models",
    ) -> None:
        if not _MODEL_AVAILABLE:
            raise ImportError(
                "joblib is required for model persistence. "
                "Install it with: pip install joblib"
            )
        self._db = SignalForgeDB(
            DatabaseConfig(path=db_path) if db_path else None
        )
        self._artifact_dir = Path(artifact_dir)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

    @property
    def db(self) -> SignalForgeDB:
        return self._db

    def save(
        self,
        model: Pipeline,
        *,
        feature_columns: tuple[str, ...],
        config: BaselineModelConfig,
        metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Serialize a model to disk and register it in the database.

        Returns the model_id.
        """
        model_id = _generate_model_id()
        version = 1
        artifact_path = str(self._artifact_dir / f"{model_id}.joblib")
        joblib.dump(model, artifact_path)

        merged_metrics = dict(metrics or {})
        merged_metadata = dict(metadata or {})

        self._db.register_model({
            "model_id": model_id,
            "model_type": config.model_type,
            "version": version,
            "feature_columns": list(feature_columns),
            "config_json": asdict(config),
            "metrics_json": merged_metrics,
            "artifact_path": artifact_path,
            "training_rows": merged_metadata.get("training_rows", 0),
            "training_symbols": merged_metadata.get("training_symbols", 0),
            "training_end_date": merged_metadata.get("training_end_date", ""),
        })
        return model_id

    def load(self, model_id: str) -> ModelArtifact | None:
        """Load a model artifact from the registry by ID."""
        row = self._db.get_model(model_id)
        if row is None:
            return None

        artifact_path = Path(row["artifact_path"])
        if not artifact_path.exists():
            return None

        model: Pipeline = joblib.load(str(artifact_path))
        config = BaselineModelConfig(**row["config_json"])

        return ModelArtifact(
            model=model,
            feature_columns=tuple(row["feature_columns"]),
            config=config,
            metrics=row["metrics_json"],
            model_id=row["model_id"],
            version=row["version"],
            model_type=row["model_type"],
            created_at=row.get("created_at", ""),
        )

    def load_best(
        self,
        model_type: str | None = None,
        metric: str = "sharpe",
    ) -> ModelArtifact | None:
        """Load the best model by a given metric.

        Parameters
        ----------
        model_type : optional filter (e.g. \"ridge\", \"random_forest\").
        metric : metric name to sort by (default \"sharpe\").
        """
        models_df = self._db.list_models(model_type=model_type, limit=200)
        if models_df.empty:
            return None

        best: str | None = None
        best_val: float = -np.inf
        for _, row in models_df.iterrows():
            try:
                metrics = json.loads(row.get("metrics_json", "{}"))
                val = float(metrics.get(metric, -np.inf))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if val > best_val:
                best_val = val
                best = row["model_id"]

        if best is None:
            return None
        return self.load(best)

    def list_models(
        self, model_type: str | None = None, limit: int = 50
    ) -> pd.DataFrame:
        """List registered models as a DataFrame."""
        return self._db.list_models(model_type=model_type, limit=limit)

    def compare_models(
        self, model_ids: list[str]
    ) -> pd.DataFrame:
        """Compare multiple models side by side.

        Returns a DataFrame with one row per model showing metadata and metrics.
        """
        rows = []
        for mid in model_ids:
            row = self._db.get_model(mid)
            if row:
                rows.append(row)
        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(rows)
        cols = ["model_id", "model_type", "version", "created_at"]
        metrics_keys: set[str] = set()
        for row in rows:
            metrics_keys.update(row.get("metrics_json", {}).keys())
        frame["metrics_json"] = frame["metrics_json"].apply(
            lambda m: m if isinstance(m, dict) else {}
        )
        for key in sorted(metrics_keys):
            frame[f"metric_{key}"] = frame["metrics_json"].apply(
                lambda m, k=key: m.get(k, np.nan)
            )
        display_cols = cols + [c for c in sorted(frame.columns) if c.startswith("metric_")]
        return frame[display_cols]

    def delete(self, model_id: str) -> bool:
        """Delete a model from the registry and remove its artifact file."""
        row = self._db.get_model(model_id)
        if row is None:
            return False
        artifact_path = Path(row["artifact_path"])
        if artifact_path.exists():
            artifact_path.unlink()
        return self._db.delete_model(model_id)

    def close(self) -> None:
        self._db.close()


def save_walkforward_model(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: dict[str, Any],
    model: Pipeline,
    *,
    feature_columns: tuple[str, ...],
    config: BaselineModelConfig,
    registry: ModelRegistry | None = None,
    artifact_dir: str | Path = "data/models",
    db_path: str | None = None,
) -> str:
    """Convenience function to save a walk-forward trained model.

    Accepts the same output tuple from ``train_baseline_walkforward``.

    Returns the model_id.
    """
    reg = registry or ModelRegistry(
        db_path=db_path,
        artifact_dir=artifact_dir,
    )
    best_sharpe = summary["risk_backtest_sharpe"].max() if "risk_backtest_sharpe" in summary.columns else np.nan
    mean_sharpe = summary["risk_backtest_sharpe"].mean() if "risk_backtest_sharpe" in summary.columns else np.nan
    mean_ic = summary["ic_spearman"].mean() if "ic_spearman" in summary.columns else np.nan

    metrics = {
        "sharpe": float(mean_sharpe) if not np.isnan(mean_sharpe) else 0.0,
        "best_split_sharpe": float(best_sharpe) if not np.isnan(best_sharpe) else 0.0,
        "ic": float(mean_ic) if not np.isnan(mean_ic) else 0.0,
        "split_count": int(metadata.get("split_count", 0)),
        "prediction_count": int(metadata.get("prediction_count", 0)),
    }

    meta = {
        "training_rows": metadata.get("row_count", 0),
        "training_symbols": metadata.get("symbol_count", 0),
        "training_end_date": str(summary["validation_end"].max()) if "validation_end" in summary.columns else "",
    }

    return reg.save(
        model,
        feature_columns=feature_columns,
        config=config,
        metrics=metrics,
        metadata=meta,
    )


def _generate_model_id() -> str:
    return uuid.uuid4().hex[:16]
