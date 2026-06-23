from __future__ import annotations

import pandas as pd
import pytest

from signalforge.hyperopt import (
    SEARCH_METHODS,
    HyperoptConfig,
    best_trial_summary,
    run_hyperparameter_optimization,
    trial_importance_data,
)
from signalforge.modeling import BaselineModelConfig

pytestmark = [
    pytest.mark.skipif(
        not __import__("signalforge.hyperopt", fromlist=["HYPEROPT_AVAILABLE"]).HYPEROPT_AVAILABLE,
        reason="optuna not installed",
    ),
]


def _research_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=130, freq="D")
    rows = []
    for sym_idx, symbol in enumerate(["AAPL", "MSFT", "NVDA", "AMZN"]):
        for day_idx, date in enumerate(dates):
            base = day_idx / 100
            rows.append({
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
            })
    return pd.DataFrame(rows)


def test_hyperopt_config_defaults():
    cfg = HyperoptConfig()
    assert cfg.model_type == "ridge"
    assert cfg.method == "bayesian"
    assert cfg.n_trials == 50


def test_hyperopt_config_validates_model_type():
    with pytest.raises(ValueError, match="unsupported model_type"):
        HyperoptConfig(model_type="invalid")


def test_hyperopt_config_validates_method():
    with pytest.raises(ValueError, match="unsupported method"):
        HyperoptConfig(method="invalid")


def test_hyperopt_config_validates_pruner():
    with pytest.raises(ValueError, match="unsupported pruner"):
        HyperoptConfig(pruner="invalid")


def test_run_hyperopt_ridge_bayesian():
    frame = _research_frame()
    cfg = HyperoptConfig(model_type="ridge", method="bayesian", n_trials=5)
    base_cfg = BaselineModelConfig(
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        purge_days=5,
    )

    result = run_hyperparameter_optimization(
        frame, config=cfg, base_config=base_cfg,
    )

    assert "best_params" in result
    assert "best_value" in result
    assert "study" in result
    assert "trials_frame" in result
    assert "alpha" in result["best_params"]
    assert isinstance(result["best_value"], float)
    assert len(result["trials_frame"]) >= 1


def test_run_hyperopt_elasticnet_bayesian():
    frame = _research_frame()
    cfg = HyperoptConfig(model_type="elasticnet", method="bayesian", n_trials=5)
    base_cfg = BaselineModelConfig(
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        purge_days=5,
    )

    result = run_hyperparameter_optimization(
        frame, config=cfg, base_config=base_cfg,
    )

    assert "alpha" in result["best_params"]
    assert "l1_ratio" in result["best_params"]


def test_run_hyperopt_random_forest_bayesian():
    frame = _research_frame()
    cfg = HyperoptConfig(model_type="random_forest", method="bayesian", n_trials=5)
    base_cfg = BaselineModelConfig(
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        purge_days=5,
        n_estimators=50,
        max_depth=10,
        min_samples_leaf=10,
    )

    result = run_hyperparameter_optimization(
        frame, config=cfg, base_config=base_cfg,
    )

    assert "n_estimators" in result["best_params"]
    assert "max_depth" in result["best_params"]
    assert "min_samples_leaf" in result["best_params"]


def test_run_hyperopt_ridge_random():
    frame = _research_frame()
    cfg = HyperoptConfig(model_type="ridge", method="random", n_trials=5)
    base_cfg = BaselineModelConfig(
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        purge_days=5,
    )

    result = run_hyperparameter_optimization(
        frame, config=cfg, base_config=base_cfg,
    )

    assert "best_params" in result
    assert result["config"]["method"] == "random"


def test_run_hyperopt_ridge_grid():
    frame = _research_frame()
    cfg = HyperoptConfig(model_type="ridge", method="grid", n_trials=10)
    base_cfg = BaselineModelConfig(
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        purge_days=5,
    )

    result = run_hyperparameter_optimization(
        frame, config=cfg, base_config=base_cfg,
    )

    assert "best_params" in result
    assert result["best_value"] is not None


def test_best_trial_summary_returns_readable_dict():
    frame = _research_frame()
    cfg = HyperoptConfig(model_type="ridge", method="bayesian", n_trials=5)
    base_cfg = BaselineModelConfig(
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        purge_days=5,
    )

    result = run_hyperparameter_optimization(
        frame, config=cfg, base_config=base_cfg,
    )
    summary = best_trial_summary(result)

    assert summary["model_type"] == "ridge"
    assert "best_value" in summary
    assert "best_params" in summary


def test_trial_importance_data_returns_dict_or_none():
    frame = _research_frame()
    cfg = HyperoptConfig(model_type="ridge", method="bayesian", n_trials=5)
    base_cfg = BaselineModelConfig(
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        purge_days=5,
    )

    result = run_hyperparameter_optimization(
        frame, config=cfg, base_config=base_cfg,
    )
    importance = trial_importance_data(result)

    if importance is not None:
        assert isinstance(importance, dict)


def test_hyperopt_with_median_pruner():
    frame = _research_frame()
    cfg = HyperoptConfig(
        model_type="ridge",
        method="bayesian",
        n_trials=5,
        pruner="median",
    )
    base_cfg = BaselineModelConfig(
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        purge_days=5,
    )

    result = run_hyperparameter_optimization(
        frame, config=cfg, base_config=base_cfg,
    )

    assert "best_params" in result


def test_hyperopt_trials_frame_sorted_by_value():
    frame = _research_frame()
    cfg = HyperoptConfig(model_type="ridge", method="random", n_trials=5)
    base_cfg = BaselineModelConfig(
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        purge_days=5,
    )

    result = run_hyperparameter_optimization(
        frame, config=cfg, base_config=base_cfg,
    )
    tf = result["trials_frame"]

    assert "number" in tf.columns
    assert "value" in tf.columns
    assert "state" in tf.columns
    values = tf["value"].dropna().values
    if len(values) > 1:
        assert (values[:-1] >= values[1:]).all()


def test_hyperopt_raises_without_optuna(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("signalforge.hyperopt.HYPEROPT_AVAILABLE", False)
    frame = _research_frame()

    with pytest.raises(ImportError, match="optuna is required"):
        run_hyperparameter_optimization(frame)


def test_hyperopt_elasticnet_grid():
    frame = _research_frame()
    cfg = HyperoptConfig(model_type="elasticnet", method="grid", n_trials=10)
    base_cfg = BaselineModelConfig(
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        purge_days=5,
    )

    result = run_hyperparameter_optimization(
        frame, config=cfg, base_config=base_cfg,
    )

    assert "best_params" in result
    assert "alpha" in result["best_params"]
    assert "l1_ratio" in result["best_params"]


def test_hyperopt_search_methods_are_exported():
    assert "bayesian" in SEARCH_METHODS
    assert "random" in SEARCH_METHODS
    assert "grid" in SEARCH_METHODS
