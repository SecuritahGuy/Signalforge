import importlib

import pandas as pd
import pytest

from signalforge.modeling import BaselineModelConfig, train_baseline_walkforward


def _lgbm_available() -> bool:
    try:
        importlib.import_module("lightgbm")
        return True
    except ImportError:
        return False


def _xgb_available() -> bool:
    try:
        importlib.import_module("xgboost")
        return True
    except ImportError:
        return False


def test_train_baseline_walkforward_outputs_predictions_summary_and_metadata():
    frame = _model_frame()
    config = BaselineModelConfig(
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        purge_days=5,
    )

    predictions, summary, metadata = train_baseline_walkforward(frame, config=config)

    assert not predictions.empty
    assert {"date", "symbol", "prediction", "split_id"}.issubset(predictions.columns)
    assert not summary.empty
    assert {
        "ic_spearman",
        "directional_hit_rate",
        "backtest_sharpe",
        "benchmark_ic_spearman",
        "benchmark_backtest_sharpe",
    }.issubset(summary.columns)
    assert metadata["split_count"] == len(summary)
    assert metadata["feature_columns"]
    assert metadata["feature_importance"]


def test_train_baseline_walkforward_supports_random_forest():
    frame = _model_frame()
    config = BaselineModelConfig(
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        purge_days=5,
        model_type="random_forest",
        n_estimators=20,
        max_depth=3,
        min_samples_leaf=5,
        n_jobs=-1,
    )

    predictions, summary, metadata = train_baseline_walkforward(frame, config=config)

    assert not predictions.empty
    assert not summary.empty
    assert metadata["config"]["model_type"] == "random_forest"
    assert metadata["config"]["n_jobs"] == -1
    assert metadata["feature_importance"]


@pytest.mark.skipif(not _lgbm_available(), reason="lightgbm not installed")
def test_train_baseline_walkforward_supports_lgbm():
    frame = _model_frame()
    config = BaselineModelConfig(
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        purge_days=5,
        model_type="lgbm",
        n_estimators=20,
        max_depth=3,
        num_leaves=10,
        learning_rate=0.1,
    )

    predictions, summary, metadata = train_baseline_walkforward(frame, config=config)

    assert not predictions.empty
    assert not summary.empty
    assert metadata["config"]["model_type"] == "lgbm"
    assert metadata["feature_importance"]


@pytest.mark.skipif(not _xgb_available(), reason="xgboost not installed")
def test_train_baseline_walkforward_supports_xgboost():
    frame = _model_frame()
    config = BaselineModelConfig(
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        purge_days=5,
        model_type="xgboost",
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
    )

    predictions, summary, metadata = train_baseline_walkforward(frame, config=config)

    assert not predictions.empty
    assert not summary.empty
    assert metadata["config"]["model_type"] == "xgboost"
    assert metadata["feature_importance"]


def _model_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=130, freq="D")
    rows = []
    for symbol_index, symbol in enumerate(["AAPL", "MSFT", "NVDA", "AMZN"]):
        for day_index, date in enumerate(dates):
            base = day_index / 100
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "return_1d": 0.001 * (symbol_index + 1),
                    "return_5d": base,
                    "momentum_5d": base,
                    "volatility_5d": 0.01 + symbol_index / 1_000,
                    "return_20d": base * 2,
                    "momentum_20d": base * 2,
                    "volatility_20d": 0.02 + symbol_index / 1_000,
                    "return_60d": base * 3,
                    "momentum_60d": base * 3,
                    "volatility_60d": 0.03 + symbol_index / 1_000,
                    "volume_change_5d": 0.05,
                    "dollar_volume": 100_000_000 + symbol_index,
                    "avg_dollar_volume_20d": 100_000_000 + symbol_index,
                    "sector_rank_momentum_20d": (symbol_index + 1) / 4,
                    "sector_rank_volatility_20d": (4 - symbol_index) / 4,
                    "return_20d_lag_1": base * 2,
                    "volatility_20d_lag_1": 0.02 + symbol_index / 1_000,
                    "rsi_14": 50.0 + symbol_index,
                    "macd_histogram_12_26_9": 0.01 * symbol_index,
                    "bollinger_pct_b_20_2": 0.5 + symbol_index / 10,
                    "atr_14": 1.0 + symbol_index,
                    "day_of_week_sin": 0.0,
                    "day_of_week_cos": 1.0,
                    "month_sin": 0.0,
                    "month_cos": 1.0,
                    "zscore_return_20d": 0.0,
                    "zscore_volatility_20d": 0.0,
                    "return_20d_x_volatility_20d": base * 2 * 0.02,
                    "momentum_factor": 0.0,
                    "low_vol_factor": 0.5,
                    "fwd_5d_return": base / 10 + symbol_index / 1_000,
                    "fwd_5d_excess_return": base / 20 + symbol_index / 1_000,
                }
            )
    return pd.DataFrame(rows)
