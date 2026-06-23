import pandas as pd

from signalforge.backtest import BacktestConfig
from signalforge.experiments import ModelSpec, run_experiment_grid


def test_run_experiment_grid_builds_leaderboard_and_split_summaries():
    leaderboard, split_summaries = run_experiment_grid(
        _experiment_frame(),
        horizons=(5,),
        feature_sets={"momentum": ("momentum_5d", "momentum_20d")},
        model_specs=(
            ModelSpec(name="ridge", model_type="ridge"),
            ModelSpec(
                name="rf_smoke",
                model_type="random_forest",
                n_estimators=10,
                max_depth=3,
                min_samples_leaf=5,
            ),
        ),
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        n_jobs=-1,
        backtest_config=BacktestConfig(
            target_volatility=0.10,
            max_drawdown_stop=0.20,
            volatility_lookback=5,
        ),
    )

    assert len(leaderboard) == 2
    assert {
        "experiment",
        "ic_mean",
        "backtest_sharpe",
        "risk_backtest_sharpe",
        "positive_ic_splits",
    }.issubset(leaderboard.columns)
    assert split_summaries


def test_run_experiment_grid_supports_executable_target():
    leaderboard, _ = run_experiment_grid(
        _experiment_frame(),
        horizons=(5,),
        feature_sets={"momentum": ("momentum_5d", "momentum_20d")},
        model_specs=(ModelSpec(name="ridge", model_type="ridge"),),
        first_train_start="2024-01-01",
        first_validation_start="2024-03-01",
        validation_months=1,
        target_kind="exec_excess",
    )

    assert leaderboard.loc[0, "target_kind"] == "exec_excess"
    assert leaderboard.loc[0, "experiment"].endswith("_exec_excess")


def _experiment_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=130, freq="D")
    rows = []
    for symbol_index, symbol in enumerate(["AAPL", "MSFT", "NVDA", "AMZN"]):
        for day_index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "momentum_5d": day_index / 100,
                    "momentum_20d": day_index / 50,
                    "fwd_5d_return": day_index / 1_000 + symbol_index / 1_000,
                    "fwd_5d_excess_return": day_index / 2_000 + symbol_index / 1_000,
                    "fwd_5d_exec_return": day_index / 900 + symbol_index / 1_000,
                    "fwd_5d_exec_excess_return": day_index / 1_800 + symbol_index / 1_000,
                }
            )
    return pd.DataFrame(rows)
