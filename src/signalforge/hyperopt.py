from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from signalforge.backtest import BacktestConfig, long_short_daily_returns
from signalforge.metrics import sharpe_ratio
from signalforge.modeling import (
    DEFAULT_FEATURE_COLUMNS,
    BaselineModelConfig,
    _build_model,
    _summarize_split,
)
from signalforge.validation import walk_forward_splits

HYPEROPT_AVAILABLE: bool
try:
    import optuna

    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

SEARCH_METHODS = frozenset({"bayesian", "random", "grid"})


@dataclass(frozen=True)
class HyperoptConfig:
    model_type: Literal["ridge", "elasticnet", "random_forest", "lgbm", "xgboost"] = "ridge"
    method: str = "bayesian"
    n_trials: int = 50
    timeout: int | None = None
    n_jobs: int = 1
    random_state: int = 42
    pruner: str = "none"
    n_startup_trials: int = 10
    n_ei_candidates: int = 24

    def __post_init__(self) -> None:
        valid_models = {"ridge", "elasticnet", "random_forest", "lgbm", "xgboost"}
        if self.model_type not in valid_models:
            raise ValueError(f"unsupported model_type: {self.model_type}; choose from {valid_models}")
        if self.method not in SEARCH_METHODS:
            raise ValueError(f"unsupported method: {self.method}; choose from {SEARCH_METHODS}")
        if self.pruner not in ("median", "none"):
            raise ValueError(f"unsupported pruner: {self.pruner}; choose from 'median' or 'none'")


def run_hyperparameter_optimization(
    research_frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    config: HyperoptConfig | None = None,
    base_config: BaselineModelConfig | None = None,
    backtest_config: BacktestConfig | None = None,
) -> dict[str, Any]:
    """Run hyperparameter optimization using Optuna.

    Returns a dictionary with keys:
        best_params : dict of best hyperparameters found.
        best_value : best objective value (mean validation Sharpe).
        study : the Optuna study object (for further analysis).
        trials_frame : DataFrame with per-trial details.
        config : the HyperoptConfig used.
        warnings : list of warning messages.
    """
    if not HYPEROPT_AVAILABLE:
        raise ImportError(
            "optuna is required for hyperparameter optimization. "
            "Install it with: pip install signalforge[optimize]"
        )

    cfg = config or HyperoptConfig()
    bcfg = base_config or BaselineModelConfig()
    result: dict[str, Any] = {"config": asdict(cfg)}

    frame = research_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=[*feature_columns, bcfg.target_col, bcfg.realized_return_col])
    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("no model-ready rows remain after dropping null features and targets")

    pruner: optuna.pruners.BasePruner | None = None
    if cfg.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=cfg.n_startup_trials,
            n_warmup_steps=5,
        )

    sampler: optuna.samplers.BaseSampler
    if cfg.method == "bayesian":
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=cfg.n_startup_trials,
            n_ei_candidates=cfg.n_ei_candidates,
            seed=cfg.random_state,
        )
    elif cfg.method == "random":
        sampler = optuna.samplers.RandomSampler(seed=cfg.random_state)
    else:
        sampler = optuna.samplers.GridSampler(
            search_space=_build_grid_search_space(cfg.model_type)
        )

    objective = _make_objective(
        frame=frame,
        feature_columns=feature_columns,
        base_config=bcfg,
        backtest_config=backtest_config,
        hyperopt_config=cfg,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"signalforge_{cfg.model_type}_{cfg.method}",
    )

    study.optimize(
        objective,
        n_trials=cfg.n_trials if cfg.method != "grid" else None,
        timeout=cfg.timeout,
        n_jobs=cfg.n_jobs,
        show_progress_bar=False,
    )

    result["best_params"] = study.best_params
    result["best_value"] = study.best_value
    result["study"] = study
    result["trials_frame"] = _trials_to_frame(study)
    result["warnings"] = _collect_warnings(study)

    return result


def _make_objective(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    base_config: BaselineModelConfig,
    backtest_config: BacktestConfig | None,
    hyperopt_config: HyperoptConfig,
) -> Callable[[optuna.Trial], float]:
    """Create the Optuna objective function for a given model type."""

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, hyperopt_config.model_type)
        bcfg = _build_trial_config(base_config, hyperopt_config.model_type, params)

        split_sharpes: list[float] = []
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

            model = _build_model(bcfg)
            model.fit(train.loc[:, feature_columns], train[bcfg.target_col])

            pred = pd.Series(
                model.predict(validation.loc[:, feature_columns]),
                index=validation.index,
            )
            pred_cols = ["date", "symbol", bcfg.target_col, bcfg.realized_return_col]
            pred_frame = validation.loc[:, pred_cols].copy()
            pred_frame["prediction"] = pred

            bt = long_short_daily_returns(
                pred_frame.rename(
                    columns={"prediction": "score", bcfg.realized_return_col: "forward_return"}
                ),
                config=backtest_config,
            )
            net_returns = bt["net_return"] if "net_return" in bt else pd.Series(dtype=float)
            sharpe = sharpe_ratio(net_returns) if len(net_returns) > 1 else -1.0
            split_sharpes.append(sharpe)

            if hyperopt_config.pruner == "median" and len(split_sharpes) >= 3:
                trial.report(np.mean(split_sharpes), split_id)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        if not split_sharpes:
            return -1.0
        return float(np.mean(split_sharpes))

    return objective


def _sample_params(trial: optuna.Trial, model_type: str) -> dict[str, Any]:
    """Sample hyperparameters from the search space for a given model type."""
    if model_type == "ridge":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True),
        }

    if model_type == "elasticnet":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }

    if model_type == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100, step=5),
        }

    if model_type == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "num_leaves": trial.suggest_int("num_leaves", 10, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }

    if model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0, log=True),
        }

    raise ValueError(f"unsupported model_type: {model_type}")


def _build_grid_search_space(model_type: str) -> dict[str, list[Any]]:
    """Build a fixed grid for grid search over the most important hyperparameters."""
    if model_type == "ridge":
        return {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    if model_type == "elasticnet":
        return {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        }
    if model_type == "random_forest":
        return {
            "n_estimators": [100, 200, 500],
            "max_depth": [5, 10, 15],
            "min_samples_leaf": [10, 25, 50],
        }
    if model_type == "lgbm":
        return {
            "n_estimators": [100, 200],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.05, 0.1],
            "max_depth": [5, 10],
        }
    if model_type == "xgboost":
        return {
            "n_estimators": [100, 200],
            "max_depth": [5, 10],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
        }
    raise ValueError(f"unsupported model_type: {model_type}")


def _build_trial_config(
    base: BaselineModelConfig,
    model_type: str,
    params: dict[str, Any],
) -> BaselineModelConfig:
    """Build a BaselineModelConfig with trial-specific hyperparameters."""
    kwargs = asdict(base)
    kwargs.update(params)
    kwargs["model_type"] = model_type
    return BaselineModelConfig(**kwargs)


def _trials_to_frame(study: Any) -> pd.DataFrame:
    """Convert an Optuna study to a DataFrame of trial results."""
    import optuna  # noqa: F811

    rows = []
    for trial in study.trials:
        rows.append({
            "number": trial.number,
            "value": trial.value,
            "state": str(trial.state),
            "duration_seconds": (
                trial.duration.total_seconds() if trial.duration else None
            ),
            **trial.params,
        })
    return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)


def _collect_warnings(study: Any) -> list[str]:
    """Collect any warnings from the study (e.g., pruned trials)."""
    pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
    failed = sum(1 for t in study.trials if t.state.name == "FAIL")
    warnings: list[str] = []
    if pruned:
        warnings.append(f"{pruned} trial(s) were pruned")
    if failed:
        warnings.append(f"{failed} trial(s) failed")
    if not study.trials:
        warnings.append("no completed trials")
    return warnings


def best_trial_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Extract a human-readable summary of the best trial from optimization results."""
    return {
        "model_type": result["config"]["model_type"],
        "method": result["config"]["method"],
        "n_trials": len(result["trials_frame"]),
        "best_value": result["best_value"],
        "best_params": result["best_params"],
    }


def trial_importance_data(result: dict[str, Any]) -> dict[str, float] | None:
    """Return hyperparameter importance scores if Optuna calculated them."""
    study = result.get("study")
    if study is None:
        return None
    try:
        import optuna  # noqa: F811

        importances = optuna.importance.get_param_importances(study)
        return importances
    except Exception:
        return None
