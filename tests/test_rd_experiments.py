import pandas as pd

from scripts.run_rd_experiments import (
    PortfolioRuleSpec,
    render_rd_summary,
    run_portfolio_rule_experiments,
)


def test_run_portfolio_rule_experiments_compares_rule_outputs():
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-22"] * 3),
            "symbol": ["A", "B", "C", "A", "B", "C"],
            "score": [0.03, 0.02, -0.01, 0.04, 0.01, -0.02],
            "fwd_20d_exec_return": [0.10, -0.05, 0.01, 0.08, -0.02, 0.01],
            "next_open": [100.0, 50.0, 25.0, 100.0, 50.0, 25.0],
            "exit_close_20d": [110.0, 47.5, 25.25, 108.0, 49.0, 25.25],
            "avg_dollar_volume_20d": [1_000_000.0] * 6,
        }
    )
    rules = (
        PortfolioRuleSpec("loose", long_fraction=1.0, max_position_weight=0.5, min_score=0.0),
        PortfolioRuleSpec("strict", long_fraction=1.0, max_position_weight=0.5, min_score=0.03),
    )

    result = run_portfolio_rule_experiments(
        predictions,
        initial_capital=1_000,
        rules=rules,
    )

    assert set(result["rule"]) == {"loose", "strict"}
    assert result["filled_trades"].min() > 0
    assert "ending_capital" in result


def test_render_rd_summary_includes_ablation_and_rule_tables():
    ablation = pd.DataFrame(
        {
            "experiment": ["exp"],
            "feature_set": ["momentum"],
            "model": ["ridge"],
            "risk_backtest_sharpe": [1.2],
            "risk_backtest_max_drawdown": [-0.1],
            "ic_mean": [0.05],
            "positive_ic_splits": [4],
        }
    )
    rules = pd.DataFrame(
        {
            "rule": ["baseline"],
            "ending_capital": [1_100.0],
            "total_return": [0.1],
            "sharpe": [1.0],
            "max_drawdown": [-0.05],
            "filled_trades": [10],
            "skipped_trades": [2],
        }
    )

    report = render_rd_summary(ablation, rules)

    assert "Feature Ablation" in report
    assert "Portfolio Rules" in report
    assert "baseline" in report
