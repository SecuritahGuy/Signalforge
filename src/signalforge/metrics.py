from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(returns: pd.Series, *, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    volatility = clean.std(ddof=1)
    if volatility == 0 or np.isnan(volatility):
        return np.nan
    return float(clean.mean() / volatility * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, *, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    clean = returns.dropna()
    downside = clean[clean < 0]
    if clean.empty or downside.empty:
        return np.nan
    downside_volatility = downside.std(ddof=1)
    if downside_volatility == 0 or np.isnan(downside_volatility):
        return np.nan
    return float(clean.mean() / downside_volatility * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    clean = returns.fillna(0.0)
    equity = (1.0 + clean).cumprod()
    drawdown = equity.div(equity.cummax()).sub(1.0)
    return float(drawdown.min())


def hit_rate(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    return float((clean > 0).mean())


def information_coefficient(predictions: pd.Series, realized: pd.Series) -> float:
    aligned = pd.concat([predictions, realized], axis=1).dropna()
    if aligned.empty:
        return np.nan
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"))
