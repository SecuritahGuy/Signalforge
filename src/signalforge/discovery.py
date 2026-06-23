from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from signalforge.fundamentals import quality_growth_eligible_rows

DISCOVERY_COLUMNS = (
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "volatility_60d",
    "volume_change_5d",
    "avg_dollar_volume_20d",
    "sector_rank_momentum_20d",
    "sector_rank_volatility_20d",
    "stock_minus_market_return_20d",
    "stock_minus_sector_return_20d",
    "relative_volume_20d",
    "price_above_sma_50",
    "price_above_sma_200",
    "distance_from_52w_high",
    "drawdown_60d",
    "market_cap",
    "revenue_growth",
    "eps_growth",
    "gross_margin",
    "operating_margin",
    "free_cash_flow_yield",
    "debt_to_equity",
    "return_on_equity",
    "forward_pe",
    "price_to_sales",
    "days_to_earnings",
)

WATCHLIST_SCORE_COLUMNS = {
    "momentum_breakouts": "momentum_score",
    "sector_leaders": "sector_strength_score",
    "volume_anomalies": "attention_score",
    "quality_growth": "quality_score",
    "value_recoveries": "value_recovery_score",
}


@dataclass(frozen=True)
class DiscoveryConfig:
    top_n: int = 25
    price_col: str = "adj_close"
    min_price: float = 5.0
    min_avg_dollar_volume_20d: float = 5_000_000.0
    min_market_cap: float | None = 300_000_000.0
    earnings_blackout_days: int = 1
    max_volatility_20d: float | None = None
    excluded_categories: tuple[str, ...] = ("benchmark", "etf")
    excluded_security_types: tuple[str, ...] = ("ETF", "Fund")
    discovery_score_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "momentum_score": 0.30,
            "quality_score": 0.20,
            "value_recovery_score": 0.20,
            "attention_score": 0.15,
            "sector_strength_score": 0.10,
            "catalyst_score": 0.05,
        }
    )


@dataclass(frozen=True)
class DiscoveryResult:
    as_of_date: pd.Timestamp
    candidates: pd.DataFrame
    watchlists: dict[str, pd.DataFrame]
    exclusions: pd.DataFrame


def run_stock_discovery(
    research_frame: pd.DataFrame,
    *,
    universe: pd.DataFrame | None = None,
    as_of_date: str | pd.Timestamp | None = None,
    existing_watchlist: Iterable[str] | pd.DataFrame | None = None,
    config: DiscoveryConfig | None = None,
) -> DiscoveryResult:
    """Run the stock discovery funnel for one as-of date.

    The input should be a broad, model-ready feature frame. Existing watchlist symbols
    are excluded so the output focuses on names newly worth reviewing.
    """
    cfg = config or DiscoveryConfig()
    _validate_config(cfg)

    frame = _normalize_frame(research_frame)
    frame = _attach_universe_metadata(frame, universe)
    frame = _add_discovery_derived_features(frame, price_col=cfg.price_col)
    resolved_date = _resolve_as_of_date(frame, as_of_date)
    snapshot = frame.loc[frame["date"] == resolved_date].copy()
    if snapshot.empty:
        raise ValueError(f"no discovery rows available for {resolved_date.date()}")
    _validate_market_cap_filter(snapshot, config=cfg)

    watchlist_symbols = _normalize_symbols(existing_watchlist)
    candidates, exclusions = apply_discovery_filters(
        snapshot,
        existing_watchlist=watchlist_symbols,
        config=cfg,
    )
    scored = score_discovery_candidates(candidates, config=cfg)
    if not scored.empty:
        scored["why_flagged"] = [
            _explain_candidate(row) for _, row in scored.iterrows()
        ]
        scored["risk_flags"] = [
            _risk_flags(row, config=cfg) for _, row in scored.iterrows()
        ]

    return DiscoveryResult(
        as_of_date=resolved_date,
        candidates=scored,
        watchlists=build_discovery_watchlists(scored, config=cfg),
        exclusions=exclusions,
    )


def apply_discovery_filters(
    snapshot: pd.DataFrame,
    *,
    existing_watchlist: set[str] | None = None,
    config: DiscoveryConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply liquidity, tradability, and already-known watchlist exclusions."""
    cfg = config or DiscoveryConfig()
    _validate_market_cap_filter(snapshot, config=cfg)
    existing = existing_watchlist or set()
    kept_rows = []
    excluded_rows = []

    for _, row in snapshot.iterrows():
        reasons = _exclusion_reasons(row, existing_watchlist=existing, config=cfg)
        if reasons:
            excluded = _base_output_row(row)
            excluded["exclusion_reasons"] = "; ".join(reasons)
            excluded_rows.append(excluded)
        else:
            kept_rows.append(row)

    candidates = pd.DataFrame(kept_rows).reset_index(drop=True)
    exclusions = pd.DataFrame(excluded_rows)
    return candidates, exclusions


def score_discovery_candidates(
    candidates: pd.DataFrame,
    *,
    config: DiscoveryConfig | None = None,
) -> pd.DataFrame:
    """Add interpretable discovery lane scores to eligible candidates."""
    if candidates.empty:
        return candidates.copy()

    cfg = config or DiscoveryConfig()
    frame = candidates.copy().reset_index(drop=True)
    frame = _add_score_helper_columns(frame)
    quality_eligible = quality_growth_eligible_rows(frame)

    score_specs = {
        "momentum_score": (
            ("sector_rank_return_20d", 0.25, True),
            ("sector_rank_momentum_20d", 0.20, True),
            ("return_60d", 0.20, True),
            ("stock_minus_sector_return_20d", 0.15, True),
            ("stock_minus_market_return_20d", 0.10, True),
            ("price_above_sma_50", 0.05, True),
            ("distance_from_52w_high", 0.05, True),
        ),
        "sector_strength_score": (
            ("sector_return_20d", 0.30, True),
            ("sector_return_60d", 0.25, True),
            ("sector_rank_return_60d", 0.20, True),
            ("stock_minus_sector_return_20d", 0.15, True),
            ("stock_minus_sector_return_60d", 0.10, True),
        ),
        "attention_score": (
            ("relative_volume_20d", 0.35, True),
            ("relative_volume_5d", 0.20, True),
            ("volume_zscore_20d", 0.20, True),
            ("volume_change_5d", 0.15, True),
            ("abs_return_1d", 0.10, True),
        ),
        "quality_score": (
            ("revenue_growth", 0.20, True),
            ("eps_growth", 0.18, True),
            ("gross_margin", 0.10, True),
            ("operating_margin", 0.10, True),
            ("return_on_equity", 0.16, True),
            ("free_cash_flow_yield", 0.10, True),
            ("debt_to_equity", 0.10, False),
            ("forward_pe", 0.03, False),
            ("price_to_sales", 0.03, False),
        ),
        "value_recovery_score": (
            ("drawdown_60d", 0.20, False),
            ("return_5d", 0.15, True),
            ("price_above_sma_20", 0.10, True),
            ("price_above_sma_50", 0.10, True),
            ("relative_volume_20d", 0.10, True),
            ("earnings_yield", 0.15, True),
            ("free_cash_flow_yield", 0.10, True),
            ("forward_pe", 0.10, False),
        ),
        "catalyst_score": (
            ("earnings_window_score", 0.25, True),
            ("earnings_surprise_last_qtr", 0.20, True),
            ("analyst_revision_30d", 0.20, True),
            ("analyst_upgrade_count_30d", 0.15, True),
            ("news_sentiment_7d", 0.10, True),
            ("news_count_7d", 0.10, True),
        ),
    }

    for score_col, components in score_specs.items():
        score, input_count = _weighted_rank_score(frame, components)
        frame[score_col] = score
        frame[f"{score_col}_inputs"] = input_count
        if score_col == "quality_score":
            frame["quality_score_eligible"] = quality_eligible
            frame.loc[~quality_eligible, "quality_score"] = 50.0
            frame.loc[~quality_eligible, "quality_score_inputs"] = 0

    total_weight = 0.0
    composite = pd.Series(0.0, index=frame.index)
    for column, weight in cfg.discovery_score_weights.items():
        if column in frame.columns:
            composite = composite.add(frame[column] * weight)
            total_weight += weight
    frame["discovery_score"] = composite.div(total_weight) if total_weight else 50.0

    return frame.sort_values("discovery_score", ascending=False).reset_index(drop=True)


def build_discovery_watchlists(
    scored_candidates: pd.DataFrame,
    *,
    config: DiscoveryConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build the five v1 discovery watchlists from scored candidates."""
    cfg = config or DiscoveryConfig()
    watchlists: dict[str, pd.DataFrame] = {}
    for bucket, score_col in WATCHLIST_SCORE_COLUMNS.items():
        if scored_candidates.empty or score_col not in scored_candidates.columns:
            watchlists[bucket] = _empty_watchlist_frame()
            continue

        input_count_col = f"{score_col}_inputs"
        bucket_candidates = scored_candidates
        if bucket == "quality_growth" and "quality_score_eligible" in bucket_candidates.columns:
            bucket_candidates = bucket_candidates.loc[bucket_candidates["quality_score_eligible"]]
        elif input_count_col in bucket_candidates.columns:
            bucket_candidates = bucket_candidates.loc[bucket_candidates[input_count_col] > 0]
        if bucket_candidates.empty:
            watchlists[bucket] = _empty_watchlist_frame()
            continue

        ranked = bucket_candidates.sort_values(
            [score_col, "discovery_score", "avg_dollar_volume_20d", "symbol"],
            ascending=[False, False, False, True],
        ).head(cfg.top_n)
        ranked = ranked.copy()
        ranked["discovery_bucket"] = bucket
        columns = [
            "date",
            "symbol",
            "name",
            "sector",
            "industry",
            "discovery_bucket",
            "discovery_score",
            score_col,
            "momentum_score",
            "sector_strength_score",
            "attention_score",
            "quality_score",
            "quality_score_eligible",
            "value_recovery_score",
            "catalyst_score",
            "adj_close",
            "market_cap",
            "avg_dollar_volume_20d",
            "return_20d",
            "return_60d",
            "stock_minus_market_return_20d",
            "stock_minus_sector_return_20d",
            "relative_volume_20d",
            "price_above_sma_50",
            "price_above_sma_200",
            "distance_from_52w_high",
            "drawdown_60d",
            "forward_pe",
            "free_cash_flow_yield",
            "days_to_earnings",
            "why_flagged",
            "risk_flags",
        ]
        selected_columns = []
        for column in columns:
            if column in ranked.columns and column not in selected_columns:
                selected_columns.append(column)
        watchlists[bucket] = ranked.loc[:, selected_columns]
    return watchlists


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"research_frame is missing required columns: {sorted(missing)}")

    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized["symbol"] = normalized["symbol"].astype(str).str.upper()
    return normalized.sort_values(["symbol", "date"]).reset_index(drop=True)


def _attach_universe_metadata(
    frame: pd.DataFrame,
    universe: pd.DataFrame | None,
) -> pd.DataFrame:
    if universe is None:
        return frame
    if "symbol" not in universe.columns:
        raise KeyError("universe is missing required column: 'symbol'")

    metadata = universe.copy()
    metadata["symbol"] = metadata["symbol"].astype(str).str.upper()
    metadata_columns = [
        column
        for column in (
            "name",
            "category",
            "sector",
            "industry",
            "country",
            "exchange",
            "security_type",
            "market_cap",
        )
        if column in metadata.columns and column not in frame.columns
    ]
    if not metadata_columns:
        return frame
    return frame.merge(metadata[["symbol", *metadata_columns]], on="symbol", how="left")


def _add_discovery_derived_features(frame: pd.DataFrame, *, price_col: str) -> pd.DataFrame:
    if price_col not in frame.columns:
        return frame

    enriched = frame.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
    grouped = enriched.groupby("symbol", sort=False)

    if "return_1d" not in enriched.columns:
        enriched["return_1d"] = grouped[price_col].pct_change()

    for window in (20, 50, 200):
        column = f"price_above_sma_{window}"
        if column not in enriched.columns:
            sma = grouped[price_col].transform(
                lambda series, window=window: series.rolling(window, min_periods=window).mean()
            )
            enriched[column] = enriched[price_col].div(sma).sub(1.0)

    if "distance_from_52w_high" not in enriched.columns:
        rolling_high_252 = grouped[price_col].transform(
            lambda series: series.rolling(252, min_periods=252).max()
        )
        enriched["distance_from_52w_high"] = enriched[price_col].div(rolling_high_252).sub(1.0)

    if "drawdown_from_52w_high" not in enriched.columns:
        enriched["drawdown_from_52w_high"] = enriched["distance_from_52w_high"]

    if "volume" in enriched.columns:
        if "relative_volume_5d" not in enriched.columns:
            avg_volume_5d = grouped["volume"].transform(
                lambda series: series.rolling(5, min_periods=5).mean()
            )
            enriched["relative_volume_5d"] = enriched["volume"].div(avg_volume_5d)
        if "volume_zscore_20d" not in enriched.columns:
            enriched["volume_zscore_20d"] = grouped["volume"].transform(_rolling_zscore_20d)

    if "downside_volatility_20d" not in enriched.columns:
        negative_returns = enriched["return_1d"].where(enriched["return_1d"] < 0)
        enriched["downside_volatility_20d"] = negative_returns.groupby(
            enriched["symbol"]
        ).transform(
            lambda series: series.rolling(20, min_periods=20).std(),
        )

    if "max_drawdown_60d" not in enriched.columns:
        if "drawdown_60d" in enriched.columns:
            enriched["max_drawdown_60d"] = enriched["drawdown_60d"]
        else:
            rolling_high_60 = grouped[price_col].transform(
                lambda series: series.rolling(60, min_periods=60).max()
            )
            enriched["max_drawdown_60d"] = enriched[price_col].div(rolling_high_60).sub(1.0)

    return enriched


def _rolling_zscore_20d(series: pd.Series) -> pd.Series:
    rolling = series.rolling(20, min_periods=20)
    std = rolling.std().replace(0.0, np.nan)
    return series.sub(rolling.mean()).div(std)


def _resolve_as_of_date(
    frame: pd.DataFrame,
    raw_date: str | pd.Timestamp | None,
) -> pd.Timestamp:
    if raw_date is None:
        return pd.Timestamp(frame["date"].max())
    as_of_date = pd.Timestamp(raw_date)
    if as_of_date not in set(frame["date"]):
        raise ValueError(f"as-of date {as_of_date.date()} is not present in research frame")
    return as_of_date


def _normalize_symbols(symbols: Iterable[str] | pd.DataFrame | None) -> set[str]:
    if symbols is None:
        return set()
    if isinstance(symbols, pd.DataFrame):
        if "symbol" not in symbols.columns:
            raise KeyError("existing_watchlist is missing required column: 'symbol'")
        raw_symbols = symbols["symbol"]
    elif isinstance(symbols, str):
        raw_symbols = symbols.split(",")
    else:
        raw_symbols = symbols
    return {str(symbol).strip().upper() for symbol in raw_symbols if str(symbol).strip()}


def _exclusion_reasons(
    row: pd.Series,
    *,
    existing_watchlist: set[str],
    config: DiscoveryConfig,
) -> list[str]:
    reasons = []
    symbol = str(row.get("symbol", "")).upper()
    if symbol in existing_watchlist:
        reasons.append("already_in_watchlist")

    category = str(row.get("category", "")).strip().lower()
    if category and category in {item.lower() for item in config.excluded_categories}:
        reasons.append(f"excluded_category:{category}")

    security_type = str(row.get("security_type", "")).strip().lower()
    if security_type and security_type in {
        item.lower() for item in config.excluded_security_types
    }:
        reasons.append(f"excluded_security_type:{security_type}")

    price = _numeric_value(row.get(config.price_col, row.get("close", np.nan)))
    if pd.isna(price):
        reasons.append("missing_price")
    elif price < config.min_price:
        reasons.append("price_below_minimum")

    avg_dollar_volume = _numeric_value(row.get("avg_dollar_volume_20d", np.nan))
    if pd.isna(avg_dollar_volume):
        reasons.append("missing_avg_dollar_volume_20d")
    elif avg_dollar_volume < config.min_avg_dollar_volume_20d:
        reasons.append("avg_dollar_volume_20d_below_minimum")

    if config.min_market_cap is not None and "market_cap" in row.index:
        market_cap = _numeric_value(row.get("market_cap"))
        if pd.isna(market_cap):
            reasons.append("missing_market_cap")
        elif market_cap < config.min_market_cap:
            reasons.append("market_cap_below_minimum")

    days_to_earnings = _numeric_value(row.get("days_to_earnings", np.nan))
    if (
        config.earnings_blackout_days >= 0
        and pd.notna(days_to_earnings)
        and 0 <= days_to_earnings <= config.earnings_blackout_days
    ):
        reasons.append("earnings_blackout")

    volatility_20d = _numeric_value(row.get("volatility_20d", np.nan))
    if (
        config.max_volatility_20d is not None
        and pd.notna(volatility_20d)
        and volatility_20d > config.max_volatility_20d
    ):
        reasons.append("extreme_volatility")

    return reasons


def _base_output_row(row: pd.Series) -> dict:
    return {
        "date": row.get("date"),
        "symbol": row.get("symbol"),
        "name": row.get("name", ""),
        "sector": row.get("sector", ""),
        "industry": row.get("industry", ""),
    }


def _add_score_helper_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "revenue_growth_yoy" in enriched.columns and "revenue_growth" not in enriched.columns:
        enriched["revenue_growth"] = enriched["revenue_growth_yoy"]
    if "eps_growth_yoy" in enriched.columns and "eps_growth" not in enriched.columns:
        enriched["eps_growth"] = enriched["eps_growth_yoy"]
    if (
        "free_cash_flow" in enriched.columns
        and "market_cap" in enriched.columns
        and "free_cash_flow_yield" not in enriched.columns
    ):
        market_cap = pd.to_numeric(enriched["market_cap"], errors="coerce")
        free_cash_flow = pd.to_numeric(enriched["free_cash_flow"], errors="coerce")
        enriched["free_cash_flow_yield"] = free_cash_flow.div(market_cap.where(market_cap > 0))
    if "return_1d" in enriched.columns:
        enriched["abs_return_1d"] = pd.to_numeric(enriched["return_1d"], errors="coerce").abs()
    if "forward_pe" in enriched.columns and "earnings_yield" not in enriched.columns:
        forward_pe = pd.to_numeric(enriched["forward_pe"], errors="coerce")
        enriched["earnings_yield"] = 1.0 / forward_pe.where(forward_pe > 0)
    if "days_to_earnings" in enriched.columns and "earnings_window_score" not in enriched.columns:
        days = pd.to_numeric(enriched["days_to_earnings"], errors="coerce")
        enriched["earnings_window_score"] = np.where(
            days.between(2, 30),
            (31.0 - days) / 29.0,
            np.nan,
        )
    return enriched


def _validate_market_cap_filter(snapshot: pd.DataFrame, *, config: DiscoveryConfig) -> None:
    if config.min_market_cap is None or "market_cap" in snapshot.columns:
        return
    raise ValueError(
        "market_cap is required when min_market_cap filtering is enabled. "
        "Enrich the research frame with fundamentals or pass --no-market-cap-filter."
    )


def _weighted_rank_score(
    frame: pd.DataFrame,
    components: tuple[tuple[str, float, bool], ...],
) -> tuple[pd.Series, pd.Series]:
    weighted_sum = pd.Series(0.0, index=frame.index)
    weight_sum = pd.Series(0.0, index=frame.index)
    input_count = pd.Series(0, index=frame.index, dtype="int64")

    for column, weight, higher_is_better in components:
        if column not in frame.columns:
            continue
        values = _numeric_series(frame[column])
        valid = values.notna()
        ranks = _percentile_rank(values, higher_is_better=higher_is_better)
        weighted_sum = weighted_sum.add(ranks.where(valid, 0.0) * weight)
        weight_sum = weight_sum.add(valid.astype(float) * weight)
        input_count = input_count.add(valid.astype(int))

    score = weighted_sum.div(weight_sum).where(weight_sum > 0, 0.5).mul(100.0)
    return score, input_count


def _percentile_rank(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    values = _numeric_series(series)
    valid_values = values.dropna()
    if valid_values.empty:
        return pd.Series(np.nan, index=series.index)
    if valid_values.nunique() <= 1:
        return pd.Series(np.where(values.notna(), 0.5, np.nan), index=series.index)
    ranked_values = values if higher_is_better else values.mul(-1.0)
    return ranked_values.rank(pct=True, method="average")


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _numeric_value(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or np.isinf(numeric):
        return np.nan
    return float(numeric)


def _explain_candidate(row: pd.Series) -> str:
    reasons: list[str] = []
    if _numeric_value(row.get("momentum_score", np.nan)) >= 80:
        reasons.append("strong cross-sectional momentum score")
    if _numeric_value(row.get("sector_strength_score", np.nan)) >= 80:
        reasons.append("sector-relative strength is ranking well")
    if _numeric_value(row.get("attention_score", np.nan)) >= 80:
        reasons.append("volume or price attention signal is elevated")
    if (
        _numeric_value(row.get("quality_score", np.nan)) >= 80
        and bool(row.get("quality_score_eligible", False))
    ):
        reasons.append("quality and growth inputs rank well")
    if _numeric_value(row.get("value_recovery_score", np.nan)) >= 80:
        reasons.append("value or recovery setup ranks well")

    sector_alpha = _numeric_value(row.get("stock_minus_sector_return_20d", np.nan))
    if pd.notna(sector_alpha) and sector_alpha > 0:
        reasons.append(f"20-day return is {_format_pct(sector_alpha)} above sector average")

    relative_volume = _numeric_value(row.get("relative_volume_20d", np.nan))
    if pd.notna(relative_volume) and relative_volume >= 1.5:
        reasons.append(f"volume is {relative_volume:.1f}x its 20-day average")

    price_above_sma_50 = _numeric_value(row.get("price_above_sma_50", np.nan))
    if pd.notna(price_above_sma_50) and price_above_sma_50 > 0:
        reasons.append("price is above its 50-day moving average")

    distance_from_high = _numeric_value(row.get("distance_from_52w_high", np.nan))
    if pd.notna(distance_from_high) and distance_from_high >= -0.05:
        reasons.append("price is within 5% of its 52-week high")

    if not reasons:
        reasons.append("ranked well enough across available discovery inputs")
    return "; ".join(reasons[:5])


def _risk_flags(row: pd.Series, *, config: DiscoveryConfig) -> str:
    risks: list[str] = []
    days_to_earnings = _numeric_value(row.get("days_to_earnings", np.nan))
    if pd.notna(days_to_earnings) and 0 <= days_to_earnings <= 7:
        risks.append(f"earnings in {int(days_to_earnings)} days")

    volatility = _numeric_value(row.get("volatility_20d", np.nan))
    if pd.notna(volatility) and volatility >= 0.05:
        risks.append(f"20-day daily volatility is {_format_pct(volatility)}")

    avg_dollar_volume = _numeric_value(row.get("avg_dollar_volume_20d", np.nan))
    if (
        pd.notna(avg_dollar_volume)
        and config.min_avg_dollar_volume_20d > 0
        and avg_dollar_volume < config.min_avg_dollar_volume_20d * 2
    ):
        risks.append("liquidity is close to the discovery minimum")

    beta = _numeric_value(row.get("beta_60d", np.nan))
    if pd.notna(beta) and beta >= 2.0:
        risks.append(f"beta is elevated at {beta:.1f}")

    price_above_sma_200 = _numeric_value(row.get("price_above_sma_200", np.nan))
    if pd.notna(price_above_sma_200) and price_above_sma_200 < 0:
        risks.append("price is below its 200-day moving average")

    forward_pe = _numeric_value(row.get("forward_pe", np.nan))
    if pd.notna(forward_pe) and forward_pe >= 50:
        risks.append(f"forward P/E is high at {forward_pe:.1f}")

    return "; ".join(risks)


def _format_pct(value: float) -> str:
    return f"{value:.1%}"


def _empty_watchlist_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "symbol",
            "sector",
            "discovery_bucket",
            "discovery_score",
            "why_flagged",
            "risk_flags",
        ]
    )


def _validate_config(config: DiscoveryConfig) -> None:
    if config.top_n <= 0:
        raise ValueError("top_n must be positive")
    if config.min_price < 0:
        raise ValueError("min_price must be non-negative")
    if config.min_avg_dollar_volume_20d < 0:
        raise ValueError("min_avg_dollar_volume_20d must be non-negative")
    if config.min_market_cap is not None and config.min_market_cap < 0:
        raise ValueError("min_market_cap must be non-negative when provided")
