from __future__ import annotations

from pathlib import Path

import pandas as pd

FUNDAMENTAL_COLUMNS = (
    "symbol",
    "as_of_date",
    "market_cap",
    "revenue_growth",
    "eps_growth",
    "gross_margin",
    "operating_margin",
    "free_cash_flow",
    "debt_to_equity",
    "return_on_equity",
    "forward_pe",
    "price_to_sales",
    "peg_ratio",
    "earnings_date",
    "source",
)

FUNDAMENTAL_REQUIRED_COLUMNS = ("symbol", "as_of_date")

FUNDAMENTAL_NUMERIC_COLUMNS = (
    "market_cap",
    "revenue_growth",
    "eps_growth",
    "gross_margin",
    "operating_margin",
    "free_cash_flow",
    "debt_to_equity",
    "return_on_equity",
    "forward_pe",
    "price_to_sales",
    "peg_ratio",
)

FUNDAMENTAL_COLUMN_ALIASES = {
    "revenue_growth_yoy": "revenue_growth",
    "eps_growth_yoy": "eps_growth",
}

QUALITY_GROWTH_REQUIRED_GROUPS = (
    ("market_cap",),
    ("revenue_growth", "revenue_growth_yoy"),
    ("eps_growth", "eps_growth_yoy"),
    ("operating_margin", "gross_margin"),
    ("return_on_equity",),
    ("debt_to_equity",),
    ("forward_pe", "price_to_sales"),
)


def load_fundamentals_csv(path: str | Path) -> pd.DataFrame:
    """Load point-in-time-style symbol fundamentals from CSV."""
    frame = pd.read_csv(path)
    return normalize_fundamentals(frame)


def normalize_fundamentals(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize symbol, dates, aliases, numeric fields, and duplicate rows."""
    normalized = frame.rename(columns=lambda column: str(column).strip()).copy()
    for alias, canonical in FUNDAMENTAL_COLUMN_ALIASES.items():
        if alias in normalized.columns and canonical not in normalized.columns:
            normalized = normalized.rename(columns={alias: canonical})

    missing = set(FUNDAMENTAL_REQUIRED_COLUMNS).difference(normalized.columns)
    if missing:
        raise KeyError(f"fundamentals are missing required columns: {sorted(missing)}")

    normalized["symbol"] = normalized["symbol"].astype(str).str.strip().str.upper()
    normalized = normalized.loc[normalized["symbol"] != ""].copy()
    normalized["as_of_date"] = pd.to_datetime(normalized["as_of_date"])
    if "earnings_date" in normalized.columns:
        normalized["earnings_date"] = pd.to_datetime(normalized["earnings_date"])

    for column in FUNDAMENTAL_NUMERIC_COLUMNS:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.sort_values(["symbol", "as_of_date"]).drop_duplicates(
        subset=["symbol", "as_of_date"],
        keep="last",
    )
    ordered_columns = [
        column for column in FUNDAMENTAL_COLUMNS if column in normalized.columns
    ]
    extra_columns = [column for column in normalized.columns if column not in ordered_columns]
    return normalized.loc[:, [*ordered_columns, *extra_columns]].reset_index(drop=True)


def latest_fundamentals_as_of(
    fundamentals: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Select the latest fundamental row per symbol available by an as-of date."""
    normalized = normalize_fundamentals(fundamentals)
    cutoff = pd.Timestamp(as_of_date)
    eligible = normalized.loc[normalized["as_of_date"] <= cutoff].copy()
    if eligible.empty:
        return eligible
    return (
        eligible.sort_values(["symbol", "as_of_date"])
        .groupby("symbol", as_index=False, sort=True)
        .tail(1)
        .reset_index(drop=True)
    )


def enrich_research_frame_with_fundamentals(
    research_frame: pd.DataFrame,
    fundamentals: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Join the latest available fundamentals per symbol to each research row."""
    required_research = {date_col, symbol_col}
    missing_research = required_research.difference(research_frame.columns)
    if missing_research:
        raise KeyError(f"research_frame is missing required columns: {sorted(missing_research)}")

    research = research_frame.copy()
    research["__row_order"] = range(len(research))
    research[date_col] = pd.to_datetime(research[date_col])
    research[symbol_col] = research[symbol_col].astype(str).str.strip().str.upper()

    normalized = normalize_fundamentals(fundamentals)
    joinable = normalized.rename(
        columns={
            "as_of_date": "fundamentals_as_of_date",
            "source": "fundamentals_source",
        }
    )
    join_columns = [column for column in joinable.columns if column != symbol_col]
    for column in join_columns:
        if column in research.columns:
            research = research.drop(columns=column)

    enriched_frames = []
    for symbol, symbol_research in research.sort_values([symbol_col, date_col]).groupby(
        symbol_col,
        sort=False,
    ):
        symbol_fundamentals = joinable.loc[joinable[symbol_col] == symbol].sort_values(
            "fundamentals_as_of_date"
        )
        if symbol_fundamentals.empty:
            enriched = symbol_research.copy()
            for column in join_columns:
                enriched[column] = pd.NA
        else:
            enriched = pd.merge_asof(
                symbol_research.sort_values(date_col),
                symbol_fundamentals[[symbol_col, *join_columns]],
                left_on=date_col,
                right_on="fundamentals_as_of_date",
                by=symbol_col,
                direction="backward",
            )
        enriched_frames.append(enriched)

    enriched = pd.concat(enriched_frames, ignore_index=True)
    enriched = enriched.sort_values("__row_order").drop(columns="__row_order")
    return enriched.reset_index(drop=True)


def quality_growth_required_fields_available(frame: pd.DataFrame) -> bool:
    """Return whether a frame contains every minimum quality-growth field group."""
    return all(
        any(column in frame.columns for column in group)
        for group in QUALITY_GROWTH_REQUIRED_GROUPS
    )


def quality_growth_eligible_rows(frame: pd.DataFrame) -> pd.Series:
    """Return a boolean mask for rows with all required quality-growth inputs populated."""
    if frame.empty:
        return pd.Series(dtype=bool, index=frame.index)

    eligible = pd.Series(True, index=frame.index)
    for group in QUALITY_GROWTH_REQUIRED_GROUPS:
        present_columns = [column for column in group if column in frame.columns]
        if not present_columns:
            return pd.Series(False, index=frame.index)
        group_values = frame.loc[:, present_columns].apply(pd.to_numeric, errors="coerce")
        eligible = eligible & group_values.notna().any(axis=1)
    return eligible
