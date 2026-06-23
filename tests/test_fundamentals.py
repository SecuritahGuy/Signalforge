from pathlib import Path

import pandas as pd
import pytest

from signalforge.fundamentals import (
    enrich_research_frame_with_fundamentals,
    latest_fundamentals_as_of,
    load_fundamentals_csv,
    normalize_fundamentals,
    quality_growth_eligible_rows,
    quality_growth_required_fields_available,
)


def test_load_fundamentals_csv_normalizes_symbols_dates_and_aliases(tmp_path: Path):
    csv_path = tmp_path / "fundamentals.csv"
    csv_path.write_text(
        "symbol,as_of_date,market_cap,revenue_growth_yoy,eps_growth_yoy,"
        "gross_margin,debt_to_equity,return_on_equity,forward_pe,source\n"
        "aapl,2024-03-31,3000000000000,0.08,0.12,0.45,1.5,1.2,28,yahoo\n"
    )

    fundamentals = load_fundamentals_csv(csv_path)

    assert fundamentals.loc[0, "symbol"] == "AAPL"
    assert fundamentals.loc[0, "as_of_date"] == pd.Timestamp("2024-03-31")
    assert "revenue_growth" in fundamentals.columns
    assert "eps_growth" in fundamentals.columns
    assert fundamentals.loc[0, "market_cap"] == 3_000_000_000_000


def test_latest_fundamentals_as_of_selects_latest_available_row_per_symbol():
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "as_of_date": ["2024-01-31", "2024-06-30", "2024-02-15", "2024-08-01"],
            "market_cap": [100, 120, 200, 220],
        }
    )

    latest = latest_fundamentals_as_of(fundamentals, "2024-07-01")

    assert latest.sort_values("symbol")["market_cap"].tolist() == [120, 200]


def test_enrich_research_frame_joins_latest_fundamentals_by_symbol_and_date():
    research_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-02-01", "2024-07-01", "2024-07-01"]),
            "symbol": ["aaa", "AAA", "BBB"],
            "adj_close": [10.0, 12.0, 20.0],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "as_of_date": ["2024-01-31", "2024-06-30", "2024-08-01"],
            "market_cap": [100.0, 120.0, 220.0],
            "revenue_growth": [0.10, 0.20, 0.30],
            "source": ["manual", "manual", "manual"],
        }
    )

    enriched = enrich_research_frame_with_fundamentals(research_frame, fundamentals)

    assert enriched["symbol"].tolist() == ["AAA", "AAA", "BBB"]
    assert enriched.loc[0, "market_cap"] == 100.0
    assert enriched.loc[1, "market_cap"] == 120.0
    assert pd.isna(enriched.loc[2, "market_cap"])
    assert enriched.loc[0, "fundamentals_as_of_date"] == pd.Timestamp("2024-01-31")
    assert enriched.loc[1, "fundamentals_as_of_date"] == pd.Timestamp("2024-06-30")
    assert pd.isna(enriched.loc[2, "fundamentals_as_of_date"])
    assert "fundamentals_source" in enriched.columns


def test_quality_growth_requirements_are_grouped_and_row_level():
    frame = pd.DataFrame(
        {
            "market_cap": [1_000_000_000, 1_000_000_000],
            "revenue_growth": [0.2, 0.2],
            "eps_growth": [0.3, 0.3],
            "gross_margin": [0.5, 0.5],
            "return_on_equity": [0.2, pd.NA],
            "debt_to_equity": [0.4, 0.4],
            "forward_pe": [22.0, 22.0],
        }
    )

    assert quality_growth_required_fields_available(frame)
    assert quality_growth_eligible_rows(frame).tolist() == [True, False]


def test_normalize_fundamentals_requires_symbol_and_as_of_date():
    with pytest.raises(KeyError, match="as_of_date"):
        normalize_fundamentals(pd.DataFrame({"symbol": ["AAA"]}))
