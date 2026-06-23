import pandas as pd
import pytest

from signalforge.discovery import DiscoveryConfig, run_stock_discovery


def test_stock_discovery_excludes_watchlist_and_builds_explainable_lanes():
    frame = pd.DataFrame(
        [
            _candidate(
                "KEEP",
                return_60d=0.60,
                stock_minus_sector_return_20d=0.20,
                relative_volume_20d=2.0,
                revenue_growth=0.40,
                forward_pe=20,
            ),
            _candidate(
                "MOMO",
                return_60d=0.55,
                stock_minus_sector_return_20d=0.18,
                relative_volume_20d=1.6,
                volume_change_5d=0.30,
                revenue_growth=0.18,
                forward_pe=28,
            ),
            _candidate(
                "VOLU",
                return_1d=0.08,
                return_60d=0.15,
                stock_minus_sector_return_20d=0.04,
                relative_volume_20d=3.5,
                volume_change_5d=1.4,
                volume_zscore_20d=3.0,
                revenue_growth=0.08,
                forward_pe=36,
            ),
            _candidate(
                "QUAL",
                return_60d=0.20,
                stock_minus_sector_return_20d=0.06,
                relative_volume_20d=1.2,
                revenue_growth=0.35,
                eps_growth=0.45,
                gross_margin=0.72,
                free_cash_flow_yield=0.06,
                debt_to_equity=0.2,
                forward_pe=24,
            ),
            _candidate(
                "VALU",
                return_5d=0.12,
                return_60d=-0.18,
                stock_minus_sector_return_20d=0.03,
                relative_volume_20d=1.8,
                drawdown_60d=-0.38,
                revenue_growth=0.05,
                free_cash_flow_yield=0.10,
                forward_pe=8,
            ),
            _candidate(
                "TINY",
                adj_close=4.50,
                avg_dollar_volume_20d=1_000_000,
                market_cap=100_000_000,
            ),
        ]
    )
    config = DiscoveryConfig(top_n=3)

    result = run_stock_discovery(
        frame,
        existing_watchlist=["KEEP"],
        as_of_date="2024-12-31",
        config=config,
    )

    assert "KEEP" not in set(result.candidates["symbol"])
    assert "TINY" not in set(result.candidates["symbol"])
    assert set(result.exclusions["symbol"]) == {"KEEP", "TINY"}
    assert "already_in_watchlist" in result.exclusions.loc[
        result.exclusions["symbol"] == "KEEP", "exclusion_reasons"
    ].iloc[0]

    assert result.watchlists["momentum_breakouts"].iloc[0]["symbol"] == "MOMO"
    assert result.watchlists["volume_anomalies"].iloc[0]["symbol"] == "VOLU"
    assert result.watchlists["quality_growth"].iloc[0]["symbol"] == "QUAL"
    assert result.watchlists["value_recoveries"].iloc[0]["symbol"] == "VALU"
    assert result.watchlists["sector_leaders"].iloc[0]["symbol"] == "MOMO"

    top_momentum = result.watchlists["momentum_breakouts"].iloc[0]
    assert "20-day return" in top_momentum["why_flagged"]
    assert "earnings in 6 days" in top_momentum["risk_flags"]


def test_stock_discovery_derives_v1_features_from_history():
    dates = pd.date_range("2024-01-01", periods=260, freq="D")
    rows = []
    for symbol, offset in (("FAST", 0), ("SLOW", 20)):
        for index, date in enumerate(dates):
            price = 50 + offset + index * (1.0 if symbol == "FAST" else 0.2)
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "adj_close": price,
                    "volume": 1_000_000 + index * (100 if symbol == "FAST" else 10),
                    "sector": "Industrials",
                    "avg_dollar_volume_20d": 20_000_000,
                    "return_20d": 0.10 if symbol == "FAST" else 0.02,
                    "return_60d": 0.30 if symbol == "FAST" else 0.04,
                    "stock_minus_market_return_20d": 0.08 if symbol == "FAST" else 0.01,
                    "stock_minus_sector_return_20d": 0.06 if symbol == "FAST" else -0.01,
                    "sector_rank_return_20d": 1.0 if symbol == "FAST" else 0.5,
                    "sector_rank_momentum_20d": 1.0 if symbol == "FAST" else 0.5,
                }
            )
    frame = pd.DataFrame(rows)
    config = DiscoveryConfig(
        top_n=2,
        min_price=0,
        min_avg_dollar_volume_20d=0,
        min_market_cap=None,
    )

    result = run_stock_discovery(frame, as_of_date=dates[-1], config=config)

    assert {"price_above_sma_200", "distance_from_52w_high", "relative_volume_5d"}.issubset(
        result.candidates.columns
    )
    latest_fast = result.candidates.loc[result.candidates["symbol"] == "FAST"].iloc[0]
    assert pd.notna(latest_fast["price_above_sma_200"])
    assert pd.notna(latest_fast["distance_from_52w_high"])
    assert pd.notna(latest_fast["relative_volume_5d"])


def test_stock_discovery_leaves_quality_lane_empty_without_fundamentals():
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-12-31"),
                "symbol": "TECH",
                "sector": "Information Technology",
                "adj_close": 75.0,
                "avg_dollar_volume_20d": 25_000_000,
                "return_20d": 0.18,
                "return_60d": 0.30,
                "stock_minus_market_return_20d": 0.12,
                "stock_minus_sector_return_20d": 0.10,
                "sector_return_20d": 0.08,
                "sector_return_60d": 0.12,
                "sector_rank_return_20d": 1.0,
                "sector_rank_return_60d": 1.0,
                "sector_rank_momentum_20d": 1.0,
                "relative_volume_20d": 1.4,
                "price_above_sma_50": 0.08,
                "drawdown_60d": -0.05,
            },
            {
                "date": pd.Timestamp("2024-12-31"),
                "symbol": "SLOW",
                "sector": "Information Technology",
                "adj_close": 55.0,
                "avg_dollar_volume_20d": 20_000_000,
                "return_20d": 0.03,
                "return_60d": 0.05,
                "stock_minus_market_return_20d": -0.01,
                "stock_minus_sector_return_20d": -0.02,
                "sector_return_20d": 0.08,
                "sector_return_60d": 0.12,
                "sector_rank_return_20d": 0.5,
                "sector_rank_return_60d": 0.5,
                "sector_rank_momentum_20d": 0.5,
                "relative_volume_20d": 1.0,
                "price_above_sma_50": 0.01,
                "drawdown_60d": -0.12,
            },
        ]
    )
    config = DiscoveryConfig(
        top_n=2,
        min_market_cap=None,
        min_avg_dollar_volume_20d=0,
    )

    result = run_stock_discovery(frame, as_of_date="2024-12-31", config=config)

    assert result.candidates["quality_score_inputs"].max() == 0
    assert result.watchlists["quality_growth"].empty


def test_stock_discovery_populates_quality_lane_with_required_fundamentals():
    frame = pd.DataFrame(
        [
            _candidate(
                "GOOD",
                revenue_growth=0.35,
                eps_growth=0.45,
                gross_margin=0.70,
                operating_margin=0.30,
                return_on_equity=0.32,
                debt_to_equity=0.2,
                forward_pe=22.0,
            ),
            _candidate(
                "OKAY",
                revenue_growth=0.10,
                eps_growth=0.12,
                gross_margin=0.42,
                operating_margin=0.15,
                return_on_equity=0.11,
                debt_to_equity=1.3,
                forward_pe=35.0,
            ),
        ]
    )
    config = DiscoveryConfig(top_n=2, min_avg_dollar_volume_20d=0)

    result = run_stock_discovery(frame, as_of_date="2024-12-31", config=config)

    assert result.candidates["quality_score_eligible"].all()
    assert result.watchlists["quality_growth"].iloc[0]["symbol"] == "GOOD"


def test_stock_discovery_applies_market_cap_filter_when_available():
    frame = pd.DataFrame(
        [
            _candidate("BIG", market_cap=1_000_000_000),
            _candidate("SMALL", market_cap=100_000_000),
        ]
    )
    config = DiscoveryConfig(min_market_cap=300_000_000, min_avg_dollar_volume_20d=0)

    result = run_stock_discovery(frame, as_of_date="2024-12-31", config=config)

    assert set(result.candidates["symbol"]) == {"BIG"}
    assert "market_cap_below_minimum" in result.exclusions.loc[
        result.exclusions["symbol"] == "SMALL", "exclusion_reasons"
    ].iloc[0]


def test_stock_discovery_requires_market_cap_when_filter_enabled():
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-12-31"),
                "symbol": "NOCAP",
                "adj_close": 40.0,
                "avg_dollar_volume_20d": 25_000_000,
            }
        ]
    )

    with pytest.raises(ValueError, match="market_cap is required"):
        run_stock_discovery(frame, as_of_date="2024-12-31")


def _candidate(
    symbol: str,
    *,
    adj_close: float = 40.0,
    avg_dollar_volume_20d: float = 25_000_000,
    market_cap: float = 1_000_000_000,
    return_1d: float = 0.02,
    return_5d: float = 0.04,
    return_20d: float = 0.12,
    return_60d: float = 0.10,
    stock_minus_sector_return_20d: float = 0.02,
    relative_volume_20d: float = 1.1,
    volume_change_5d: float = 0.05,
    volume_zscore_20d: float = 0.5,
    revenue_growth: float = 0.10,
    eps_growth: float = 0.12,
    gross_margin: float = 0.40,
    operating_margin: float = 0.12,
    free_cash_flow_yield: float = 0.03,
    debt_to_equity: float = 0.8,
    return_on_equity: float = 0.10,
    forward_pe: float = 25.0,
    drawdown_60d: float = -0.08,
) -> dict:
    return {
        "date": pd.Timestamp("2024-12-31"),
        "symbol": symbol,
        "name": symbol,
        "sector": "Information Technology",
        "industry": "Software",
        "adj_close": adj_close,
        "avg_dollar_volume_20d": avg_dollar_volume_20d,
        "market_cap": market_cap,
        "return_1d": return_1d,
        "return_5d": return_5d,
        "return_20d": return_20d,
        "return_60d": return_60d,
        "stock_minus_market_return_20d": stock_minus_sector_return_20d / 2,
        "stock_minus_sector_return_20d": stock_minus_sector_return_20d,
        "stock_minus_sector_return_60d": return_60d / 2,
        "sector_return_20d": 0.08,
        "sector_return_60d": 0.12,
        "sector_rank_return_20d": min(1.0, 0.5 + return_20d),
        "sector_rank_return_60d": min(1.0, 0.5 + return_60d),
        "sector_rank_momentum_20d": min(1.0, 0.5 + return_20d),
        "sector_rank_volatility_20d": 0.4,
        "relative_volume_20d": relative_volume_20d,
        "volume_change_5d": volume_change_5d,
        "volume_zscore_20d": volume_zscore_20d,
        "volatility_20d": 0.06,
        "volatility_60d": 0.04,
        "price_above_sma_20": 0.03,
        "price_above_sma_50": 0.05,
        "price_above_sma_200": 0.02,
        "distance_from_52w_high": -0.03,
        "drawdown_60d": drawdown_60d,
        "revenue_growth": revenue_growth,
        "eps_growth": eps_growth,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "free_cash_flow_yield": free_cash_flow_yield,
        "debt_to_equity": debt_to_equity,
        "return_on_equity": return_on_equity,
        "forward_pe": forward_pe,
        "days_to_earnings": 6,
    }
