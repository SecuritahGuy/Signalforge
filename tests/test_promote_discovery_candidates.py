import pandas as pd

from scripts.promote_discovery_candidates import (
    append_promotions,
    build_promotion_plan,
    build_summary,
)


def test_build_promotion_plan_selects_eligible_candidates_only():
    candidates = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "name": ["A", "B", "C"],
            "sector": ["Tech", "Health", "Energy"],
            "industry": ["Software", "Tools", "Oil"],
            "promotion_status": ["eligible_for_review", "monitoring", "eligible_for_review"],
            "discovery_score": [75.0, 90.0, 55.0],
            "lane_count": [3, 4, 2],
            "lanes_matched": ["momentum", "momentum, volume", "value"],
            "appearances": [4, 5, 4],
            "monitoring_age_days": [7, 10, 7],
        }
    )
    universe = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "name": ["SPDR S&P 500 ETF"],
            "category": ["benchmark"],
            "sector": ["ETF"],
            "industry": ["Broad Market"],
            "notes": ["Benchmark"],
        }
    )

    plan = build_promotion_plan(
        candidates,
        universe,
        max_symbols=2,
        min_discovery_score=60.0,
        category="promoted_discovery",
    )

    assert plan.loc[0, "symbol"] == "AAA"
    assert plan.loc[0, "promotion_plan_status"] == "ready_to_promote"
    assert plan.loc[plan["symbol"].eq("BBB"), "promotion_plan_blockers"].item() == (
        "not_eligible_for_review"
    )
    assert plan.loc[plan["symbol"].eq("CCC"), "promotion_plan_blockers"].item() == (
        "discovery_score_below_60"
    )


def test_build_promotion_plan_blocks_ready_candidates_above_max_symbols():
    candidates = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "name": ["A", "B", "C"],
            "sector": ["Tech", "Health", "Energy"],
            "industry": ["Software", "Tools", "Oil"],
            "promotion_status": ["eligible_for_review"] * 3,
            "discovery_score": [90.0, 80.0, 70.0],
        }
    )
    universe = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "name": ["SPDR S&P 500 ETF"],
            "category": ["benchmark"],
            "sector": ["ETF"],
            "industry": ["Broad Market"],
            "notes": ["Benchmark"],
        }
    )

    plan = build_promotion_plan(
        candidates,
        universe,
        max_symbols=2,
        min_discovery_score=60.0,
        category="promoted_discovery",
    )

    ready = plan.loc[plan["promotion_plan_status"].eq("ready_to_promote")]
    blocked = plan.loc[plan["promotion_plan_status"].eq("blocked")]

    assert ready["symbol"].tolist() == ["AAA", "BBB"]
    assert blocked["symbol"].tolist() == ["CCC"]
    assert blocked["promotion_plan_blockers"].item() == "max_symbols_limit_2"


def test_build_promotion_plan_applies_auto_approval_thresholds():
    candidates = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "name": ["A", "B", "C", "D"],
            "sector": ["Tech", "Tech", "Health", "Energy"],
            "industry": ["Software", "Hardware", "Tools", "Oil"],
            "promotion_status": ["eligible_for_review"] * 4,
            "discovery_score": [90.0, 85.0, 80.0, 75.0],
            "lane_count": [3, 3, 1, 3],
            "appearances": [5, 5, 5, 2],
            "monitoring_age_days": [10, 10, 10, 10],
        }
    )
    universe = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "name": ["SPDR S&P 500 ETF"],
            "category": ["benchmark"],
            "sector": ["ETF"],
            "industry": ["Broad Market"],
            "notes": ["Benchmark"],
        }
    )

    plan = build_promotion_plan(
        candidates,
        universe,
        max_symbols=5,
        min_discovery_score=70.0,
        min_lane_count=2,
        min_appearances=3,
        min_monitoring_age_days=5,
        max_sector_symbols=1,
        category="promoted_discovery",
    )

    assert plan.loc[plan["symbol"].eq("AAA"), "promotion_plan_status"].item() == (
        "ready_to_promote"
    )
    assert plan.loc[plan["symbol"].eq("BBB"), "promotion_plan_blockers"].item() == (
        "max_sector_symbols_1"
    )
    assert plan.loc[plan["symbol"].eq("CCC"), "promotion_plan_blockers"].item() == (
        "lane_count_below_2"
    )
    assert plan.loc[plan["symbol"].eq("DDD"), "promotion_plan_blockers"].item() == (
        "appearances_below_3"
    )


def test_append_promotions_preserves_universe_schema_and_dedupes():
    universe = pd.DataFrame(
        {
            "symbol": ["SPY", "AAA"],
            "name": ["SPDR S&P 500 ETF", "Existing A"],
            "category": ["benchmark", "manual"],
            "sector": ["ETF", "Tech"],
            "industry": ["Broad Market", "Software"],
            "notes": ["Benchmark", "Existing"],
        }
    )
    promotions = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "name": ["A", "B"],
            "category": ["promoted_discovery", "promoted_discovery"],
            "sector": ["Tech", "Health"],
            "industry": ["Software", "Tools"],
            "notes": ["promote A", "promote B"],
        }
    )

    updated = append_promotions(universe, promotions)

    assert updated["symbol"].tolist() == ["SPY", "AAA", "BBB"]
    assert updated.loc[updated["symbol"].eq("AAA"), "name"].item() == "Existing A"
    assert updated.loc[updated["symbol"].eq("BBB"), "category"].item() == (
        "promoted_discovery"
    )


def test_build_summary_reports_ready_symbols():
    plan = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "promotion_plan_status": ["ready_to_promote", "blocked"],
        }
    )

    summary = build_summary(plan, approved=False)

    assert summary["approved"] is False
    assert summary["ready_to_promote_count"] == 1
    assert summary["ready_symbols"] == ["AAA"]
