import pandas as pd

from scripts.run_symbol_discovery_rd import (
    build_monitoring_candidates,
    select_top_fraction,
    update_monitoring_state,
)


def test_select_top_fraction_keeps_top_third_by_discovery_score():
    candidates = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E"],
            "discovery_score": [50, 90, 70, 60, 80],
        }
    )

    selected = select_top_fraction(candidates, top_fraction=1 / 3)

    assert selected["symbol"].tolist() == ["B", "E"]


def test_monitoring_candidates_require_period_and_appearances_before_review():
    today = pd.Timestamp("2026-06-10")
    candidates = pd.DataFrame(
        {
            "date": [today],
            "symbol": ["XYZ"],
            "name": ["Example"],
            "sector": ["Tech"],
            "industry": ["Software"],
            "discovery_score": [72.0],
        }
    )
    prior_state = pd.DataFrame(
        {
            "symbol": ["XYZ"],
            "first_seen": ["2026-06-01"],
            "last_seen": ["2026-06-05"],
            "appearances": [2],
            "max_discovery_score": [75.0],
        }
    )
    lanes = pd.DataFrame(
        {"symbol": ["XYZ"], "lanes_matched": ["momentum_breakouts"], "lane_count": [1]}
    )

    monitored = build_monitoring_candidates(
        candidates,
        prior_state,
        lane_membership=lanes,
        active_symbols=set(),
        tracked_symbols=set(),
        as_of_date=today,
        monitoring_days=5,
        min_appearances=3,
        min_discovery_score=60.0,
    )

    assert monitored.loc[0, "promotion_status"] == "eligible_for_review"
    assert monitored.loc[0, "appearances"] == 3
    assert monitored.loc[0, "monitoring_age_days"] == 9


def test_monitoring_candidates_block_tracked_and_active_symbols():
    today = pd.Timestamp("2026-06-10")
    candidates = pd.DataFrame(
        {
            "date": [today],
            "symbol": ["XYZ"],
            "name": ["Example"],
            "sector": ["Tech"],
            "industry": ["Software"],
            "discovery_score": [72.0],
        }
    )
    lanes = pd.DataFrame(columns=["symbol", "lanes_matched", "lane_count"])

    monitored = build_monitoring_candidates(
        candidates,
        pd.DataFrame(),
        lane_membership=lanes,
        active_symbols={"XYZ"},
        tracked_symbols={"XYZ"},
        as_of_date=today,
        monitoring_days=5,
        min_appearances=3,
        min_discovery_score=60.0,
    )

    blockers = monitored.loc[0, "promotion_blockers"]
    assert monitored.loc[0, "promotion_status"] == "monitoring"
    assert "already_active_in_paper" in blockers
    assert "already_tracked" in blockers
    assert "appearances_below_3" in blockers


def test_update_monitoring_state_preserves_max_score():
    candidates = pd.DataFrame(
        {
            "symbol": ["XYZ"],
            "name": ["Example"],
            "sector": ["Tech"],
            "industry": ["Software"],
            "first_seen": ["2026-06-01"],
            "last_seen": ["2026-06-10"],
            "appearances": [3],
            "discovery_score": [65.0],
            "lanes_matched": ["momentum_breakouts"],
            "promotion_status": ["eligible_for_review"],
            "promotion_blockers": [""],
        }
    )
    previous = pd.DataFrame(
        {
            "symbol": ["XYZ"],
            "first_seen": ["2026-06-01"],
            "last_seen": ["2026-06-05"],
            "appearances": [2],
            "max_discovery_score": [80.0],
        }
    )

    state = update_monitoring_state(previous, candidates, as_of_date=pd.Timestamp("2026-06-10"))

    assert state.loc[0, "max_discovery_score"] == 80.0
    assert state.loc[0, "latest_discovery_score"] == 65.0
    assert state.loc[0, "promotion_status"] == "eligible_for_review"
