import json

import pandas as pd

from signalforge.discovery import DiscoveryResult
from signalforge.discovery_report import build_multi_lane_candidates, write_discovery_outputs


def test_discovery_outputs_write_report_candidates_and_populated_lanes(tmp_path):
    result = _discovery_result()
    stale_empty_lane = tmp_path / "lane_quality_growth.csv"
    stale_empty_lane.write_text("stale\n")

    write_discovery_outputs(
        result,
        tmp_path,
        source_universe=pd.DataFrame({"symbol": ["AAA", "BBB", "CCC", "DDD"]}),
    )

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "candidates.csv").exists()
    assert (tmp_path / "lane_momentum_breakouts.csv").exists()
    assert (tmp_path / "lane_sector_leaders.csv").exists()
    assert (tmp_path / "lane_value_recoveries.csv").exists()
    assert not (tmp_path / "lane_quality_growth.csv").exists()
    assert not (tmp_path / "lane_volume_anomalies.csv").exists()
    assert (tmp_path / "report.md").exists()

    candidates = pd.read_csv(tmp_path / "candidates.csv")
    assert candidates["symbol"].tolist() == ["AAA", "BBB", "CCC"]

    momentum = pd.read_csv(tmp_path / "lane_momentum_breakouts.csv")
    assert "lane_reason" in momentum.columns
    assert "Momentum breakout candidate" in momentum.loc[0, "lane_reason"]

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["source_universe_count"] == 4
    assert summary["candidate_count"] == 3
    assert summary["watchlist_exclusion_count"] == 1
    assert summary["multi_lane_candidate_count"] == 2

    report = (tmp_path / "report.md").read_text()
    assert "# Stock Discovery Report" in report
    assert "Source universe count: 4" in report
    assert "Watchlist exclusion count: 1" in report
    assert "Final candidate count: 3" in report
    assert "## Multi-Lane Candidates" in report
    assert "AAA" in report
    assert "BBB" in report
    assert (
        "quality_growth is empty because required fundamental fields are not present "
        "in the current price/volume/sector-only research frame."
    ) in report


def test_multi_lane_candidates_are_identified_deterministically():
    result = _discovery_result()

    multi_lane = build_multi_lane_candidates(
        {
            lane: frame
            for lane, frame in result.watchlists.items()
            if frame is not None and not frame.empty
        },
        candidates=result.candidates,
    )

    assert multi_lane["symbol"].tolist() == ["AAA", "BBB"]
    assert multi_lane.loc[0, "lanes_matched"] == "momentum_breakouts, sector_leaders"
    assert multi_lane.loc[0, "lane_count"] == 2
    assert multi_lane.loc[0, "composite_score"] == 91.0
    assert multi_lane.loc[1, "lanes_matched"] == "momentum_breakouts, value_recoveries"


def _discovery_result() -> DiscoveryResult:
    candidates = pd.DataFrame(
        [
            _candidate("AAA", "Alpha", "Technology", 91.0, 98.0, 95.0, 30.0, 50.0),
            _candidate("BBB", "Beta", "Industrials", 88.0, 90.0, 20.0, 35.0, 94.0),
            _candidate("CCC", "Gamma", "Financials", 77.0, 40.0, 91.0, 45.0, 55.0),
        ]
    )
    exclusions = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-21"),
                "symbol": "WATCH",
                "exclusion_reasons": "already_in_watchlist",
            }
        ]
    )
    watchlists = {
        "momentum_breakouts": candidates.loc[candidates["symbol"].isin(["AAA", "BBB"])].copy(),
        "sector_leaders": candidates.loc[candidates["symbol"].isin(["AAA", "CCC"])].copy(),
        "volume_anomalies": pd.DataFrame(),
        "quality_growth": pd.DataFrame(),
        "value_recoveries": candidates.loc[candidates["symbol"].isin(["BBB"])].copy(),
    }
    return DiscoveryResult(
        as_of_date=pd.Timestamp("2026-05-21"),
        candidates=candidates,
        watchlists=watchlists,
        exclusions=exclusions,
    )


def _candidate(
    symbol: str,
    name: str,
    sector: str,
    discovery_score: float,
    momentum_score: float,
    sector_strength_score: float,
    attention_score: float,
    value_recovery_score: float,
) -> dict:
    return {
        "date": pd.Timestamp("2026-05-21"),
        "symbol": symbol,
        "name": name,
        "sector": sector,
        "industry": "Example Industry",
        "discovery_score": discovery_score,
        "momentum_score": momentum_score,
        "sector_strength_score": sector_strength_score,
        "attention_score": attention_score,
        "quality_score": 50.0,
        "quality_score_inputs": 0,
        "value_recovery_score": value_recovery_score,
        "avg_dollar_volume_20d": 10_000_000.0,
    }
