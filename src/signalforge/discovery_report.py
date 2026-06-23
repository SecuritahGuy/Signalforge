from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from signalforge.discovery import WATCHLIST_SCORE_COLUMNS, DiscoveryResult
from signalforge.fundamentals import quality_growth_required_fields_available

LANE_DISPLAY_NAMES = {
    "momentum_breakouts": "Momentum Breakouts",
    "sector_leaders": "Sector Leaders",
    "volume_anomalies": "Volume Anomalies",
    "quality_growth": "Quality Growth",
    "value_recoveries": "Value Recoveries",
}

LANE_REASON_LABELS = {
    "momentum_breakouts": "Momentum breakout candidate",
    "sector_leaders": "Sector leader candidate",
    "volume_anomalies": "Volume anomaly candidate",
    "quality_growth": "Quality growth candidate",
    "value_recoveries": "Value/recovery candidate",
}

LANE_REASON_BODIES = {
    "momentum_breakouts": (
        "strong recent trend score and relative price strength after excluding "
        "tracked watchlist names."
    ),
    "sector_leaders": (
        "ranked highly versus sector peers on available price/volume/sector features."
    ),
    "volume_anomalies": "unusual volume behavior relative to recent history.",
    "quality_growth": "ranked highly on available fundamental quality/growth inputs.",
    "value_recoveries": "recovery-style price behavior based on current price-derived features.",
}

def write_discovery_outputs(
    result: DiscoveryResult,
    output_dir: str | Path,
    *,
    source_universe: pd.DataFrame | int | None = None,
) -> dict[str, Path]:
    """Write machine-readable and human-readable discovery artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _remove_stale_lane_files(output_path)

    candidates = _candidate_export_frame(result.candidates)
    candidates_path = output_path / "candidates.csv"
    candidates.to_csv(candidates_path, index=False)

    exclusions_path = output_path / "exclusions.csv"
    result.exclusions.to_csv(exclusions_path, index=False)

    lane_frames = _populated_lane_frames(result.watchlists)
    lane_paths: dict[str, Path] = {}
    for lane, frame in lane_frames.items():
        lane_path = output_path / f"lane_{lane}.csv"
        frame.to_csv(lane_path, index=False)
        lane_paths[lane] = lane_path

    multi_lane = build_multi_lane_candidates(lane_frames, candidates=result.candidates)
    summary = build_discovery_summary(
        result,
        source_universe=source_universe,
        populated_lanes=lane_frames,
        multi_lane=multi_lane,
    )

    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")

    report_path = output_path / "report.md"
    report_path.write_text(
        render_discovery_report(
            result,
            source_universe_count=summary.get("source_universe_count"),
            watchlist_exclusion_count=summary["watchlist_exclusion_count"],
            populated_lanes=lane_frames,
            multi_lane=multi_lane,
        )
    )

    artifacts = {
        "summary": summary_path,
        "candidates": candidates_path,
        "exclusions": exclusions_path,
        "report": report_path,
    }
    artifacts.update({f"lane_{lane}": path for lane, path in lane_paths.items()})
    return artifacts


def build_discovery_summary(
    result: DiscoveryResult,
    *,
    source_universe: pd.DataFrame | int | None = None,
    populated_lanes: dict[str, pd.DataFrame] | None = None,
    multi_lane: pd.DataFrame | None = None,
) -> dict:
    lane_frames = populated_lanes or _populated_lane_frames(result.watchlists)
    multi_lane_frame = multi_lane
    if multi_lane_frame is None:
        multi_lane_frame = build_multi_lane_candidates(lane_frames, candidates=result.candidates)
    summary = {
        "as_of_date": result.as_of_date.date().isoformat(),
        "candidate_count": int(len(result.candidates)),
        "exclusion_count": int(len(result.exclusions)),
        "watchlist_exclusion_count": _watchlist_exclusion_count(result.exclusions),
        "watchlists": {name: int(len(frame)) for name, frame in result.watchlists.items()},
        "populated_watchlists": {name: int(len(frame)) for name, frame in lane_frames.items()},
        "multi_lane_candidate_count": int(len(multi_lane_frame)),
        "note": (
            "Discovery output flags candidates for research review only; it is not a buy signal "
            "or broker execution instruction."
        ),
    }
    source_count = _source_universe_count(source_universe)
    if source_count is not None:
        summary["source_universe_count"] = source_count
    missing = missing_lane_explanations(result)
    if missing:
        summary["missing_lane_explanations"] = missing
    return summary


def render_discovery_report(
    result: DiscoveryResult,
    *,
    source_universe_count: int | None,
    watchlist_exclusion_count: int,
    populated_lanes: dict[str, pd.DataFrame] | None = None,
    multi_lane: pd.DataFrame | None = None,
    top_n: int = 10,
) -> str:
    lane_frames = populated_lanes or _populated_lane_frames(result.watchlists)
    multi_lane_frame = multi_lane
    if multi_lane_frame is None:
        multi_lane_frame = build_multi_lane_candidates(lane_frames, candidates=result.candidates)

    lines = [
        "# Stock Discovery Report",
        "",
        f"- As-of date: {result.as_of_date.date().isoformat()}",
    ]
    if source_universe_count is not None:
        lines.append(f"- Source universe count: {source_universe_count}")
    lines.extend(
        [
            f"- Watchlist exclusion count: {watchlist_exclusion_count}",
            f"- Final candidate count: {len(result.candidates)}",
            "",
            "## Lane Counts",
            "",
            "| Lane | Count |",
            "| --- | ---: |",
        ]
    )
    for lane in WATCHLIST_SCORE_COLUMNS:
        lines.append(f"| {_md(LANE_DISPLAY_NAMES[lane])} | {len(result.watchlists[lane])} |")

    lines.extend(["", "## Top Candidates By Lane", ""])
    if lane_frames:
        for lane, frame in lane_frames.items():
            lines.append(f"### {LANE_DISPLAY_NAMES[lane]}")
            lines.append("")
            lines.extend(_lane_table(frame.head(top_n), lane))
            lines.append("")
    else:
        lines.extend(["No populated lanes for this discovery run.", ""])

    lines.extend(["## Multi-Lane Candidates", ""])
    if multi_lane_frame.empty:
        lines.extend(["No symbols appeared in more than one populated lane.", ""])
    else:
        lines.extend(_multi_lane_table(multi_lane_frame))
        lines.append("")

    missing = missing_lane_explanations(result)
    if missing:
        lines.extend(["## Missing Lane Explanations", ""])
        for lane in WATCHLIST_SCORE_COLUMNS:
            if lane in missing:
                lines.append(f"- `{lane}`: {missing[lane]}")
        lines.append("")

    return "\n".join(lines)


def build_multi_lane_candidates(
    lane_frames: dict[str, pd.DataFrame],
    *,
    candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return symbols that appear in more than one populated lane."""
    symbol_to_lanes: dict[str, list[str]] = {}
    for lane in WATCHLIST_SCORE_COLUMNS:
        frame = lane_frames.get(lane)
        if frame is None or frame.empty or "symbol" not in frame.columns:
            continue
        for symbol in frame["symbol"].dropna().astype(str).str.upper().unique():
            symbol_to_lanes.setdefault(symbol, []).append(lane)

    repeated = {
        symbol: lanes for symbol, lanes in symbol_to_lanes.items() if len(set(lanes)) > 1
    }
    if not repeated:
        return pd.DataFrame(
            columns=[
                "symbol",
                "name",
                "sector",
                "industry",
                "lanes_matched",
                "lane_count",
                "composite_score",
            ]
        )

    base = _multi_lane_base_frame(lane_frames, candidates)
    rows = []
    for symbol, lanes in repeated.items():
        row = base.loc[base["symbol"] == symbol].head(1)
        source = row.iloc[0] if not row.empty else pd.Series({"symbol": symbol})
        ordered_lanes = [lane for lane in WATCHLIST_SCORE_COLUMNS if lane in set(lanes)]
        rows.append(
            {
                "symbol": symbol,
                "name": source.get("name", ""),
                "sector": source.get("sector", ""),
                "industry": source.get("industry", ""),
                "lanes_matched": ", ".join(ordered_lanes),
                "lane_count": len(ordered_lanes),
                "composite_score": source.get("discovery_score", pd.NA),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["lane_count", "composite_score", "symbol"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def missing_lane_explanations(result: DiscoveryResult) -> dict[str, str]:
    missing: dict[str, str] = {}
    for lane in WATCHLIST_SCORE_COLUMNS:
        frame = result.watchlists.get(lane)
        if frame is not None and not frame.empty:
            continue
        if lane == "quality_growth" and _quality_inputs_absent(result.candidates):
            missing[lane] = (
                "quality_growth is empty because required fundamental fields are not present "
                "in the current price/volume/sector-only research frame."
            )
        else:
            missing[lane] = "No eligible candidates had usable inputs for this lane."
    return missing


def lane_candidate_reason(lane: str, row: pd.Series) -> str:
    symbol = str(row.get("symbol", "")).upper()
    label = LANE_REASON_LABELS.get(lane, "Discovery candidate")
    body = LANE_REASON_BODIES.get(lane, "ranked highly on available discovery inputs.")
    return f"{symbol} - {label}: {body}"


def _candidate_export_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    frame = candidates.copy()
    if "discovery_score" in frame.columns:
        return frame.sort_values(
            ["discovery_score", "symbol"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)
    return frame.sort_values("symbol").reset_index(drop=True)


def _populated_lane_frames(watchlists: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    populated = {}
    for lane in WATCHLIST_SCORE_COLUMNS:
        frame = watchlists.get(lane)
        if frame is None or frame.empty:
            continue
        populated[lane] = _lane_export_frame(lane, frame)
    return populated


def _lane_export_frame(lane: str, frame: pd.DataFrame) -> pd.DataFrame:
    lane_frame = frame.copy()
    score_col = WATCHLIST_SCORE_COLUMNS[lane]
    sort_columns = [
        column
        for column in (score_col, "discovery_score", "avg_dollar_volume_20d", "symbol")
        if column in lane_frame.columns
    ]
    if sort_columns:
        ascending = [False if column != "symbol" else True for column in sort_columns]
        lane_frame = lane_frame.sort_values(
            sort_columns,
            ascending=ascending,
            na_position="last",
        )
    lane_frame["lane_reason"] = [
        lane_candidate_reason(lane, row) for _, row in lane_frame.iterrows()
    ]
    return lane_frame.reset_index(drop=True)


def _multi_lane_base_frame(
    lane_frames: dict[str, pd.DataFrame],
    candidates: pd.DataFrame | None,
) -> pd.DataFrame:
    if candidates is not None and not candidates.empty and "symbol" in candidates.columns:
        base = candidates.copy()
    else:
        frames = [frame for frame in lane_frames.values() if not frame.empty]
        base = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if base.empty:
        return pd.DataFrame(columns=["symbol"])
    base["symbol"] = base["symbol"].astype(str).str.upper()
    if "discovery_score" in base.columns:
        base = base.sort_values(["discovery_score", "symbol"], ascending=[False, True])
    return base.drop_duplicates(subset=["symbol"], keep="first")


def _lane_table(frame: pd.DataFrame, lane: str) -> list[str]:
    score_col = WATCHLIST_SCORE_COLUMNS[lane]
    lines = [
        "| Symbol | Name | Sector | Composite Score | Lane Score | Reason |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            f"{_md(row.get('symbol', ''))} | "
            f"{_md(row.get('name', ''))} | "
            f"{_md(row.get('sector', ''))} | "
            f"{_format_score(row.get('discovery_score'))} | "
            f"{_format_score(row.get(score_col))} | "
            f"{_md(row.get('lane_reason', lane_candidate_reason(lane, row)))} |"
        )
    return lines


def _multi_lane_table(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| Symbol | Name | Sector | Lanes Matched | Lane Count | Composite Score |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            f"{_md(row.get('symbol', ''))} | "
            f"{_md(row.get('name', ''))} | "
            f"{_md(row.get('sector', ''))} | "
            f"{_md(row.get('lanes_matched', ''))} | "
            f"{int(row.get('lane_count', 0))} | "
            f"{_format_score(row.get('composite_score'))} |"
        )
    return lines


def _quality_inputs_absent(candidates: pd.DataFrame) -> bool:
    if candidates.empty:
        return True
    if "quality_score_eligible" in candidates.columns:
        return not candidates["quality_score_eligible"].fillna(False).astype(bool).any()
    if "quality_score_inputs" in candidates.columns:
        input_count = pd.to_numeric(
            candidates["quality_score_inputs"],
            errors="coerce",
        ).fillna(0)
        return input_count.max() <= 0
    return not quality_growth_required_fields_available(candidates)


def _watchlist_exclusion_count(exclusions: pd.DataFrame) -> int:
    if exclusions.empty or "exclusion_reasons" not in exclusions.columns:
        return 0
    return int(
        exclusions["exclusion_reasons"].astype(str).str.contains("already_in_watchlist").sum()
    )


def _source_universe_count(source_universe: pd.DataFrame | int | None) -> int | None:
    if source_universe is None:
        return None
    if isinstance(source_universe, int):
        return source_universe
    if "symbol" in source_universe.columns:
        return int(source_universe["symbol"].dropna().astype(str).str.upper().nunique())
    return int(len(source_universe))


def _remove_stale_lane_files(output_dir: Path) -> None:
    stale_names = {"scored_candidates.csv"}
    stale_names.update(f"{lane}.csv" for lane in WATCHLIST_SCORE_COLUMNS)
    for path in output_dir.glob("lane_*.csv"):
        path.unlink()
    for name in stale_names:
        path = output_dir / name
        if path.exists():
            path.unlink()


def _format_score(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _md(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")
