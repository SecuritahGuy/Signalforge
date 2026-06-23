from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from signalforge.data import UNIVERSE_COLUMNS, load_universe_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote eligible discovery candidates into the traded paper universe."
    )
    parser.add_argument(
        "--promotion-candidates",
        default="reports/symbol_discovery_rd/promotion_candidates.csv",
    )
    parser.add_argument("--universe", default="data/reference/tracked_universe.csv")
    parser.add_argument("--output-prefix", default="reports/symbol_discovery_promotion_plan")
    parser.add_argument("--max-symbols", type=int, default=5)
    parser.add_argument("--min-discovery-score", type=float, default=60.0)
    parser.add_argument("--min-lane-count", type=int, default=0)
    parser.add_argument("--min-appearances", type=int, default=0)
    parser.add_argument("--min-monitoring-age-days", type=int, default=0)
    parser.add_argument("--max-sector-symbols", type=int, default=None)
    parser.add_argument("--category", default="promoted_discovery")
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args()

    candidates = _read_candidates(Path(args.promotion_candidates))
    universe = load_universe_csv(args.universe)
    plan = build_promotion_plan(
        candidates,
        universe,
        max_symbols=args.max_symbols,
        min_discovery_score=args.min_discovery_score,
        min_lane_count=args.min_lane_count,
        min_appearances=args.min_appearances,
        min_monitoring_age_days=args.min_monitoring_age_days,
        max_sector_symbols=args.max_sector_symbols,
        category=args.category,
    )
    summary = build_summary(
        plan,
        approved=args.approve,
        thresholds={
            "max_symbols": args.max_symbols,
            "min_discovery_score": args.min_discovery_score,
            "min_lane_count": args.min_lane_count,
            "min_appearances": args.min_appearances,
            "min_monitoring_age_days": args.min_monitoring_age_days,
            "max_sector_symbols": args.max_sector_symbols,
        },
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output_prefix.with_name(output_prefix.name + "_candidates.csv"), index=False)
    output_prefix.with_name(output_prefix.name + "_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    output_prefix.with_name(output_prefix.name + "_report.md").write_text(
        render_promotion_report(summary, plan)
    )

    if args.approve:
        promoted = plan.loc[plan["promotion_plan_status"].eq("ready_to_promote")]
        updated = append_promotions(universe, promoted)
        updated.to_csv(args.universe, index=False)
        print(f"promoted {len(promoted):,} symbols into {args.universe}")
    else:
        print(
            "wrote promotion plan only; rerun with --approve to update "
            f"{args.universe}"
        )


def build_promotion_plan(
    candidates: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    max_symbols: int,
    min_discovery_score: float,
    min_lane_count: int = 0,
    min_appearances: int = 0,
    min_monitoring_age_days: int = 0,
    max_sector_symbols: int | None = None,
    category: str,
) -> pd.DataFrame:
    if max_symbols <= 0:
        raise ValueError("max_symbols must be positive")
    if min_lane_count < 0:
        raise ValueError("min_lane_count must be non-negative")
    if min_appearances < 0:
        raise ValueError("min_appearances must be non-negative")
    if min_monitoring_age_days < 0:
        raise ValueError("min_monitoring_age_days must be non-negative")
    if max_sector_symbols is not None and max_sector_symbols <= 0:
        raise ValueError("max_sector_symbols must be positive")
    if candidates.empty:
        return _empty_plan_frame()

    frame = candidates.copy()
    for column in ("symbol", "name", "sector", "industry", "promotion_status"):
        if column not in frame:
            frame[column] = ""
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["discovery_score"] = pd.to_numeric(
        frame.get("discovery_score", 0.0),
        errors="coerce",
    ).fillna(0.0)
    frame["lane_count"] = _numeric_column(frame, "lane_count")
    frame["appearances"] = _numeric_column(frame, "appearances")
    frame["monitoring_age_days"] = _numeric_column(frame, "monitoring_age_days")

    tracked = set(universe["symbol"].astype(str).str.upper())
    rows = []
    sector_counts: dict[str, int] = {}
    for _, row in frame.sort_values("discovery_score", ascending=False).iterrows():
        blockers = []
        if row["promotion_status"] != "eligible_for_review":
            blockers.append("not_eligible_for_review")
        if row["symbol"] in tracked:
            blockers.append("already_tracked")
        if row["discovery_score"] < min_discovery_score:
            blockers.append(f"discovery_score_below_{min_discovery_score:g}")
        if int(row.get("lane_count", 0)) < min_lane_count:
            blockers.append(f"lane_count_below_{min_lane_count}")
        if int(row.get("appearances", 0)) < min_appearances:
            blockers.append(f"appearances_below_{min_appearances}")
        if int(row.get("monitoring_age_days", 0)) < min_monitoring_age_days:
            blockers.append(f"monitoring_age_below_{min_monitoring_age_days}_days")
        sector = str(row.get("sector", "") or "")
        if not blockers and max_sector_symbols is not None:
            sector_count = sector_counts.get(sector, 0)
            if sector_count >= max_sector_symbols:
                blockers.append(f"max_sector_symbols_{max_sector_symbols}")
            else:
                sector_counts[sector] = sector_count + 1

        status = "ready_to_promote" if not blockers else "blocked"
        rows.append(
            {
                "symbol": row["symbol"],
                "name": row.get("name", ""),
                "category": category,
                "sector": row.get("sector", ""),
                "industry": row.get("industry", ""),
                "notes": _notes(row),
                "discovery_score": row["discovery_score"],
                "lane_count": int(row.get("lane_count", 0)),
                "lanes_matched": row.get("lanes_matched", ""),
                "appearances": int(row.get("appearances", 0)),
                "monitoring_age_days": int(row.get("monitoring_age_days", 0)),
                "first_seen": row.get("first_seen", ""),
                "last_seen": row.get("last_seen", ""),
                "promotion_plan_status": status,
                "promotion_plan_blockers": "; ".join(blockers),
            }
    )

    plan = pd.DataFrame(rows)
    ready_index = plan.loc[plan["promotion_plan_status"].eq("ready_to_promote")].head(
        max_symbols
    ).index
    overflow_index = plan.loc[
        plan["promotion_plan_status"].eq("ready_to_promote")
        & ~plan.index.isin(ready_index)
    ].index
    if len(overflow_index) > 0:
        plan.loc[overflow_index, "promotion_plan_status"] = "blocked"
        plan.loc[overflow_index, "promotion_plan_blockers"] = f"max_symbols_limit_{max_symbols}"

    ready = plan.loc[ready_index]
    blocked = plan.loc[~plan.index.isin(ready_index)]
    return pd.concat([ready, blocked], ignore_index=True).loc[:, _plan_columns()]


def append_promotions(universe: pd.DataFrame, promotions: pd.DataFrame) -> pd.DataFrame:
    if promotions.empty:
        return universe
    rows = promotions.loc[:, [*UNIVERSE_COLUMNS, "notes"]].copy()
    existing_columns = list(universe.columns)
    for column in existing_columns:
        if column not in rows:
            rows[column] = pd.NA
    combined = pd.concat([universe, rows.loc[:, existing_columns]], ignore_index=True)
    combined["symbol"] = combined["symbol"].astype(str).str.upper()
    return combined.drop_duplicates(subset=["symbol"], keep="first")


def build_summary(plan: pd.DataFrame, *, approved: bool, thresholds: dict | None = None) -> dict:
    ready = plan.loc[plan["promotion_plan_status"].eq("ready_to_promote")]
    return {
        "approved": approved,
        "candidate_count": int(len(plan)),
        "ready_to_promote_count": int(len(ready)),
        "blocked_count": int(len(plan) - len(ready)),
        "ready_symbols": ready["symbol"].tolist() if not ready.empty else [],
        "thresholds": thresholds or {},
        "note": (
            "Promotion appends eligible discovery candidates to the traded universe. "
            "The next paper workflow can score them, but promotion itself does not create orders."
        ),
    }


def render_promotion_report(summary: dict, plan: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# SignalForge Discovery Promotion Plan",
            "",
            f"Approved: `{summary['approved']}`",
            f"Candidates reviewed: `{summary['candidate_count']}`",
            f"Ready to promote: `{summary['ready_to_promote_count']}`",
            f"Blocked: `{summary['blocked_count']}`",
            "",
            "## Ready To Promote",
            "",
            _table(plan.loc[plan["promotion_plan_status"].eq("ready_to_promote")]),
            "",
            "## Blocked",
            "",
            _table(plan.loc[plan["promotion_plan_status"].eq("blocked")].head(25)),
            "",
        ]
    )


def _notes(row: pd.Series) -> str:
    bits = [
        "Promoted from symbol discovery monitoring",
        f"score={float(row.get('discovery_score', 0.0)):.2f}",
    ]
    lanes = str(row.get("lanes_matched", "") or "").strip()
    if lanes:
        bits.append(f"lanes={lanes}")
    return "; ".join(bits)


def _read_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_candidates()
    return pd.read_csv(path)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0)


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [
        "symbol",
        "name",
        "sector",
        "discovery_score",
        "lane_count",
        "appearances",
        "monitoring_age_days",
        "promotion_plan_blockers",
    ]
    display = frame.loc[:, [column for column in columns if column in frame]].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    rows = [list(display.columns), ["---"] * len(display.columns)]
    rows.extend(display.astype(str).values.tolist())
    return "\n".join("| " + " | ".join(str(value) for value in row) + " |" for row in rows)


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "promotion_status", "discovery_score"])


def _empty_plan_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_plan_columns())


def _plan_columns() -> list[str]:
    return [
        "symbol",
        "name",
        "category",
        "sector",
        "industry",
        "notes",
        "discovery_score",
        "lane_count",
        "lanes_matched",
        "appearances",
        "monitoring_age_days",
        "first_seen",
        "last_seen",
        "promotion_plan_status",
        "promotion_plan_blockers",
    ]


if __name__ == "__main__":
    main()
