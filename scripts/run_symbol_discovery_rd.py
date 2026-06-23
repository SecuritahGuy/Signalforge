from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from signalforge.discovery import DiscoveryConfig, run_stock_discovery


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run research-only symbol discovery and update monitoring state."
    )
    parser.add_argument("--research-frame", default="data/processed/research_frame.csv")
    parser.add_argument("--universe", default="data/reference/sp500_universe.csv")
    parser.add_argument("--existing-watchlist", default="data/reference/tracked_universe.csv")
    parser.add_argument("--ledger", default="data/paper/paper_trading_ledger.csv")
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--output-dir", default="reports/symbol_discovery_rd")
    parser.add_argument("--top-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-avg-dollar-volume-20d", type=float, default=5_000_000.0)
    parser.add_argument("--min-market-cap", type=float, default=None)
    parser.add_argument("--earnings-blackout-days", type=int, default=1)
    parser.add_argument("--monitoring-days", type=int, default=5)
    parser.add_argument("--min-appearances", type=int, default=3)
    parser.add_argument("--min-discovery-score", type=float, default=60.0)
    args = parser.parse_args()

    research_frame = pd.read_csv(args.research_frame)
    universe = _read_csv_if_exists(args.universe)
    existing_watchlist = _read_csv_if_exists(args.existing_watchlist)
    ledger = _read_csv_if_exists(args.ledger)

    result = run_stock_discovery(
        research_frame,
        universe=None if universe.empty else universe,
        existing_watchlist=None if existing_watchlist.empty else existing_watchlist,
        as_of_date=args.as_of_date,
        config=DiscoveryConfig(
            top_n=50,
            min_price=args.min_price,
            min_avg_dollar_volume_20d=args.min_avg_dollar_volume_20d,
            min_market_cap=args.min_market_cap,
            earnings_blackout_days=args.earnings_blackout_days,
        ),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "monitoring_state.csv"
    previous_state = _read_csv_if_exists(state_path)

    lane_membership = _lane_membership(result.watchlists)
    top_candidates = select_top_fraction(result.candidates, top_fraction=args.top_fraction)
    monitored = build_monitoring_candidates(
        top_candidates,
        previous_state,
        lane_membership=lane_membership,
        active_symbols=_active_symbols(ledger),
        tracked_symbols=_symbols(existing_watchlist),
        as_of_date=result.as_of_date,
        monitoring_days=args.monitoring_days,
        min_appearances=args.min_appearances,
        min_discovery_score=args.min_discovery_score,
    )
    state = update_monitoring_state(previous_state, monitored, as_of_date=result.as_of_date)
    promotion_candidates = monitored.loc[monitored["promotion_status"].eq("eligible_for_review")]
    summary = build_summary(
        result,
        top_candidates=monitored,
        promotion_candidates=promotion_candidates,
        source_universe=universe,
        args=args,
    )

    monitored.to_csv(output_dir / "candidates.csv", index=False)
    state.to_csv(state_path, index=False)
    promotion_candidates.to_csv(output_dir / "promotion_candidates.csv", index=False)
    result.exclusions.to_csv(output_dir / "exclusions.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (output_dir / "report.md").write_text(
        render_report(summary, monitored, promotion_candidates)
    )
    print(f"wrote symbol discovery R&D artifacts to {output_dir}")


def select_top_fraction(candidates: pd.DataFrame, *, top_fraction: float) -> pd.DataFrame:
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    if candidates.empty:
        return candidates.copy()
    count = max(1, math.ceil(len(candidates) * top_fraction))
    return candidates.sort_values("discovery_score", ascending=False).head(count).copy()


def build_monitoring_candidates(
    candidates: pd.DataFrame,
    previous_state: pd.DataFrame,
    *,
    lane_membership: pd.DataFrame,
    active_symbols: set[str],
    tracked_symbols: set[str],
    as_of_date: pd.Timestamp,
    monitoring_days: int,
    min_appearances: int,
    min_discovery_score: float,
) -> pd.DataFrame:
    if candidates.empty:
        return _empty_candidates()
    state = _state_lookup(previous_state)
    output = candidates.copy()
    output["symbol"] = output["symbol"].astype(str).str.upper()
    output = output.merge(lane_membership, on="symbol", how="left")
    output["lanes_matched"] = output["lanes_matched"].fillna("")
    output["lane_count"] = output["lane_count"].fillna(0).astype(int)

    rows = []
    for _, row in output.iterrows():
        symbol = row["symbol"]
        prior = state.get(symbol, {})
        first_seen = pd.Timestamp(prior.get("first_seen", as_of_date))
        appearances = int(prior.get("appearances", 0)) + 1
        monitoring_age_days = max(0, (as_of_date.normalize() - first_seen.normalize()).days)
        reasons = _promotion_blockers(
            row,
            symbol=symbol,
            active_symbols=active_symbols,
            tracked_symbols=tracked_symbols,
            monitoring_age_days=monitoring_age_days,
            appearances=appearances,
            monitoring_days=monitoring_days,
            min_appearances=min_appearances,
            min_discovery_score=min_discovery_score,
        )
        enriched = row.to_dict()
        enriched.update(
            {
                "monitoring_status": "monitoring",
                "promotion_status": "eligible_for_review" if not reasons else "monitoring",
                "promotion_blockers": "; ".join(reasons),
                "first_seen": first_seen.date().isoformat(),
                "last_seen": as_of_date.date().isoformat(),
                "appearances": appearances,
                "monitoring_age_days": monitoring_age_days,
            }
        )
        rows.append(enriched)
    return pd.DataFrame(rows).sort_values(
        ["promotion_status", "discovery_score", "symbol"],
        ascending=[True, False, True],
    )


def update_monitoring_state(
    previous_state: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    columns = [
        "symbol",
        "name",
        "sector",
        "industry",
        "first_seen",
        "last_seen",
        "appearances",
        "latest_discovery_score",
        "max_discovery_score",
        "latest_lanes_matched",
        "promotion_status",
        "promotion_blockers",
    ]
    if previous_state.empty:
        state = pd.DataFrame(columns=columns)
    else:
        state = previous_state.copy()
        for column in columns:
            if column not in state:
                state[column] = pd.NA
        state = state.loc[:, columns]
        state["symbol"] = state["symbol"].astype(str).str.upper()

    by_symbol = {
        str(row["symbol"]).upper(): index
        for index, row in state.iterrows()
    }
    for _, row in candidates.iterrows():
        symbol = str(row["symbol"]).upper()
        latest_score = float(row.get("discovery_score", 0.0))
        payload = {
            "symbol": symbol,
            "name": row.get("name", ""),
            "sector": row.get("sector", ""),
            "industry": row.get("industry", ""),
            "first_seen": row.get("first_seen", as_of_date.date().isoformat()),
            "last_seen": row.get("last_seen", as_of_date.date().isoformat()),
            "appearances": int(row.get("appearances", 1)),
            "latest_discovery_score": latest_score,
            "max_discovery_score": latest_score,
            "latest_lanes_matched": row.get("lanes_matched", ""),
            "promotion_status": row.get("promotion_status", "monitoring"),
            "promotion_blockers": row.get("promotion_blockers", ""),
        }
        if symbol in by_symbol:
            index = by_symbol[symbol]
            previous_max = pd.to_numeric(
                pd.Series([state.loc[index, "max_discovery_score"]]), errors="coerce"
            ).fillna(0.0).iloc[0]
            payload["max_discovery_score"] = max(float(previous_max), latest_score)
            for key, value in payload.items():
                state.loc[index, key] = value
        else:
            state = pd.concat([state, pd.DataFrame([payload])], ignore_index=True)
            by_symbol[symbol] = len(state) - 1

    return state.sort_values(["promotion_status", "latest_discovery_score", "symbol"], ascending=[True, False, True])


def build_summary(
    result,
    *,
    top_candidates: pd.DataFrame,
    promotion_candidates: pd.DataFrame,
    source_universe: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    return {
        "as_of_date": result.as_of_date.date().isoformat(),
        "source_universe_count": int(source_universe["symbol"].nunique())
        if not source_universe.empty and "symbol" in source_universe
        else None,
        "eligible_after_filters": int(len(result.candidates)),
        "top_fraction": args.top_fraction,
        "monitored_candidate_count": int(len(top_candidates)),
        "promotion_candidate_count": int(len(promotion_candidates)),
        "monitoring_days_required": args.monitoring_days,
        "appearances_required": args.min_appearances,
        "min_discovery_score": args.min_discovery_score,
        "note": (
            "Research-only discovery feed. Candidates are monitored before manual review and "
            "are not added to the traded universe automatically."
        ),
    }


def render_report(
    summary: dict,
    candidates: pd.DataFrame,
    promotion_candidates: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# SignalForge Symbol Discovery R&D",
            "",
            f"As-of date: `{summary['as_of_date']}`",
            "",
            "## Gate Summary",
            "",
            f"- Source universe count: `{summary['source_universe_count']}`",
            f"- Eligible after filters: `{summary['eligible_after_filters']}`",
            f"- Top fraction monitored: `{summary['top_fraction']:.2%}`",
            f"- Monitored candidates: `{summary['monitored_candidate_count']}`",
            f"- Promotion-review candidates: `{summary['promotion_candidate_count']}`",
            f"- Monitoring days required: `{summary['monitoring_days_required']}`",
            f"- Appearances required: `{summary['appearances_required']}`",
            f"- Minimum discovery score: `{summary['min_discovery_score']:.1f}`",
            "",
            "## Promotion Review Candidates",
            "",
            _candidate_table(promotion_candidates.head(25)),
            "",
            "## Top Monitored Candidates",
            "",
            _candidate_table(candidates.head(50)),
            "",
        ]
    )


def _promotion_blockers(
    row: pd.Series,
    *,
    symbol: str,
    active_symbols: set[str],
    tracked_symbols: set[str],
    monitoring_age_days: int,
    appearances: int,
    monitoring_days: int,
    min_appearances: int,
    min_discovery_score: float,
) -> list[str]:
    blockers = []
    if symbol in active_symbols:
        blockers.append("already_active_in_paper")
    if symbol in tracked_symbols:
        blockers.append("already_tracked")
    if monitoring_age_days < monitoring_days:
        blockers.append(f"monitoring_age_below_{monitoring_days}_days")
    if appearances < min_appearances:
        blockers.append(f"appearances_below_{min_appearances}")
    if float(row.get("discovery_score", 0.0)) < min_discovery_score:
        blockers.append(f"discovery_score_below_{min_discovery_score:g}")
    return blockers


def _lane_membership(watchlists: dict[str, pd.DataFrame]) -> pd.DataFrame:
    lane_rows = []
    for lane, frame in watchlists.items():
        if frame is None or frame.empty or "symbol" not in frame:
            continue
        for symbol in frame["symbol"].dropna().astype(str).str.upper().unique():
            lane_rows.append({"symbol": symbol, "lane": lane})
    if not lane_rows:
        return pd.DataFrame(columns=["symbol", "lanes_matched", "lane_count"])
    lanes = pd.DataFrame(lane_rows)
    grouped = lanes.groupby("symbol")["lane"].agg(lambda values: ", ".join(sorted(set(values))))
    output = grouped.reset_index().rename(columns={"lane": "lanes_matched"})
    output["lane_count"] = output["lanes_matched"].str.split(", ").map(len)
    return output


def _candidate_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [
        "symbol",
        "name",
        "sector",
        "industry",
        "discovery_score",
        "lane_count",
        "lanes_matched",
        "return_20d",
        "return_60d",
        "avg_dollar_volume_20d",
        "monitoring_age_days",
        "appearances",
        "promotion_status",
        "promotion_blockers",
        "why_flagged",
        "risk_flags",
    ]
    display = frame.loc[:, [column for column in columns if column in frame]].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    rows = [list(display.columns), ["---"] * len(display.columns)]
    rows.extend(display.astype(str).values.tolist())
    return "\n".join("| " + " | ".join(str(value) for value in row) + " |" for row in rows)


def _read_csv_if_exists(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def _symbols(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "symbol" not in frame:
        return set()
    return set(frame["symbol"].dropna().astype(str).str.upper())


def _active_symbols(ledger: pd.DataFrame) -> set[str]:
    if ledger.empty or not {"status", "symbol"}.issubset(ledger.columns):
        return set()
    active = ledger.loc[ledger["status"].isin(["planned", "open"]), "symbol"]
    return set(active.dropna().astype(str).str.upper())


def _state_lookup(state: pd.DataFrame) -> dict[str, dict]:
    if state.empty or "symbol" not in state:
        return {}
    normalized = state.copy()
    normalized["symbol"] = normalized["symbol"].astype(str).str.upper()
    return {
        row["symbol"]: row
        for row in normalized.to_dict(orient="records")
    }


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "name",
            "sector",
            "industry",
            "discovery_score",
            "promotion_status",
            "promotion_blockers",
        ]
    )


if __name__ == "__main__":
    main()
