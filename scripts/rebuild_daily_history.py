from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.run_daily_paper_workflow import HISTORY_COLUMNS, build_history_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild daily_runs/history.csv from timestamped run_summary.json files."
    )
    parser.add_argument("--tracking-root", default="reports/daily_runs")
    parser.add_argument("--history", default="reports/daily_runs/history.csv")
    args = parser.parse_args()

    rows = rebuild_history_rows(Path(args.tracking_root))
    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rebuilt {len(rows)} history rows at {history_path}")


def rebuild_history_rows(tracking_root: Path) -> list[dict[str, object]]:
    rows = []
    for summary_path in sorted(tracking_root.glob("*/run_summary.json")):
        summary = json.loads(summary_path.read_text())
        rows.append(build_history_row(summary))
    return rows


if __name__ == "__main__":
    main()
