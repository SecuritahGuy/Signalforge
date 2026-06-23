from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove skipped-only rows from the persistent paper ledger."
    )
    parser.add_argument("--ledger", default="data/paper/paper_trading_ledger.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--archive", default="data/paper/paper_trading_skipped_archive.csv")
    parser.add_argument("--summary-output", default="reports/paper_ledger_compaction_summary.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    output_path = Path(args.output) if args.output else ledger_path
    archive_path = Path(args.archive)
    summary_path = Path(args.summary_output)

    ledger = pd.read_csv(ledger_path)
    compacted, skipped = compact_ledger(ledger)
    summary = {
        "ledger": str(ledger_path),
        "output": str(output_path),
        "archive": str(archive_path),
        "input_rows": int(len(ledger)),
        "output_rows": int(len(compacted)),
        "archived_skipped_rows": int(len(skipped)),
        "dry_run": bool(args.dry_run),
    }

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        compacted.to_csv(output_path, index=False)
        if not skipped.empty:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            _append_archive(archive_path, skipped)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


def compact_ledger(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    skipped_mask = ledger["status"] == "skipped"
    compacted = ledger.loc[~skipped_mask].copy().reset_index(drop=True)
    skipped = ledger.loc[skipped_mask].copy().reset_index(drop=True)
    return compacted, skipped


def _append_archive(path: Path, skipped: pd.DataFrame) -> None:
    if path.exists():
        existing = pd.read_csv(path)
        archived = pd.concat([existing, skipped], ignore_index=True)
        archived = archived.drop_duplicates(subset=["order_id"], keep="last")
    else:
        archived = skipped
    archived.to_csv(path, index=False)


if __name__ == "__main__":
    main()
