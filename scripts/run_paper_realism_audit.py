from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from signalforge.data import load_price_csv
from signalforge.paper_audit import (
    PaperAuditConfig,
    render_audit_markdown,
    run_paper_realism_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the paper account for realism and ledger consistency."
    )
    parser.add_argument("--ledger", default="data/paper/paper_trading_ledger.csv")
    parser.add_argument("--prices", default="data/raw/yahoo_prices.csv")
    parser.add_argument("--account-summary", default="reports/paper_account_summary.json")
    parser.add_argument("--output-prefix", default="reports/paper_realism_audit")
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--initial-capital", type=float, default=2_000.0)
    parser.add_argument("--max-sector-exposure", type=float, default=0.35)
    parser.add_argument("--max-gross-exposure", type=float, default=1.0)
    parser.add_argument("--max-price-staleness-days", type=int, default=3)
    parser.add_argument("--fail-on-errors", action="store_true")
    args = parser.parse_args()

    ledger = pd.read_csv(args.ledger)
    prices = load_price_csv(args.prices)
    account_summary = _load_json(Path(args.account_summary))
    audit = run_paper_realism_audit(
        ledger,
        prices,
        account_summary=account_summary,
        as_of_date=args.as_of_date,
        config=PaperAuditConfig(
            initial_capital=args.initial_capital,
            max_sector_exposure=args.max_sector_exposure,
            max_gross_exposure=args.max_gross_exposure,
            max_price_staleness_days=args.max_price_staleness_days,
        ),
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_name(output_prefix.name + "_summary.json").write_text(
        json.dumps(audit, indent=2, default=str) + "\n"
    )
    output_prefix.with_name(output_prefix.name + "_report.md").write_text(
        render_audit_markdown(audit)
    )
    print(
        f"wrote paper realism audit with {audit['error_count']} errors and "
        f"{audit['warning_count']} warnings"
    )
    if args.fail_on_errors and audit["error_count"] > 0:
        raise SystemExit(1)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
