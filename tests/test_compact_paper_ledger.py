import pandas as pd

from scripts.compact_paper_ledger import compact_ledger


def test_compact_ledger_splits_skipped_rows_from_lifecycle_rows():
    ledger = pd.DataFrame(
        {
            "order_id": ["one", "two", "three"],
            "status": ["open", "skipped", "closed"],
            "symbol": ["A", "B", "C"],
        }
    )

    compacted, skipped = compact_ledger(ledger)

    assert list(compacted["order_id"]) == ["one", "three"]
    assert list(skipped["order_id"]) == ["two"]
