from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    train_index: pd.Index
    validation_index: pd.Index


def walk_forward_splits(
    frame: pd.DataFrame,
    *,
    date_col: str = "date",
    first_train_start: str | pd.Timestamp,
    first_validation_start: str | pd.Timestamp,
    validation_months: int = 12,
    purge_days: int = 0,
    embargo_days: int = 0,
) -> Iterator[WalkForwardSplit]:
    """Yield expanding-window walk-forward splits.

    Purge removes training rows immediately before the validation window.
    Embargo removes training rows immediately after the validation window.
    """
    if validation_months <= 0:
        raise ValueError("validation_months must be positive")
    if purge_days < 0 or embargo_days < 0:
        raise ValueError("purge_days and embargo_days must be non-negative")
    if date_col not in frame.columns:
        raise KeyError(f"frame is missing required column: {date_col}")

    dates = pd.to_datetime(frame[date_col])
    train_start = pd.Timestamp(first_train_start)
    validation_start = pd.Timestamp(first_validation_start)
    max_date = dates.max()

    while validation_start <= max_date:
        validation_end = (
            validation_start + pd.DateOffset(months=validation_months) - pd.Timedelta(days=1)
        )
        if validation_end > max_date:
            break

        purge_start = validation_start - pd.Timedelta(days=purge_days)
        embargo_end = validation_end + pd.Timedelta(days=embargo_days)

        train_mask = (dates >= train_start) & (dates < purge_start)
        train_mask |= (dates > embargo_end) & (dates < validation_start)
        validation_mask = (dates >= validation_start) & (dates <= validation_end)

        train_index = frame.index[train_mask]
        validation_index = frame.index[validation_mask]

        if len(train_index) and len(validation_index):
            yield WalkForwardSplit(
                train_start=train_start,
                train_end=purge_start - pd.Timedelta(days=1),
                validation_start=validation_start,
                validation_end=validation_end,
                train_index=train_index,
                validation_index=validation_index,
            )

        validation_start = validation_start + pd.DateOffset(months=validation_months)
