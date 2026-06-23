import pandas as pd

from signalforge.validation import walk_forward_splits


def test_walk_forward_split_applies_purge_before_validation():
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", "2020-03-31", freq="D")})

    split = next(
        walk_forward_splits(
            frame,
            first_train_start="2020-01-01",
            first_validation_start="2020-03-01",
            validation_months=1,
            purge_days=5,
        )
    )

    train_dates = frame.loc[split.train_index, "date"]
    validation_dates = frame.loc[split.validation_index, "date"]

    assert train_dates.max() == pd.Timestamp("2020-02-24")
    assert validation_dates.min() == pd.Timestamp("2020-03-01")
    assert validation_dates.max() == pd.Timestamp("2020-03-31")
