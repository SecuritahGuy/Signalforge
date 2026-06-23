import pandas as pd

from signalforge.labels import excess_forward_return, forward_return


def test_forward_return_is_grouped_by_symbol():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "symbol": ["A", "A", "B", "B"],
            "adj_close": [10.0, 11.0, 20.0, 18.0],
        }
    )

    result = forward_return(prices, horizon=1)

    assert round(result.iloc[0], 4) == 0.1
    assert pd.isna(result.iloc[1])
    assert round(result.iloc[2], 4) == -0.1
    assert pd.isna(result.iloc[3])


def test_excess_forward_return_aligns_benchmark_by_date():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["A", "A"],
            "adj_close": [100.0, 110.0],
        }
    )
    benchmark = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "adj_close": [200.0, 210.0],
        }
    )

    result = excess_forward_return(prices, benchmark, horizon=1)

    assert round(result.iloc[0], 4) == 0.05
