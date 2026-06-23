import pandas as pd
import pytest

from signalforge.providers.yahoo import normalize_yfinance_download


def test_normalize_yfinance_download_handles_multi_ticker_frame():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    columns = pd.MultiIndex.from_product(
        [["AAPL", "SPY"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )
    raw = pd.DataFrame(
        [
            [100, 101, 99, 100, 100, 1_000, 200, 202, 198, 201, 201, 2_000],
            [101, 102, 100, 101, 101, 1_100, 201, 203, 199, 202, 202, 2_100],
        ],
        index=dates,
        columns=columns,
    )

    prices = normalize_yfinance_download(raw, ["AAPL", "SPY"])

    assert prices.columns.tolist() == [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert prices["symbol"].tolist() == ["AAPL", "AAPL", "SPY", "SPY"]


def test_normalize_yfinance_download_handles_single_ticker_frame():
    raw = pd.DataFrame(
        {
            "Open": [100],
            "High": [101],
            "Low": [99],
            "Close": [100],
            "Adj Close": [100],
            "Volume": [1_000],
        },
        index=pd.to_datetime(["2024-01-01"]),
    )

    prices = normalize_yfinance_download(raw, ["AAPL"])

    assert prices.loc[0, "symbol"] == "AAPL"
    assert prices.loc[0, "adj_close"] == 100


def test_normalize_yfinance_download_rejects_empty_response():
    with pytest.raises(ValueError, match="no price data"):
        normalize_yfinance_download(pd.DataFrame(), ["AAPL"])
