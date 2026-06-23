from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from signalforge.data import PRICE_COLUMNS, validate_price_frame
from signalforge.intraday import normalize_yfinance_intraday


def download_yahoo_prices(
    symbols: Sequence[str],
    *,
    start: str,
    end: str | None = None,
    auto_adjust: bool = False,
    progress: bool = False,
) -> pd.DataFrame:
    """Download daily OHLCV data through yfinance and normalize it.

    yfinance is a third-party Yahoo Finance client, not an official Yahoo SDK.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for Yahoo downloads. Install with "
            "`pip install -e '.[data]'` or `pip install yfinance`."
        ) from exc

    normalized_symbols = sorted({symbol.upper() for symbol in symbols})
    if not normalized_symbols:
        raise ValueError("symbols must not be empty")

    raw = yf.download(
        tickers=normalized_symbols,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=auto_adjust,
        actions=False,
        group_by="ticker",
        progress=progress,
        threads=True,
    )
    return normalize_yfinance_download(raw, normalized_symbols)


def download_yahoo_intraday_marks(
    symbols: Sequence[str],
    *,
    interval: str = "1m",
    period: str = "1d",
    progress: bool = False,
) -> pd.DataFrame:
    """Download recent intraday OHLCV marks through yfinance and normalize them."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for Yahoo downloads. Install with "
            "`pip install -e '.[data]'` or `pip install yfinance`."
        ) from exc

    normalized_symbols = sorted({symbol.upper() for symbol in symbols})
    if not normalized_symbols:
        raise ValueError("symbols must not be empty")

    raw = yf.download(
        tickers=normalized_symbols,
        period=period,
        interval=interval,
        auto_adjust=False,
        actions=False,
        group_by="ticker",
        progress=progress,
        threads=True,
    )
    return normalize_yfinance_intraday(raw, normalized_symbols)


def normalize_yfinance_download(raw: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    """Normalize yfinance's single- or multi-ticker frame into PRICE_COLUMNS."""
    if raw.empty:
        raise ValueError("yfinance returned no price data")

    frames = []
    normalized_symbols = [symbol.upper() for symbol in symbols]

    if isinstance(raw.columns, pd.MultiIndex):
        for symbol in normalized_symbols:
            if symbol not in raw.columns.get_level_values(0):
                continue
            symbol_frame = raw[symbol].copy()
            frames.append(_normalize_symbol_frame(symbol_frame, symbol))
    else:
        if len(normalized_symbols) != 1:
            raise ValueError("single-index yfinance data requires exactly one symbol")
        frames.append(_normalize_symbol_frame(raw.copy(), normalized_symbols[0]))

    if not frames:
        raise ValueError("no requested symbols were present in yfinance response")

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.dropna(subset=["open", "high", "low", "close", "adj_close", "volume"])
    validate_price_frame(prices)
    return prices.loc[:, PRICE_COLUMNS].sort_values(["symbol", "date"]).reset_index(drop=True)


def _normalize_symbol_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    normalized = frame.rename(columns=rename_map)
    if "adj_close" not in normalized.columns and "close" in normalized.columns:
        normalized["adj_close"] = normalized["close"]
    normalized = normalized.reset_index()
    date_column = "Date" if "Date" in normalized.columns else normalized.columns[0]
    normalized = normalized.rename(columns={date_column: "date"})
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.tz_localize(None)
    normalized["symbol"] = symbol
    return normalized
