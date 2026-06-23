from __future__ import annotations

from pathlib import Path

import pandas as pd

from signalforge.data import load_price_csv
from signalforge.paper import summarize_paper_account


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_paper_summary(
    ledger_path: str | Path = "data/paper/paper_trading_ledger.csv",
    prices_path: str | Path = "data/raw/yahoo_prices.csv",
    initial_capital: float = 2_000.0,
) -> dict:
    """Load the paper ledger and latest prices, return an account summary."""
    ledger_file = _resolve(ledger_path)
    prices_file = _resolve(prices_path)

    if not ledger_file.exists():
        return {"error": f"ledger not found: {ledger_file}"}
    if not prices_file.exists():
        return {"error": f"prices not found: {prices_file}"}

    ledger = pd.read_csv(ledger_file)
    prices = load_price_csv(str(prices_file))
    return summarize_paper_account(ledger, prices, initial_capital=initial_capital)


def load_paper_positions(
    ledger_path: str | Path = "data/paper/paper_trading_ledger.csv",
) -> list[dict]:
    """Load open positions from the paper ledger."""
    ledger_file = _resolve(ledger_path)
    if not ledger_file.exists():
        return []

    ledger = pd.read_csv(ledger_file)
    open_positions = ledger.loc[ledger["status"] == "open"]
    if open_positions.empty:
        return []

    return open_positions.fillna("").to_dict(orient="records")


def load_paper_ledger(
    ledger_path: str | Path = "data/paper/paper_trading_ledger.csv",
) -> list[dict]:
    """Load the full paper ledger."""
    ledger_file = _resolve(ledger_path)
    if not ledger_file.exists():
        return []

    ledger = pd.read_csv(ledger_file)
    return ledger.fillna("").to_dict(orient="records")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p
