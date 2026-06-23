from __future__ import annotations

import csv
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from io import StringIO
from urllib.request import Request, urlopen

import pandas as pd

from signalforge.data import UNIVERSE_COLUMNS

DEFAULT_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

BENCHMARK_ROWS = {
    "SPY": {
        "symbol": "SPY",
        "name": "SPDR S&P 500 ETF",
        "category": "benchmark",
        "sector": "ETF",
        "industry": "Broad Market",
        "notes": "Default excess-return benchmark",
    }
}


@dataclass(frozen=True)
class BroadUniverseConfig:
    source: str = "sp500"
    include_benchmark: bool = True
    benchmark_symbol: str = "SPY"
    limit: int | None = None


def build_broad_universe(config: BroadUniverseConfig | None = None) -> pd.DataFrame:
    """Build a broad discovery universe without editing the starter watchlist."""
    cfg = config or BroadUniverseConfig()
    if cfg.source == "sp500":
        universe = load_sp500_universe()
    elif cfg.source == "us_listed":
        universe = load_us_listed_universe()
    else:
        raise ValueError(f"unsupported broad universe source: {cfg.source!r}")

    universe = _normalize_universe(universe)
    if cfg.limit is not None:
        if cfg.limit <= 0:
            raise ValueError("limit must be positive when provided")
        universe = universe.head(cfg.limit)
    if cfg.include_benchmark:
        universe = append_benchmark(universe, benchmark_symbol=cfg.benchmark_symbol)
    return universe.reset_index(drop=True)


def load_sp500_universe(*, source_url: str = DEFAULT_SP500_URL) -> pd.DataFrame:
    """Load current S&P 500 constituents from the public Wikipedia table."""
    html = _read_url_text(source_url)
    return parse_sp500_constituents_html(html)


def parse_sp500_constituents_html(html: str) -> pd.DataFrame:
    """Parse the S&P 500 constituents table into SignalForge's universe schema."""
    tables = _HTMLTableParser.parse(html)
    for table in tables:
        if not table:
            continue
        header = table[0]
        header_index = {column.strip(): index for index, column in enumerate(header)}
        required = {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry"}
        if not required.issubset(header_index):
            continue

        rows = []
        for raw_row in table[1:]:
            if len(raw_row) <= max(header_index.values()):
                continue
            symbol = _normalize_symbol(raw_row[header_index["Symbol"]])
            if not symbol:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "name": raw_row[header_index["Security"]].strip(),
                    "category": "sp500",
                    "sector": raw_row[header_index["GICS Sector"]].strip(),
                    "industry": raw_row[header_index["GICS Sub-Industry"]].strip(),
                    "notes": "Generated from S&P 500 constituents source",
                }
            )
        if rows:
            return pd.DataFrame(rows)
    raise ValueError("could not find an S&P 500 constituents table in source HTML")


def load_us_listed_universe(
    *,
    nasdaq_url: str = NASDAQ_LISTED_URL,
    other_url: str = OTHER_LISTED_URL,
) -> pd.DataFrame:
    """Load a broad NASDAQ Trader-listed US equity universe.

    This source gives broad exchange coverage and ETF/test-issue flags, but not sector
    metadata. Rows use Unknown sector/industry until a fundamentals source is joined.
    """
    nasdaq = parse_nasdaq_listed_text(_read_url_text(nasdaq_url))
    other = parse_other_listed_text(_read_url_text(other_url))
    return pd.concat([nasdaq, other], ignore_index=True)


def parse_nasdaq_listed_text(text: str) -> pd.DataFrame:
    rows = []
    for row in _pipe_rows(text):
        symbol = _normalize_symbol(row.get("Symbol", ""))
        if not symbol or row.get("ETF") != "N" or row.get("Test Issue") != "N":
            continue
        security_name = row.get("Security Name", "").strip()
        if _is_non_common_listing(security_name):
            continue
        rows.append(
            {
                "symbol": symbol,
                "name": security_name,
                "category": "us_listed",
                "sector": "Unknown",
                "industry": "Unknown",
                "exchange": "NASDAQ",
                "security_type": "Common Stock",
                "notes": "Generated from NASDAQ Trader nasdaqlisted.txt",
            }
        )
    return pd.DataFrame(rows)


def parse_other_listed_text(text: str) -> pd.DataFrame:
    exchange_map = {
        "A": "NYSE American",
        "N": "NYSE",
        "P": "NYSE Arca",
        "Z": "Cboe BZX",
        "V": "IEX",
    }
    rows = []
    for row in _pipe_rows(text):
        symbol = _normalize_symbol(row.get("ACT Symbol", ""))
        if not symbol or row.get("ETF") != "N" or row.get("Test Issue") != "N":
            continue
        security_name = row.get("Security Name", "").strip()
        if _is_non_common_listing(security_name):
            continue
        rows.append(
            {
                "symbol": symbol,
                "name": security_name,
                "category": "us_listed",
                "sector": "Unknown",
                "industry": "Unknown",
                "exchange": exchange_map.get(row.get("Exchange", ""), row.get("Exchange", "")),
                "security_type": "Common Stock",
                "notes": "Generated from NASDAQ Trader otherlisted.txt",
            }
        )
    return pd.DataFrame(rows)


def append_benchmark(
    universe: pd.DataFrame,
    *,
    benchmark_symbol: str = "SPY",
) -> pd.DataFrame:
    benchmark_symbol = benchmark_symbol.upper()
    if benchmark_symbol in set(universe["symbol"].str.upper()):
        return universe
    if benchmark_symbol not in BENCHMARK_ROWS:
        raise ValueError(f"unsupported benchmark symbol: {benchmark_symbol!r}")
    benchmark = pd.DataFrame([BENCHMARK_ROWS[benchmark_symbol]])
    for column in universe.columns:
        if column not in benchmark.columns:
            benchmark[column] = pd.NA
    return pd.concat([universe, benchmark[universe.columns]], ignore_index=True)


def _normalize_universe(universe: pd.DataFrame) -> pd.DataFrame:
    missing = set(UNIVERSE_COLUMNS).difference(universe.columns)
    if missing:
        raise KeyError(f"broad universe is missing required columns: {sorted(missing)}")
    normalized = universe.copy()
    normalized["symbol"] = normalized["symbol"].map(_normalize_symbol)
    normalized = normalized.loc[normalized["symbol"] != ""]
    normalized = normalized.drop_duplicates(subset=["symbol"], keep="first")
    core_columns = list(UNIVERSE_COLUMNS)
    extra_columns = [column for column in normalized.columns if column not in core_columns]
    return normalized.loc[:, [*core_columns, *extra_columns]].sort_values("symbol")


def _pipe_rows(text: str) -> list[dict[str, str]]:
    lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.startswith("File Creation Time")
    ]
    return list(csv.DictReader(StringIO("\n".join(lines)), delimiter="|"))


def _read_url_text(url: str) -> str:
    request = Request(
        url,
        headers={"User-Agent": "SignalForge research universe builder/0.1"},
    )
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _normalize_symbol(symbol: str) -> str:
    normalized = unescape(str(symbol)).strip().upper()
    normalized = normalized.replace(".", "-")
    normalized = normalized.replace("/", "-")
    return normalized


def _is_non_common_listing(security_name: str) -> bool:
    lowered = security_name.lower()
    excluded_terms = (
        " warrant",
        " warrants",
        " right",
        " rights",
        " unit",
        " units",
        " preferred",
        " depositary",
        " note",
        " notes",
        " bond",
        " bonds",
        " debenture",
        " acquisition corp",
    )
    return any(term in lowered for term in excluded_terms)


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._table_depth = 0
        self._current_table: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    @classmethod
    def parse(cls, html: str) -> list[list[list[str]]]:
        parser = cls()
        parser.feed(html)
        return parser.tables

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._current_table = []
            return
        if self._table_depth != 1:
            return
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._table_depth != 1 and tag != "table":
            return
        if tag in {"td", "th"} and self._current_cell is not None:
            cell_text = " ".join("".join(self._current_cell).split())
            self._current_row = self._current_row or []
            self._current_row.append(cell_text)
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if self._current_row:
                self._current_table.append(self._current_row)
            self._current_row = None
        elif tag == "table" and self._table_depth:
            if self._table_depth == 1 and self._current_table:
                self.tables.append(self._current_table)
            self._table_depth -= 1
