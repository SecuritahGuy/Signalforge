import pandas as pd

from signalforge.universe import (
    BroadUniverseConfig,
    append_benchmark,
    build_broad_universe,
    parse_nasdaq_listed_text,
    parse_other_listed_text,
    parse_sp500_constituents_html,
)


def test_parse_sp500_constituents_html_maps_to_universe_contract():
    html = """
    <html><body>
      <table><tr><th>Other</th></tr><tr><td>ignore</td></tr></table>
      <table>
        <tr>
          <th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th>
        </tr>
        <tr>
          <td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td>
          <td>Multi-Sector Holdings</td>
        </tr>
        <tr>
          <td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td>
          <td>Technology Hardware</td>
        </tr>
      </table>
    </body></html>
    """

    universe = parse_sp500_constituents_html(html)

    assert universe["symbol"].tolist() == ["BRK-B", "AAPL"]
    assert set(["symbol", "name", "category", "sector", "industry"]).issubset(
        universe.columns
    )
    assert universe.loc[0, "category"] == "sp500"


def test_nasdaq_trader_parsers_filter_etfs_tests_and_non_common_rows():
    nasdaq_text = (
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
        "Round Lot Size|ETF|NextShares\n"
        "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
        "QQQ|Invesco QQQ Trust|G|N|N|100|Y|N\n"
        "TEST|Test Row|G|Y|N|100|N|N\n"
        "WARR|Example Warrants|G|N|N|100|N|N\n"
        "File Creation Time: 0521202618:01|||||||\n"
    )
    other_text = (
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "IBM|International Business Machines Corporation Common Stock|N|IBM|N|100|N|IBM\n"
        "SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY\n"
        "UNIT|Example Units|A|UNIT|N|100|N|UNIT\n"
        "File Creation Time: 0521202618:01|||||||\n"
    )

    nasdaq = parse_nasdaq_listed_text(nasdaq_text)
    other = parse_other_listed_text(other_text)

    assert nasdaq["symbol"].tolist() == ["AAPL"]
    assert other["symbol"].tolist() == ["IBM"]
    assert other.loc[0, "exchange"] == "NYSE"


def test_build_broad_universe_normalizes_dedupes_and_appends_benchmark(monkeypatch):
    raw = pd.DataFrame(
        {
            "symbol": ["msft", "MSFT", "AAPL"],
            "name": ["Microsoft", "Microsoft duplicate", "Apple"],
            "category": ["sp500", "sp500", "sp500"],
            "sector": ["Information Technology"] * 3,
            "industry": ["Software", "Software", "Hardware"],
        }
    )
    monkeypatch.setattr("signalforge.universe.load_sp500_universe", lambda: raw)

    universe = build_broad_universe(BroadUniverseConfig(source="sp500"))

    assert universe["symbol"].tolist() == ["AAPL", "MSFT", "SPY"]
    assert universe.loc[universe["symbol"] == "SPY", "category"].iloc[0] == "benchmark"


def test_append_benchmark_does_not_duplicate_existing_symbol():
    universe = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "name": ["Existing SPY"],
            "category": ["benchmark"],
            "sector": ["ETF"],
            "industry": ["Broad Market"],
        }
    )

    result = append_benchmark(universe)

    assert len(result) == 1
