"""S&P 500 universe management.

Provides the current S&P 500 constituent list with optional
Wikipedia refresh. Ships a static fallback list that requires
no network access.
"""

import datetime
import urllib.request
from html.parser import HTMLParser
from typing import Optional


# Static S&P 500 list (updated 2025-03). ~503 tickers due to dual-class shares.
# Dots replaced with hyphens for yfinance (BRK.B -> BRK-B).
_SP500_TICKERS: list[str] = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM",
    "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM",
    "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP",
    "AMT", "AMZN", "ANET", "AON", "AOS", "APA", "APD", "APH", "APO", "APP",
    "APTV", "ARE", "ARES", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP",
    "AZO", "BA", "BAC", "BALL", "BAX", "BBY", "BDX", "BEN", "BF-B", "BG",
    "BIIB", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", "BRO",
    "BSX", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE",
    "CBRE", "CCI", "CCL", "CDNS", "CDW", "CEG", "CF", "CFG", "CHD", "CHRW",
    "CHTR", "CI", "CIEN", "CINF", "CL", "CLX", "CMCSA", "CME", "CMG", "CMI",
    "CMS", "CNC", "CNP", "COF", "COIN", "COO", "COP", "COR", "COST", "CPAY",
    "CPB", "CPRT", "CPT", "CRH", "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX",
    "CTAS", "CTRA", "CTSH", "CTVA", "CVNA", "CVS", "CVX", "D", "DAL", "DASH",
    "DD", "DDOG", "DE", "DECK", "DELL", "DG", "DGX", "DHI", "DHR", "DIS",
    "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA",
    "DVN", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL",
    "ELV", "EME", "EMR", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ERIE", "ES",
    "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXE", "EXPD", "EXPE", "EXR",
    "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FICO", "FIS",
    "FISV", "FITB", "FIX", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD",
    "GDDY", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW", "GM",
    "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS",
    "HBAN", "HCA", "HD", "HIG", "HII", "HLT", "HOLX", "HON", "HOOD", "HPE",
    "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBKR", "IBM",
    "ICE", "IDXX", "IEX", "IFF", "INCY", "INTC", "INTU", "INVH", "IP", "IQV",
    "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI",
    "JKHY", "JNJ", "JPM", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC",
    "KMB", "KMI", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH", "LHX",
    "LII", "LIN", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS",
    "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK",
    "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MKC", "MLM", "MMM", "MNST",
    "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MS", "MSCI",
    "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE",
    "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS",
    "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC",
    "ON", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PAYC", "PAYX", "PCAR", "PCG",
    "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD",
    "PLTR", "PM", "PNC", "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU",
    "PSA", "PSX", "PTC", "PWR", "PYPL", "QCOM", "RCL", "REG",
    "REGN", "RF", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG",
    "RTX", "RVTY", "SBAC", "SBUX", "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA",
    "SNPS", "SO", "SOLV", "SPG", "SPGI", "SRE", "STE", "STLD", "STT",
    "STX", "STZ", "SW", "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP",
    "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TGT", "TJX", "TMO",
    "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN",
    "TT", "TTD", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS",
    "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VICI", "VLO", "VLTO",
    "VMC", "VRSK", "VRSN", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT",
    "WBD", "WDAY", "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB",
    "WSM", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "YUM",
    "ZBH", "ZBRA", "ZTS",
]

_CACHED_TICKERS: Optional[list[str]] = None
_LAST_REFRESH: Optional[datetime.datetime] = None


def get_sp500_tickers(refresh: bool = False) -> list[str]:
    """Return current S&P 500 ticker list.

    Args:
        refresh: If True, attempt to scrape Wikipedia for the latest
                 list. Falls back to static list on failure.

    Returns:
        List of ~503 ticker strings.
    """
    global _CACHED_TICKERS, _LAST_REFRESH

    if not refresh and _CACHED_TICKERS is not None:
        return _CACHED_TICKERS.copy()

    if refresh:
        try:
            tickers = _scrape_wikipedia_sp500()
            if len(tickers) >= 490:
                _CACHED_TICKERS = tickers
                _LAST_REFRESH = datetime.datetime.now()
                return _CACHED_TICKERS.copy()
        except Exception:
            pass

    _CACHED_TICKERS = _SP500_TICKERS.copy()
    return _CACHED_TICKERS.copy()


class _SP500TableParser(HTMLParser):
    """Minimal HTML parser for Wikipedia S&P 500 table."""

    def __init__(self):
        super().__init__()
        self.tickers: list[str] = []
        self._in_table = False
        self._in_tbody = False
        self._in_row = False
        self._in_td = False
        self._in_a = False
        self._col_idx = 0
        self._table_count = 0

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "table" and "wikitable" in attrs_dict.get("class", ""):
            self._table_count += 1
            if self._table_count == 1:
                self._in_table = True
        if self._in_table:
            if tag == "tbody":
                self._in_tbody = True
            elif tag == "tr":
                self._in_row = True
                self._col_idx = 0
            elif tag == "td":
                self._in_td = True
                self._col_idx += 1
            elif tag == "a" and self._in_td and self._col_idx == 1:
                self._in_a = True

    def handle_endtag(self, tag):
        if tag == "table" and self._in_table:
            self._in_table = False
        elif tag == "tbody":
            self._in_tbody = False
        elif tag == "tr":
            self._in_row = False
        elif tag == "td":
            self._in_td = False
        elif tag == "a":
            self._in_a = False

    def handle_data(self, data):
        if self._in_a and self._col_idx == 1 and self._in_table:
            ticker = data.strip()
            if ticker and ticker.isalpha() or "-" in ticker or "." in ticker:
                self.tickers.append(ticker)


def _scrape_wikipedia_sp500() -> list[str]:
    """Scrape Wikipedia for current S&P 500 constituents."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8")

    parser = _SP500TableParser()
    parser.feed(html)
    tickers = [t.replace(".", "-") for t in parser.tickers]
    return sorted(set(tickers))
