"""Market definitions (US / India) and safe symbol resolution."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo


class SymbolError(ValueError):
    """The user supplied a symbol we refuse to look up."""


@dataclass(frozen=True)
class Market:
    code: str
    label: str
    currency: str
    locale: str
    tz: str
    open_hm: tuple
    close_hm: tuple
    index_symbol: str   # broad index used as market-context features
    vix_symbol: str     # volatility index used as market-context features
    cost_bps: float     # rough one-way trading cost used in backtests
    cooldown: int       # min bars between chart signals


US = Market("US", "US (NYSE/NASDAQ)", "USD", "en-US", "America/New_York",
            (9, 30), (16, 0), "^GSPC", "^VIX", cost_bps=2.0, cooldown=2)
# India: STT (0.1% each side on delivery) + stamp duty + exchange fees + slippage ~ 12 bps/side.
IN = Market("IN", "India (NSE/BSE)", "INR", "en-IN", "Asia/Kolkata",
            (9, 15), (15, 30), "^NSEI", "^INDIAVIX", cost_bps=12.0, cooldown=3)

NIFTY = {
    "RELIANCE": "Reliance Industries", "TCS": "Tata Consultancy Services", "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank", "INFY": "Infosys", "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel", "ITC": "ITC", "HINDUNILVR": "Hindustan Unilever",
    "LT": "Larsen & Toubro", "KOTAKBANK": "Kotak Mahindra Bank", "AXISBANK": "Axis Bank",
    "BAJFINANCE": "Bajaj Finance", "BAJAJFINSV": "Bajaj Finserv", "MARUTI": "Maruti Suzuki",
    "ASIANPAINT": "Asian Paints", "TITAN": "Titan Company", "SUNPHARMA": "Sun Pharma",
    "NTPC": "NTPC", "POWERGRID": "Power Grid Corp", "ONGC": "ONGC", "ULTRACEMCO": "UltraTech Cement",
    "WIPRO": "Wipro", "HCLTECH": "HCL Technologies", "TECHM": "Tech Mahindra",
    "NESTLEIND": "Nestle India", "TATASTEEL": "Tata Steel", "JSWSTEEL": "JSW Steel",
    "ADANIENT": "Adani Enterprises", "ADANIPORTS": "Adani Ports", "COALINDIA": "Coal India",
    "M&M": "Mahindra & Mahindra", "DRREDDY": "Dr. Reddy's", "CIPLA": "Cipla",
    "EICHERMOT": "Eicher Motors", "HEROMOTOCO": "Hero MotoCorp", "BAJAJ-AUTO": "Bajaj Auto",
    "GRASIM": "Grasim", "HINDALCO": "Hindalco", "BPCL": "BPCL", "TATACONSUM": "Tata Consumer",
    "BRITANNIA": "Britannia", "DIVISLAB": "Divi's Laboratories", "APOLLOHOSP": "Apollo Hospitals",
    "INDUSINDBK": "IndusInd Bank", "SBILIFE": "SBI Life", "HDFCLIFE": "HDFC Life", "TRENT": "Trent",
}
# Liquid US large caps for the cross-sectional model. NB: today's survivors, so backtests on them are optimistic.
US_LARGE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "ADBE", "CRM", "AMD", "INTC", "CSCO",
    "QCOM", "TXN", "NFLX", "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "BRK-B", "UNH", "JNJ", "LLY", "PFE", "MRK",
    "ABBV", "TMO", "WMT", "COST", "HD", "MCD", "NKE", "KO", "PEP", "PG", "XOM", "CVX", "CAT", "BA", "GE", "HON",
    "UNP", "DIS", "T", "VZ",
]
IN_INDICES = {"^NSEI": "Nifty 50", "^NSEBANK": "Nifty Bank", "^BSESN": "BSE Sensex"}
US_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "NVDA": "NVIDIA",
    "META": "Meta", "TSLA": "Tesla", "NFLX": "Netflix", "AMD": "AMD", "JPM": "JPMorgan Chase",
    "V": "Visa", "WMT": "Walmart", "SPY": "S&P 500 ETF", "^GSPC": "S&P 500",
}

# Letters, digits and the few punctuation marks real tickers use (M&M, BAJAJ-AUTO, ^NSEI, X.NS).
_SYMBOL_RE = re.compile(r"[A-Z0-9&^.\-]{1,20}")
_IN_INDEX_PREFIXES = ("^NSE", "^BSE", "^CNX", "^INDIA")


def resolve_symbol(raw: str, market: str = "auto") -> tuple[str, Market]:
    """Validate a user symbol and return (yfinance_symbol, Market).

    market: "auto" | "in" | "us". Bare names are matched against the Nifty list, so
    "RELIANCE" becomes "RELIANCE.NS". Pass market="us" for US tickers that collide with
    Indian names (e.g. the INFY ADR).
    """
    s = (raw or "").upper().strip()
    if not _SYMBOL_RE.fullmatch(s) or ".." in s:
        raise SymbolError(f"Invalid symbol: {raw!r}")

    hint = (market or "auto").lower()
    if s.endswith((".NS", ".BO")) or s.startswith(_IN_INDEX_PREFIXES):
        mk = IN
    elif hint in ("in", "india"):
        mk = IN
    elif hint == "us":
        mk = US
    else:
        mk = IN if s in NIFTY else US

    if mk is IN and not s.startswith("^") and not s.endswith((".NS", ".BO")):
        s += ".NS"
    return s, mk


def base_symbol(symbol: str) -> str:
    return symbol[:-3] if symbol.endswith((".NS", ".BO")) else symbol


def display_name(symbol: str) -> str:
    base = base_symbol(symbol)
    if symbol in IN_INDICES:
        return IN_INDICES[symbol]
    if symbol.endswith((".NS", ".BO")):
        return NIFTY.get(base, base)
    return US_NAMES.get(base, base)


def exchange_label(symbol: str, mk: Market) -> str:
    if symbol.startswith("^"):
        return "Index"
    if symbol.endswith(".NS"):
        return "NSE"
    if symbol.endswith(".BO"):
        return "BSE"
    return "US"


def is_session_open(mk: Market, last_bar_ts: datetime | None, now: datetime | None = None) -> bool:
    """Open = weekday, inside exchange hours, and the newest bar is fresh (handles holidays)."""
    tz = ZoneInfo(mk.tz)
    now = now.astimezone(tz) if now else datetime.now(tz)
    if now.weekday() >= 5:
        return False
    if not (mk.open_hm <= (now.hour, now.minute) < mk.close_hm):
        return False
    if last_bar_ts is None:
        return False
    return (now - last_bar_ts).total_seconds() <= 20 * 60
