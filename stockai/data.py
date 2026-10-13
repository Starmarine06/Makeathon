"""Market data access: yfinance wrapped with a TTL cache and a lock (yfinance is not thread-safe)."""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import yfinance as yf

from .markets import Market

_cache: dict = {}
_cache_lock = threading.Lock()
_yf_lock = threading.Lock()


def cached(key, ttl: float, loader):
    """Return loader() cached for ttl seconds. Empty results are never cached."""
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    value = loader()
    empty = value is None or (hasattr(value, "empty") and value.empty)
    if not empty:
        with _cache_lock:
            _cache[key] = (now, value)
            if len(_cache) > 256:
                for k, _ in sorted(_cache.items(), key=lambda kv: kv[1][0])[:64]:
                    del _cache[k]
    return value


def _download(symbol: str, **kwargs) -> pd.DataFrame:
    with _yf_lock:
        return yf.download(symbol, progress=False, auto_adjust=True, threads=False, **kwargs)


def tidy(df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
    """Flatten yfinance columns, keep OHLCV, make the index a clean DatetimeIndex."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    need = ["Open", "High", "Low", "Close", "Volume"]
    if not set(need).issubset(df.columns):
        return pd.DataFrame()
    df = df[need].dropna(subset=["Close"])
    df["Volume"] = df["Volume"].fillna(0)
    idx = pd.DatetimeIndex(df.index)
    if not intraday:
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        idx = idx.normalize()
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_daily(symbol: str, years: int = 5, ttl: float = 300) -> pd.DataFrame:
    return cached(("daily", symbol, years), ttl,
                  lambda: tidy(_download(symbol, period=f"{years}y", interval="1d")))


def fetch_context(mk: Market, years: int = 5, ttl: float = 900) -> pd.DataFrame:
    """Daily market index + volatility index closes. Missing pieces become NaN columns."""
    def load():
        out = pd.DataFrame()
        for col, sym in (("mkt_close", mk.index_symbol), ("vix_close", mk.vix_symbol)):
            try:
                d = tidy(_download(sym, period=f"{years}y", interval="1d"))
            except Exception:
                d = pd.DataFrame()
            s = d["Close"].rename(col) if not d.empty else pd.Series(dtype=float, name=col)
            out = pd.concat([out, s], axis=1)
        return out if not out.empty else pd.DataFrame(columns=["mkt_close", "vix_close"])
    return cached(("ctx", mk.code, years), ttl, load)


def align_context(ctx: pd.DataFrame | None, index: pd.DatetimeIndex) -> pd.DataFrame | None:
    """Forward-fill context onto the stock's dates (past values only, so no look-ahead)."""
    if ctx is None or ctx.empty:
        return None
    return ctx.reindex(ctx.index.union(index)).ffill().reindex(index)


def fetch_session(symbol: str, mk: Market, ttl: float = 10) -> dict | None:
    """Latest trading session aggregated from 1-minute bars, in exchange-local time."""
    def load():
        raw = tidy(_download(symbol, period="5d", interval="1m"), intraday=True)
        if raw.empty:
            return None
        idx = raw.index
        idx = idx.tz_localize("UTC") if idx.tz is None else idx
        raw.index = idx.tz_convert(mk.tz)
        last_day = raw.index[-1].date()
        day = raw[np.array([d == last_day for d in raw.index.date])]
        opens = day["Open"].dropna()
        return {
            "date": last_day.isoformat(),
            "open": float(opens.iloc[0] if len(opens) else day["Close"].iloc[0]),
            "high": float(day["High"].max()),
            "low": float(day["Low"].min()),
            "close": float(day["Close"].iloc[-1]),
            "volume": int(day["Volume"].sum()),
            "last_ts": day.index[-1].to_pydatetime(),
        }
    return cached(("session", symbol), ttl, load)


def fetch_earnings(symbol: str, ttl: float = 6 * 3600) -> pd.DataFrame:
    """Earnings calendar: index = report date (exchange-local, tz-naive), column `surprise` in percent.

    Includes upcoming scheduled dates (surprise NaN). Empty frame if yfinance has nothing.
    """
    def load():
        try:
            with _yf_lock:
                e = yf.Ticker(symbol).get_earnings_dates(limit=60)
        except Exception:
            return pd.DataFrame()
        if e is None or e.empty:
            return pd.DataFrame()
        idx = pd.DatetimeIndex(e.index)
        idx = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        out = pd.DataFrame({"surprise": pd.to_numeric(e["Surprise(%)"], errors="coerce").to_numpy()}, index=idx)
        return out[~out.index.duplicated(keep="last")].sort_index()
    return cached(("earnings", symbol), ttl, load)


def fetch_analyst_actions(symbol: str, ttl: float = 6 * 3600) -> pd.DataFrame:
    """Analyst rating / price-target changes. Index = date, columns `action`, `target_chg`. US coverage only."""
    def load():
        try:
            with _yf_lock:
                u = yf.Ticker(symbol).upgrades_downgrades
        except Exception:
            return pd.DataFrame()
        if u is None or u.empty or "Action" not in u:
            return pd.DataFrame()
        idx = pd.DatetimeIndex(u.index)
        idx = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        cur = pd.to_numeric(u.get("currentPriceTarget"), errors="coerce").to_numpy()
        pri = pd.to_numeric(u.get("priorPriceTarget"), errors="coerce").to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            chg = np.where((cur > 0) & (pri > 0), cur / pri - 1, np.nan)
        return pd.DataFrame({"action": u["Action"].astype(str).str.lower().to_numpy(),
                             "target_chg": np.clip(chg, -0.5, 0.5)}, index=idx).sort_index()
    return cached(("analyst", symbol), ttl, load)
