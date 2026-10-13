"""Semantic layer: news sentiment, earnings events and analyst actions as model features.

Price-only models can only pattern-match. This module gives the model *reasons*: what the press is
saying about the company (FinBERT-scored headlines), what is scheduled (earnings) and what happened
last time (earnings surprise), and what professionals changed their minds about (analyst actions).

Causality rule: a feature on trading day t only uses information published before the close of t.
A headline published after the close (or on a weekend/holiday) is credited to the next trading day.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests

from . import data as data_mod
from .features import ANALYST_FEATURES, EVENT_FEATURES, NEWS_FEATURES, SEMANTIC_FEATURES
from .markets import Market, base_symbol, display_name

log = logging.getLogger("stockai.semantic")

CACHE_DIR = Path(os.environ.get("STOCKAI_CACHE", Path(__file__).resolve().parent.parent / "cache")) / "news"
NEWS_FEATURE_DAYS = 30        # longest news look-back any feature uses
MIN_COVERAGE = 0.20           # a semantic feature present on fewer rows than this is dropped from training
_UA = {"User-Agent": "Mozilla/5.0 (compatible; stockai/1.0)"}
_REGION = {"US": ("en-US", "US", "US:en"), "IN": ("en-IN", "IN", "IN:en")}
_io_lock = threading.Lock()


# --------------------------------------------------------------------------- sentiment scoring

class Scorer:
    name = "base"

    def score(self, texts: list[str]) -> list[float]:  # -1 (bearish) .. +1 (bullish)
        raise NotImplementedError


class VaderScorer(Scorer):
    name = "vader"

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._an = SentimentIntensityAnalyzer()

    def score(self, texts):
        return [float(self._an.polarity_scores(t)["compound"]) for t in texts]


class FinbertScorer(Scorer):
    """ProsusAI/finbert: trained on financial text, so 'beats estimates' is positive and 'probe' negative."""
    name = "finbert"

    def __init__(self):
        from transformers import pipeline
        self._pipe = pipeline("text-classification", model="ProsusAI/finbert", top_k=None,
                              truncation=True, max_length=64)

    def score(self, texts):
        out = []
        for i in range(0, len(texts), 64):
            for res in self._pipe(texts[i:i + 64], batch_size=32):
                p = {r["label"].lower(): r["score"] for r in res}
                out.append(float(p.get("positive", 0.0) - p.get("negative", 0.0)))
        return out


_scorers: dict[str, Scorer] = {}


def get_scorer(name: str | None = None) -> Scorer:
    """name: 'finbert' | 'vader' | 'auto' (FinBERT if installed, else VADER). Env STOCKAI_SCORER overrides auto."""
    name = (name or os.environ.get("STOCKAI_SCORER") or "auto").lower()
    with _io_lock:
        if name in _scorers:
            return _scorers[name]
        if name == "auto":
            try:
                sc = FinbertScorer()
            except Exception as exc:
                log.warning("FinBERT unavailable (%s); falling back to VADER", exc)
                sc = VaderScorer()
        elif name == "finbert":
            sc = FinbertScorer()
        elif name == "vader":
            sc = VaderScorer()
        else:
            raise ValueError(f"unknown scorer {name!r}")
        _scorers[name] = _scorers[sc.name] = sc
        return sc


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def score_headlines(titles: list[str], scorer: Scorer) -> list[float]:
    """Score with a persistent per-scorer cache so each distinct headline is only ever scored once."""
    path = CACHE_DIR / f"scores_{scorer.name}.json"
    with _io_lock:
        cache = json.loads(path.read_text()) if path.exists() else {}
    todo = sorted({t for t in titles if _key(t) not in cache})
    if todo:
        for t, s in zip(todo, scorer.score(todo)):
            cache[_key(t)] = round(s, 4)
        with _io_lock:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache))
    return [cache[_key(t)] for t in titles]


# --------------------------------------------------------------------------- headline retrieval

def news_query(symbol: str, mk: Market) -> str:
    """Google News search terms for a company. Bare tickers are ambiguous, so use the company name."""
    name = display_name(symbol)
    if symbol.startswith("^"):
        return f'"{name}" (market OR index OR stocks)'
    extra = "(share price OR stock OR NSE)" if mk.code == "IN" else "(stock OR shares OR earnings)"
    return f'"{name}" {extra}'


def _fetch_window(query: str, start: date, end: date, mk: Market) -> list[dict] | None:
    """Headlines published in [start, end). None means the request failed (retry later), [] means none."""
    import feedparser
    hl, gl, ceid = _REGION[mk.code]
    url = (f"https://news.google.com/rss/search?q={quote_plus(f'{query} after:{start} before:{end}')}"
           f"&hl={hl}&gl={gl}&ceid={ceid}")
    for attempt in range(3):
        try:
            r = requests.get(url, headers=_UA, timeout=15)
        except requests.RequestException:
            time.sleep(1 + attempt)
            continue
        if r.status_code == 200:
            break
        if r.status_code in (429, 503):
            time.sleep(3 * (attempt + 1))
            continue
        return None
    else:
        return None
    items = []
    for e in feedparser.parse(r.content).entries:
        try:
            ts = parsedate_to_datetime(e.published).astimezone(timezone.utc)
        except Exception:
            continue
        src = (e.get("source") or {}).get("title", "")
        title = re.sub(r"\s+", " ", e.get("title", "")).strip()
        if src and title.endswith(f" - {src}"):
            title = title[: -len(src) - 3]
        if title:
            items.append({"ts": ts.isoformat(), "title": title, "source": src, "link": e.get("link", "")})
    return items


def _store_path(symbol: str) -> Path:
    return CACHE_DIR / (re.sub(r"[^A-Za-z0-9_-]", "_", symbol) + ".json")


def _weeks(start: date, end: date) -> list[date]:
    d = start - timedelta(days=start.weekday())
    out = []
    while d <= end:
        out.append(d)
        d += timedelta(days=7)
    return out


def load_news(symbol: str, mk: Market, start: date, end: date, scorer: Scorer,
              fetch: bool = True, max_new: int | None = None, workers: int = 4) -> pd.DataFrame:
    """Scored headlines in [start, end] with columns ts (UTC), title, source, link, score.

    Weekly windows are fetched once and cached on disk. Recent windows (still filling up) are always
    re-fetched. `max_new` caps how many uncached windows one call may fetch (newest first), which keeps
    web requests fast; training passes None to fetch everything.
    """
    path = _store_path(symbol)
    with _io_lock:
        store = json.loads(path.read_text()) if path.exists() else {"windows": [], "items": []}
    done = set(store["windows"])
    today = datetime.now(timezone.utc).date()

    if fetch:
        todo = [w for w in _weeks(start, end)
                if w.isoformat() not in done or w + timedelta(days=9) > today]
        todo.sort(reverse=True)
        if max_new is not None:
            todo = todo[:max_new]
        query = news_query(symbol, mk)

        def job(w):
            time.sleep(0.15)
            return w, _fetch_window(query, w, w + timedelta(days=7), mk)

        if todo:
            known = {(i["title"], i["ts"]) for i in store["items"]}
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for w, items in ex.map(job, todo):
                    if items is None:
                        continue
                    done.add(w.isoformat())
                    for it in items:
                        if (it["title"], it["ts"]) not in known:
                            known.add((it["title"], it["ts"]))
                            store["items"].append(it)
            store["windows"] = sorted(done)
            with _io_lock:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(store))

    df = pd.DataFrame(store["items"], columns=["ts", "title", "source", "link"])
    if df.empty:
        return df.assign(score=pd.Series(dtype=float))
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    df = df[(df["ts"] >= lo) & (df["ts"] < hi)].sort_values("ts").reset_index(drop=True)
    df["score"] = score_headlines(df["title"].tolist(), scorer) if len(df) else pd.Series(dtype=float)
    return df


# --------------------------------------------------------------------------- feature construction

def trading_day_position(index: pd.DatetimeIndex, ts_utc: pd.Series, mk: Market) -> np.ndarray:
    """Row of the trading day each timestamp is credited to (len(index) = after the last bar).

    Published at/before the exchange close -> that day (or the next trading day if it was a
    weekend/holiday). Published after the close -> the next trading day.
    """
    local = ts_utc.dt.tz_convert(mk.tz)
    minutes = local.dt.hour * 60 + local.dt.minute
    after_close = minutes > mk.close_hm[0] * 60 + mk.close_hm[1]
    eff = local.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(after_close.astype(int), unit="D")
    return index.searchsorted(eff.to_numpy(), side="left")


def news_features(index: pd.DatetimeIndex, news: pd.DataFrame, mk: Market) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=index, columns=NEWS_FEATURES)
    if news is None or news.empty:
        return out
    pos = trading_day_position(index, news["ts"], mk)
    ok = pos < len(index)
    pos, sc = pos[ok], news["score"].to_numpy()[ok]
    n = len(index)
    cnt = pd.Series(np.bincount(pos, minlength=n), index=index, dtype=float)
    tot = pd.Series(np.bincount(pos, weights=sc, minlength=n), index=index)
    neg = pd.Series(np.bincount(pos, weights=(sc < -0.3).astype(float), minlength=n), index=index)

    def mean_over(k):
        c = cnt.rolling(k, min_periods=1).sum()
        return tot.rolling(k, min_periods=1).sum() / c.where(c > 0)

    out["sent_1d"] = tot / cnt.where(cnt > 0)
    out["sent_3d"] = mean_over(3)
    out["sent_7d"] = mean_over(7)
    out["sent_shock"] = out["sent_3d"] - mean_over(NEWS_FEATURE_DAYS)   # is today's tone unusual for this name?
    c7 = cnt.rolling(7, min_periods=1).sum()
    out["neg_share_7d"] = neg.rolling(7, min_periods=1).sum() / c7.where(c7 > 0)
    return out


def event_features(index: pd.DatetimeIndex, earnings: pd.DataFrame) -> pd.DataFrame:
    """days_to_earn: calendar days to the next report (0 = reports today). days_since_earn / last_surprise
    describe the last report strictly *before* today, so an after-close report is not leaked into its own day."""
    out = pd.DataFrame(np.nan, index=index, columns=EVENT_FEATURES)
    if earnings is None or earnings.empty:
        return out
    e_dates = earnings.index.to_numpy(dtype="datetime64[D]")
    surprise = earnings["surprise"].to_numpy(dtype=float)
    d = index.to_numpy(dtype="datetime64[D]")
    nxt = np.searchsorted(e_dates, d, side="left")        # first report on/after d
    to = np.full(len(d), 90.0)
    has_next = nxt < len(e_dates)
    to[has_next] = np.minimum((e_dates[nxt[has_next]] - d[has_next]).astype(int), 90)
    prev = nxt - 1                                        # last report strictly before d
    has_prev = prev >= 0
    since = np.full(len(d), np.nan)
    last = np.full(len(d), np.nan)
    since[has_prev] = np.minimum((d[has_prev] - e_dates[prev[has_prev]]).astype(int), 120)
    last[has_prev] = np.clip(surprise[prev[has_prev]], -50, 50)
    out["days_to_earn"], out["days_since_earn"], out["last_surprise"] = to, since, last
    return out


def analyst_features(index: pd.DatetimeIndex, actions: pd.DataFrame) -> pd.DataFrame:
    """Net upgrades and mean price-target revision over the previous 30 calendar days (excluding today)."""
    out = pd.DataFrame(np.nan, index=index, columns=ANALYST_FEATURES)
    if actions is None or actions.empty:
        return out
    cal = pd.date_range(min(actions.index.min(), index.min()), max(actions.index.max(), index.max()))
    up = (actions["action"] == "up").groupby(level=0).sum().reindex(cal, fill_value=0)
    down = (actions["action"] == "down").groupby(level=0).sum().reindex(cal, fill_value=0)
    tc = actions["target_chg"]
    tc_sum = tc.fillna(0).groupby(level=0).sum().reindex(cal, fill_value=0)
    tc_n = tc.notna().groupby(level=0).sum().reindex(cal, fill_value=0)
    net = (up - down).rolling(30).sum().shift(1)
    n30 = tc_n.rolling(30).sum().shift(1)
    chg = tc_sum.rolling(30).sum().shift(1) / n30.where(n30 > 0)
    out["analyst_net_30d"] = net.reindex(index).to_numpy()
    out["target_chg_30d"] = chg.reindex(index).to_numpy()
    first = actions.index.min()
    out.loc[index < first + pd.Timedelta(days=30)] = np.nan     # before coverage began: unknown, not zero
    return out


def build_semantic(symbol: str, mk: Market, index: pd.DatetimeIndex, scorer: Scorer | str | None = None,
                   start: date | None = None, fetch: bool = True, max_new: int | None = None,
                   cache_ttl: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Semantic feature frame aligned to `index`, plus info (scorer, coverage, headlines)."""
    scorer = scorer if isinstance(scorer, Scorer) else get_scorer(scorer)
    start = start or (index[0].date() - timedelta(days=NEWS_FEATURE_DAYS))
    end = datetime.now(timezone.utc).date()
    news = load_news(symbol, mk, start - timedelta(days=NEWS_FEATURE_DAYS), end, scorer, fetch, max_new)
    ttl = {} if cache_ttl is None else {"ttl": cache_ttl}
    earn = data_mod.fetch_earnings(symbol, **ttl) if not symbol.startswith("^") else pd.DataFrame()
    acts = data_mod.fetch_analyst_actions(symbol, **ttl) if not symbol.startswith("^") else pd.DataFrame()

    sem = pd.concat([news_features(index, news, mk), event_features(index, earn),
                     analyst_features(index, acts)], axis=1)[SEMANTIC_FEATURES]
    sem.loc[index < pd.Timestamp(start)] = np.nan
    info = {"scorer": scorer.name, "news_start": start.isoformat(), "headlines": len(news),
            "coverage": {c: float(sem[c].notna().mean()) for c in SEMANTIC_FEATURES},
            "earnings": earn, "news": news}
    return sem, info


def usable_semantic(sem: pd.DataFrame, rows: pd.DatetimeIndex) -> list[str]:
    """Semantic columns with enough real data on `rows` to be worth training on."""
    return [c for c in SEMANTIC_FEATURES if sem.loc[rows, c].notna().mean() >= MIN_COVERAGE]
