"""Feature engineering shared by training and serving.

Every feature is scale-free (ratios, returns, z-like oscillators) so one model can't
memorise a price level, and every feature is causal: row t only uses data up to close t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WARMUP = 60  # leading rows whose rolling windows are incomplete

TECH_FEATURES = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20",
    "vol_10", "vol_20", "vol_regime",
    "dist_sma10", "dist_sma20", "dist_sma50", "dist_ema20", "sma20_slope",
    "bb_pos", "bb_width",
    "rsi14", "stoch_k14",
    "macd_pct", "macd_hist_pct",
    "atr14_pct",
    "vol_rel20",
    "gap", "intraday_range", "body", "close_loc",
    "dist_high_252", "dist_low_252",
    "dow",
    # market context (NaN when the index / VIX could not be fetched; XGBoost handles NaN)
    "mkt_ret_1", "mkt_ret_5", "rel_ret_5", "vix_rel", "vix_chg_1",
]

# Semantic features: what is being *said* and *scheduled* about the company, not just how price moved.
# Built in stockai/semantic.py; every one only uses information published before the close of row t.
NEWS_FEATURES = ["sent_1d", "sent_3d", "sent_7d", "sent_shock", "neg_share_7d"]
EVENT_FEATURES = ["days_to_earn", "days_since_earn", "last_surprise"]
ANALYST_FEATURES = ["analyst_net_30d", "target_chg_30d"]
SEMANTIC_FEATURES = NEWS_FEATURES + EVENT_FEATURES + ANALYST_FEATURES
FEATURES = TECH_FEATURES + SEMANTIC_FEATURES


def build_features(df: pd.DataFrame, ctx: pd.DataFrame | None = None,
                   sem: pd.DataFrame | None = None) -> pd.DataFrame:
    """df: daily OHLCV. ctx: optional frame (same index) with mkt_close / vix_close.
    sem: optional frame (same index) with SEMANTIC_FEATURES columns; missing -> NaN."""
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    c = c.where(c > 0)
    f = pd.DataFrame(index=df.index)

    r = c.pct_change()
    for n in (1, 2, 3, 5, 10, 20):
        f[f"ret_{n}"] = c.pct_change(n)

    f["vol_10"] = r.rolling(10).std()
    f["vol_20"] = r.rolling(20).std()
    f["vol_regime"] = f["vol_10"] / r.rolling(60).std()

    sma = {n: c.rolling(n).mean() for n in (10, 20, 50)}
    for n, s in sma.items():
        f[f"dist_sma{n}"] = c / s - 1
    f["dist_ema20"] = c / c.ewm(span=20, adjust=False).mean() - 1
    f["sma20_slope"] = sma[20].pct_change(5)

    bb_std = c.rolling(20).std()
    band = 4 * bb_std
    f["bb_pos"] = (c - (sma[20] - 2 * bb_std)) / band.where(band > 0)
    f["bb_width"] = band / sma[20]

    delta = c.diff()
    avg_up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["rsi14"] = 100 - 100 / (1 + avg_up / avg_dn)

    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    f["stoch_k14"] = 100 * (c - lo14) / (hi14 - lo14).where(hi14 > lo14)

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_pct"] = macd / c
    f["macd_hist_pct"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    f["atr14_pct"] = tr.rolling(14).mean() / c

    vr = v / v.rolling(20).mean().where(lambda s: s > 0)
    f["vol_rel20"] = np.log(vr.where(vr > 0))

    f["gap"] = o / prev_c - 1
    f["intraday_range"] = (h - l) / o.where(o > 0)
    f["body"] = (c - o) / o.where(o > 0)
    f["close_loc"] = (c - l) / (h - l).where(h > l)

    f["dist_high_252"] = c / c.rolling(252, min_periods=WARMUP).max() - 1
    f["dist_low_252"] = c / c.rolling(252, min_periods=WARMUP).min() - 1
    f["dow"] = df.index.dayofweek.astype(float)

    mkt = ctx["mkt_close"] if ctx is not None and "mkt_close" in ctx else pd.Series(np.nan, index=df.index)
    vix = ctx["vix_close"] if ctx is not None and "vix_close" in ctx else pd.Series(np.nan, index=df.index)
    f["mkt_ret_1"] = mkt.pct_change(fill_method=None)
    f["mkt_ret_5"] = mkt.pct_change(5, fill_method=None)
    f["rel_ret_5"] = f["ret_5"] - f["mkt_ret_5"]
    f["vix_rel"] = vix / vix.rolling(252, min_periods=WARMUP).mean() - 1
    f["vix_chg_1"] = vix.pct_change(fill_method=None)

    for col in SEMANTIC_FEATURES:
        f[col] = sem[col].reindex(df.index) if sem is not None and col in sem else np.nan

    return f[FEATURES].replace([np.inf, -np.inf], np.nan)


def make_target(df: pd.DataFrame) -> pd.Series:
    """1 if the next close is higher than today's close. Last row is NaN (unknown)."""
    nxt = df["Close"].shift(-1)
    return (nxt > df["Close"]).astype(float).where(nxt.notna())
