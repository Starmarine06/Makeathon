"""Cross-sectional stock ranking: the way systematic funds actually trade equities.

Instead of guessing one stock's next-day direction (a coin flip), rank a whole universe on the
well-documented return drivers (12-1 momentum, short-term reversal, volatility, earnings surprise,
optionally news tone), hold the top K, rebalance every H days, and judge the result only on
walk-forward, out-of-sample, after-cost returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from . import data as data_mod
from .features import ANALYST_FEATURES, EVENT_FEATURES, NEWS_FEATURES, WARMUP, build_features
from .markets import Market

H = 5                     # holding period / rebalance interval in trading days
MIN_NAMES = 15            # skip dates with fewer ranked stocks than this
MIN_TRAIN_DAYS = 750      # ~3 years before the first out-of-sample prediction
TEST_BLOCK = 126          # retrain every ~6 months

PRICE_FEATURES = ["mom_12_1", "ret_126", "ret_63", "ret_20", "ret_10", "ret_5", "ret_1", "vol_20", "vol_60",
                  "vol_regime", "dist_high_252", "dist_low_252", "dist_sma50", "rsi14", "vol_rel20",
                  "atr14_pct", "rel_ret_5", "bb_pos", "idio_vol"]
PARAMS = dict(n_estimators=250, learning_rate=0.03, max_depth=3, min_child_weight=50, subsample=0.7,
              colsample_bytree=0.7, reg_lambda=10.0, random_state=7, n_jobs=4)


def stock_frame(symbol: str, df: pd.DataFrame, ctx, sem=None) -> pd.DataFrame:
    f = build_features(df, ctx, sem)
    c, r = df["Close"], df["Close"].pct_change()
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1          # the classic Jegadeesh-Titman signal
    f["ret_63"], f["ret_126"] = c.pct_change(63), c.pct_change(126)
    f["vol_60"] = r.rolling(60).std()
    mkt = ctx["mkt_close"].pct_change(fill_method=None) if ctx is not None and "mkt_close" in ctx else None
    f["idio_vol"] = (r - mkt).rolling(60).std() if mkt is not None else np.nan
    f["fwd"] = c.shift(-H) / c - 1
    f["symbol"] = symbol
    f = f.iloc[WARMUP:]
    return f.set_index("symbol", append=True)                # index: (date, symbol)


def load_panel(symbols: list[str], mk: Market, years: int, news: bool = False, scorer=None,
               log=print) -> tuple[pd.DataFrame, list[str]]:
    """Stack every symbol into one (date, symbol) panel. Returns (panel, feature_names)."""
    from . import semantic
    ctx_raw = data_mod.fetch_context(mk, years=years, ttl=0)
    frames, skipped = [], []
    for sym in symbols:
        df = data_mod.fetch_daily(sym, years=years, ttl=0)
        if len(df) < MIN_TRAIN_DAYS // 2:
            skipped.append(sym)
            continue
        ctx = data_mod.align_context(ctx_raw, df.index)
        extra = None
        try:
            if news:
                last = df.index[-1].date()
                extra, _ = semantic.build_semantic(sym, mk, df.index, scorer=scorer, cache_ttl=0,
                                                   start=last.replace(year=last.year - 4))
            else:   # earnings are cheap; headlines are slow (opt in with news=True)
                earn = data_mod.fetch_earnings(sym, ttl=0)
                extra = pd.concat([pd.DataFrame(np.nan, index=df.index, columns=NEWS_FEATURES),
                                   semantic.event_features(df.index, earn),
                                   pd.DataFrame(np.nan, index=df.index, columns=ANALYST_FEATURES)], axis=1)
        except Exception as exc:
            log(f"  {sym}: semantic features failed ({exc}); price features only")
        frames.append(stock_frame(sym, df, ctx, extra))
    if skipped:
        log(f"  skipped (no data): {', '.join(skipped)}")
    panel = pd.concat(frames).sort_index()
    sem_cols = [c for c in NEWS_FEATURES + EVENT_FEATURES + ANALYST_FEATURES if panel[c].notna().mean() > 0.2]
    return panel, PRICE_FEATURES + sem_cols


def ranked(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cross-sectional percentile rank per date, centred on 0. Missing -> 0 (neutral).

    Ranking removes the market level, so the model can only learn *relative* attractiveness."""
    x = panel[cols].groupby(level=0).rank(pct=True) - 0.5
    return x.fillna(0.0)


def walk_forward_scores(panel: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Out-of-sample score for every (date, symbol) after MIN_TRAIN_DAYS. Each block is predicted by a
    model trained only on dates at least H days earlier (no overlap of forward-return windows)."""
    X = ranked(panel, cols)
    y = panel["fwd"].groupby(level=0).rank(pct=True) - 0.5
    all_dates = panel.index.get_level_values(0)
    dates = all_dates.unique().sort_values()
    out = []
    for start in range(MIN_TRAIN_DAYS, len(dates), TEST_BLOCK):
        test_dates = dates[start:start + TEST_BLOCK]
        train_dates = dates[:start - H]
        tr = all_dates.isin(train_dates) & y.notna().to_numpy()
        te = all_dates.isin(test_dates)
        m = XGBRegressor(**PARAMS).fit(X[tr], y[tr])
        out.append(pd.Series(m.predict(X[te]), index=panel.index[te]))
    return pd.concat(out)


def _stats(net: pd.Series) -> dict:
    ppy = 252 / H
    eq = (1 + net).cumprod()
    sd = net.std()
    years = len(net) / ppy
    return {"ann_return": float(eq.iloc[-1] ** (1 / years) - 1), "ann_vol": float(sd * np.sqrt(ppy)),
            "sharpe": float(net.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0,
            "max_drawdown": float((eq / eq.cummax() - 1).min()), "total_return": float(eq.iloc[-1] - 1)}


def backtest_topk(scores: pd.Series, fwd: pd.Series, k: int, cost_bps: float,
                  vol_target: float | None = None) -> dict:
    """Hold the top-k by score, rebalance every H days (non-overlapping), pay cost on turnover.

    vol_target: scale exposure by target / trailing realised vol of the strategy (capped at 1x, rest in cash)."""
    dates = scores.index.get_level_values(0).unique().sort_values()[::H]
    prev: set = set()
    gross, net, bench, turn = [], [], [], []
    for d in dates:
        s = scores.xs(d, level=0).dropna()
        f = fwd.xs(d, level=0).reindex(s.index)
        if len(s) < MIN_NAMES or f.isna().all():
            continue
        picks = set(s.nlargest(k).index)
        t = 1.0 if not prev else len(picks - prev) / k
        g = float(f[list(picks)].mean())
        gross.append(g)
        net.append(g - 2 * t * cost_bps / 1e4)
        bench.append(float(f.mean()))
        turn.append(t)
        prev = picks
    net_s, bench_s = pd.Series(net), pd.Series(bench)
    diff = net_s - bench_s
    res = {"periods": len(net), "net": _stats(net_s), "gross_sharpe": _stats(pd.Series(gross))["sharpe"],
           "benchmark": _stats(bench_s), "turnover_per_rebalance": float(np.mean(turn)),
           "hit_rate_vs_benchmark": float((net_s > bench_s).mean()),
           "excess_per_period_t": float(diff.mean() / (diff.std() / np.sqrt(len(diff))))}
    if vol_target:
        realised = net_s.rolling(12, min_periods=6).std().shift(1) * np.sqrt(252 / H)
        scale = (vol_target / realised).clip(upper=1.0).fillna(1.0)
        res["vol_targeted"] = _stats(net_s * scale)
        res["vol_targeted"]["avg_exposure"] = float(scale.mean())
    return res


def information_coefficient(scores: pd.Series, fwd: pd.Series) -> dict:
    """Per-date rank correlation between score and the realised forward return (sampled every H days)."""
    dates = scores.index.get_level_values(0).unique().sort_values()[::H]
    ics = []
    for d in dates:
        a = scores.xs(d, level=0)
        b = fwd.xs(d, level=0).reindex(a.index)
        ok = a.notna() & b.notna()
        if ok.sum() >= MIN_NAMES:
            ics.append(a[ok].rank().corr(b[ok].rank()))
    ics = pd.Series(ics).dropna()
    return {"mean_ic": float(ics.mean()), "ic_t": float(ics.mean() / (ics.std() / np.sqrt(len(ics)))),
            "pct_positive": float((ics > 0).mean()), "n": int(len(ics))}


def evaluate(panel: pd.DataFrame, cols: list[str], mk: Market, k: int, vol_target: float | None = 0.15) -> dict:
    scores = walk_forward_scores(panel, cols)
    fwd = panel["fwd"].reindex(scores.index)
    mom = panel["mom_12_1"].reindex(scores.index)
    return {
        "model": {**backtest_topk(scores, fwd, k, mk.cost_bps, vol_target), "ic": information_coefficient(scores, fwd)},
        "momentum_baseline": {**backtest_topk(mom, fwd, k, mk.cost_bps, vol_target), "ic": information_coefficient(mom, fwd)},
        "oos_start": str(scores.index.get_level_values(0).min().date()),
        "oos_end": str(scores.index.get_level_values(0).max().date()),
        "names": int(panel.index.get_level_values(1).nunique()), "features": cols, "k": k, "horizon": H,
        "cost_bps_per_side": mk.cost_bps,
    }
