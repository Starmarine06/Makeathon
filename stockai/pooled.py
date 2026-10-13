"""Pooled dip-risk model: one model trained across ~50 stocks of a market.

A single stock has only a few hundred out-of-sample days, too few to tell skill from luck. Pooling gives
~20x the data. A "dip" is defined relative to the stock's own volatility (worst low in the next 5 sessions
at least K x 5-day sigma below today's close), so the model can't score by merely noticing that volatile
stocks fall further: it has to find risk *beyond* normal volatility (drawdown position, momentum,
earnings timing, news tone when available).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from . import xsection as xs
from .markets import Market
from .model import save_model

K = 1.5                    # dip = drawdown worse than K x (daily sigma x sqrt(horizon))
HORIZON = 5
PARAMS = dict(n_estimators=500, learning_rate=0.03, max_depth=3, min_child_weight=100, subsample=0.7,
              colsample_bytree=0.7, reg_lambda=10.0, eval_metric="logloss", random_state=11, n_jobs=4)
BLOCK = 20                 # trading days per bootstrap block (returns and vol are autocorrelated)


def dip_threshold(sigma_daily: pd.Series | float):
    """Drop (as a negative fraction) that counts as a dip for a stock with this daily volatility."""
    return -K * sigma_daily * np.sqrt(HORIZON)


def labels(panel: pd.DataFrame) -> pd.Series:
    ok = panel["dd"].notna() & panel["vol_60"].notna() & (panel["vol_60"] > 0)
    return (panel["dd"] <= dip_threshold(panel["vol_60"])).astype(float).where(ok)


def _auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(set(y)) > 1 else 0.5


def block_bootstrap_auc(dates, y, p, n=300, seed=0) -> tuple[float, float]:
    """5th-95th percentile AUC, resampling blocks of consecutive dates (keeps the time dependence)."""
    codes = pd.factorize(pd.Series(dates))[0] // BLOCK
    groups = [np.flatnonzero(codes == b) for b in np.unique(codes)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        out.append(_auc(y[idx], p[idx]))
    return float(np.percentile(out, 5)), float(np.percentile(out, 95))


def fit_pooled(panel: pd.DataFrame, cols: list[str], mk: Market):
    y_all = labels(panel)
    keep = y_all.notna().to_numpy()
    data = panel.loc[keep, cols + ["vol_regime"]].copy() if "vol_regime" not in cols else panel.loc[keep, cols].copy()
    y = y_all[keep].to_numpy()
    d = data.index.get_level_values(0)
    dates = d.unique().sort_values()
    i_tr, i_va = int(len(dates) * 0.70), int(len(dates) * 0.80)
    tr = d.isin(dates[:i_tr])
    va = d.isin(dates[i_tr + HORIZON:i_va])
    te = d.isin(dates[i_va + HORIZON:])
    clf = XGBClassifier(**PARAMS, early_stopping_rounds=40)
    clf.fit(data.loc[tr, cols], y[tr], eval_set=[(data.loc[va, cols], y[va])], verbose=False)
    best_n = int(clf.best_iteration) + 1
    p = clf.predict_proba(data.loc[te, cols])[:, 1]
    yt = y[te]
    auc = _auc(yt, p)
    lo, hi = block_bootstrap_auc(d[te], yt, p)
    top = p >= np.quantile(p, 0.8)
    base = float(yt.mean())
    meta = {
        "schema": 2, "kind": "dip_pooled", "market": mk.code, "features": cols, "k_sigma": K,
        "horizon_days": HORIZON, "best_iteration": best_n,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "xgboost": xgboost.__version__,
        "names": int(panel.index.get_level_values(1).nunique()),
        "train_base_rate": float(y[tr].mean()),
        "holdout": {"start": str(dates[i_va + HORIZON].date()), "end": str(dates[-1].date()), "rows": int(te.sum()),
                    "base_rate": base, "auc": auc, "auc_ci90": [lo, hi],
                    "volatility_regime_only_auc": _auc(yt, data.loc[te, "vol_regime"].fillna(1.0).to_numpy()),
                    "dip_rate_when_top_quintile": float(yt[top].mean()),
                    "lift_top_quintile": float(yt[top].mean() / base) if base > 0 else None,
                    "dip_rate_when_bottom_quintile": float(yt[p <= np.quantile(p, 0.2)].mean())},
        # Skill = the whole 90% block-bootstrap interval of the holdout AUC sits above 0.5.
        "has_skill": bool(lo > 0.5),
        "semantic": {"features": [c for c in cols if c in xs.NEWS_FEATURES + xs.EVENT_FEATURES + xs.ANALYST_FEATURES]},
    }
    imp = sorted(zip(cols, clf.feature_importances_), key=lambda t: -t[1])[:8]
    meta["top_features"] = {k: float(v) for k, v in imp}
    return clf, meta


def train_pooled(symbols: list[str], mk: Market, years: int = 10, news: bool = False, log=print):
    panel, cols = xs.load_panel(symbols, mk, years, news=news, log=log)
    clf, meta = fit_pooled(panel, cols, mk)
    path = save_model(f"POOLED_{mk.code}__dip", clf, meta)
    return meta, path
