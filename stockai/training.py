"""Honest model training: chronological split, early stopping, walk-forward CV, costed backtest."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from . import data as data_mod
from .features import TECH_FEATURES, WARMUP, build_features, make_target
from .markets import IN, Market, resolve_symbol
from .model import save_model
from .semantic import NEWS_FEATURE_DAYS, build_semantic, usable_semantic
from .signals import backtest

# Shallow, heavily regularised trees: daily returns are mostly noise, so capacity is the enemy.
PARAMS = dict(n_estimators=600, learning_rate=0.02, max_depth=3, min_child_weight=10,
              subsample=0.7, colsample_bytree=0.7, reg_lambda=5.0, gamma=0.1,
              eval_metric="logloss", random_state=42)
TRAIN_FRAC, VAL_FRAC = 0.70, 0.80   # remaining 20% is the untouched holdout
MIN_ROWS = 400


def _auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(set(y)) > 1 else 0.5


def _fit_one(data: pd.DataFrame, cols: list[str], i_tr: int, i_va: int, gap: int = 1):
    """Fit on data[:i_tr], early-stop on the next slice, walk-forward CV on everything before the holdout."""
    X = lambda d: d[cols]
    train, val, test = data.iloc[:i_tr], data.iloc[i_tr + gap:i_va], data.iloc[i_va + gap:]  # purge gaps
    clf = XGBClassifier(**PARAMS, early_stopping_rounds=40)
    clf.fit(X(train), train["target"], eval_set=[(X(val), val["target"])], verbose=False)
    best_n = int(clf.best_iteration) + 1
    fit_part = data.iloc[:i_va]
    cv = []
    for tr_i, te_i in TimeSeriesSplit(n_splits=5, gap=gap).split(fit_part):
        m = XGBClassifier(**{**PARAMS, "n_estimators": best_n})
        m.fit(X(fit_part.iloc[tr_i]), fit_part["target"].iloc[tr_i])
        cv.append(_auc(fit_part["target"].iloc[te_i], m.predict_proba(X(fit_part.iloc[te_i]))[:, 1]))
    return {"clf": clf, "best_n": best_n, "cv": cv,
            "p_val": clf.predict_proba(X(val))[:, 1], "p_test": clf.predict_proba(X(test))[:, 1]}


def fit_model(df: pd.DataFrame, ctx: pd.DataFrame | None, mk: Market, symbol: str,
              sem: pd.DataFrame | None = None, sem_info: dict | None = None):
    """Fit on the oldest 70%, early-stop on the next 10%, report on the final 20%.

    With `sem` (semantic features) three models are fitted on identical rows: technical-only,
    semantic-only and combined. The combined one is served; the other two are the ablation that says
    whether the semantics actually add predictive information.

    Returns (clf, meta). Holdout rows are never used for fitting, early stopping or thresholds.
    """
    feats = build_features(df, ctx, sem)
    data = feats.copy()
    data["target"] = make_target(df)
    data = data.iloc[WARMUP:].dropna(subset=["target"])

    sem_cols: list[str] = []
    if sem is not None and sem_info:
        # Train only where headlines exist, otherwise "no news" would just encode "old data".
        data = data[data.index >= pd.Timestamp(sem_info["news_start"]) + pd.Timedelta(days=NEWS_FEATURE_DAYS)]
        sem_cols = usable_semantic(sem, data.index)
    n = len(data)
    if n < MIN_ROWS:
        raise ValueError(f"{symbol}: only {n} usable rows, need at least {MIN_ROWS}")

    i_tr, i_va = int(n * TRAIN_FRAC), int(n * VAL_FRAC)
    train, val, test = data.iloc[:i_tr], data.iloc[i_tr + 1:i_va], data.iloc[i_va + 1:]
    y_test = test["target"].to_numpy()

    cols = TECH_FEATURES + sem_cols
    main = _fit_one(data, cols, i_tr, i_va)
    clf, best_n, cv, p_val, p_test = main["clf"], main["best_n"], main["cv"], main["p_val"], main["p_test"]
    buy_th = float(max(0.5, np.quantile(p_val, 0.70)))
    sell_th = float(min(0.5, np.quantile(p_val, 0.30)))

    auc = _auc(y_test, p_test)
    pos = int(y_test.sum())
    neg = len(y_test) - pos
    # Std-error of AUC under the null (no skill): is the holdout AUC distinguishable from 0.5?
    se = float(np.sqrt((pos + neg + 1) / (12 * max(pos, 1) * max(neg, 1))))
    z = (auc - 0.5) / se
    acc = float(accuracy_score(y_test, p_test >= 0.5))
    baseline = float(max(y_test.mean(), 1 - y_test.mean()))

    ablation = None
    if sem_cols:
        tech = _fit_one(data, TECH_FEATURES, i_tr, i_va)
        only = _fit_one(data, sem_cols, i_tr, i_va)
        ablation = {
            "technical_only": {"auc": _auc(y_test, tech["p_test"]), "cv_auc": float(np.mean(tech["cv"]))},
            "semantic_only": {"auc": _auc(y_test, only["p_test"]), "cv_auc": float(np.mean(only["cv"]))},
            "combined": {"auc": auc, "cv_auc": float(np.mean(cv))},
        }
        ablation["semantic_adds"] = bool(auc > ablation["technical_only"]["auc"]
                                         and np.mean(cv) > ablation["technical_only"]["cv_auc"])

    close = df["Close"]
    next_ret = (close.shift(-1) / close - 1).reindex(test.index).to_numpy()
    bt = backtest(p_test, next_ret, buy_th, sell_th, mk.cost_bps)

    importances = sorted(zip(cols, clf.feature_importances_), key=lambda t: -t[1])[:8]
    sem_share = float(sum(v for k, v in zip(cols, clf.feature_importances_) if k in sem_cols))
    meta = {
        "schema": 2, "symbol": symbol, "market": mk.code, "interval": "1d",
        "features": cols, "buy_th": buy_th, "sell_th": sell_th, "cooldown": mk.cooldown,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "xgboost": xgboost.__version__, "best_iteration": best_n,
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "in_sample_until": str(val.index[-1].date()),
        "holdout_start": str(test.index[0].date()), "holdout_end": str(test.index[-1].date()),
        "metrics": {"auc": auc, "auc_z": float(z), "accuracy": acc, "baseline_accuracy": baseline,
                    "cv_auc_mean": float(np.mean(cv)), "cv_auc_std": float(np.std(cv)), "cv_auc_folds": cv},
        "backtest": bt,
        # Edge claim needs holdout AUC >~1.64 standard errors above 0.5 *and* positive walk-forward CV.
        "has_edge": bool(z >= 1.64 and np.mean(cv) > 0.5),
        "top_features": {k: float(v) for k, v in importances},
        "semantic": {"enabled": bool(sem_cols), "features": sem_cols, "importance_share": sem_share,
                     "scorer": (sem_info or {}).get("scorer"), "news_start": (sem_info or {}).get("news_start"),
                     "headlines": (sem_info or {}).get("headlines", 0), "ablation": ablation},
    }
    return clf, meta


DIP_HORIZON = 5          # trading days ahead
DIP_QUANTILE = 0.80      # a "dip" is a worse-than-typical drawdown for this stock (~1 in 5 windows)


def dip_target(df: pd.DataFrame, horizon: int = DIP_HORIZON) -> pd.Series:
    """Worst low in the next `horizon` sessions relative to today's close (<= 0). Last rows NaN."""
    worst = df["Low"].rolling(horizon).min().shift(-horizon)
    return (worst / df["Close"] - 1).where(worst.notna()).clip(upper=0)


def fit_dip_model(df: pd.DataFrame, ctx: pd.DataFrame | None, mk: Market, symbol: str,
                  sem: pd.DataFrame | None = None, sem_info: dict | None = None):
    """Probability that the stock falls at least `threshold` below today's close within 5 sessions.

    Drops are far more predictable than direction because volatility clusters and bad news raises risk.
    The threshold is this stock's own 80th-percentile drawdown *measured on the training slice only*.
    Reports a volatility-only baseline so we can see what news/events add beyond plain volatility.
    """
    feats = build_features(df, ctx, sem)
    dd = dip_target(df)
    data = feats.copy()
    data["dd"] = dd
    data = data.iloc[WARMUP:].dropna(subset=["dd"])
    sem_cols: list[str] = []
    if sem is not None and sem_info:
        data = data[data.index >= pd.Timestamp(sem_info["news_start"]) + pd.Timedelta(days=NEWS_FEATURE_DAYS)]
        sem_cols = usable_semantic(sem, data.index)
    n = len(data)
    if n < MIN_ROWS:
        raise ValueError(f"{symbol}: only {n} usable rows, need at least {MIN_ROWS}")
    i_tr, i_va = int(n * TRAIN_FRAC), int(n * VAL_FRAC)
    gap = DIP_HORIZON
    threshold = float(min(-0.01, np.quantile(data["dd"].iloc[:i_tr], 1 - DIP_QUANTILE)))   # negative number
    data["target"] = (data["dd"] <= threshold).astype(float)

    cols = TECH_FEATURES + sem_cols
    main = _fit_one(data, cols, i_tr, i_va, gap)
    test = data.iloc[i_va + gap:]
    y = test["target"].to_numpy()
    p = main["p_test"]
    auc = _auc(y, p)
    vol_auc = _auc(y, test["vol_20"].fillna(test["vol_20"].median()).to_numpy())     # "just look at volatility"
    pos, neg = int(y.sum()), len(y) - int(y.sum())
    se = float(np.sqrt((pos + neg + 1) / (12 * max(pos, 1) * max(neg, 1))))
    top = p >= np.quantile(p, 0.8)
    base_rate = float(y.mean())
    meta = {
        "schema": 2, "kind": "dip", "symbol": symbol, "market": mk.code, "features": cols,
        "horizon_days": DIP_HORIZON, "threshold": threshold,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_iteration": main["best_n"],
        "train_base_rate": float(data["target"].iloc[:i_tr].mean()),
        "holdout": {"start": str(test.index[0].date()), "end": str(test.index[-1].date()), "rows": len(test),
                    "base_rate": base_rate, "auc": auc, "auc_z": float((auc - 0.5) / se),
                    "volatility_only_auc": vol_auc, "cv_auc_mean": float(np.mean(main["cv"])),
                    "dip_rate_when_top_quintile": float(y[top].mean()) if top.any() else None,
                    "lift_top_quintile": float(y[top].mean() / base_rate) if top.any() and base_rate > 0 else None},
        # "Has skill" = clearly better than chance AND positive in walk-forward CV.
        "has_skill": bool((auc - 0.5) / se >= 1.64 and np.mean(main["cv"]) > 0.5),
        "semantic": {"enabled": bool(sem_cols), "features": sem_cols, "scorer": (sem_info or {}).get("scorer"),
                     "news_start": (sem_info or {}).get("news_start")},
    }
    return main["clf"], meta


def train_symbol(raw_symbol: str, market: str = "auto", years: int = 10,
                 news: bool = True, news_years: int = 4, scorer: str | None = None):
    symbol, mk = resolve_symbol(raw_symbol, market)
    df = data_mod.fetch_daily(symbol, years=years, ttl=0)
    if df.empty:
        raise ValueError(f"No price data for {symbol}")
    ctx = data_mod.align_context(data_mod.fetch_context(mk, years=years, ttl=0), df.index)
    sem = sem_info = None
    if news:
        start = max(df.index[0].date(), (datetime.now(timezone.utc) - timedelta(days=365 * news_years)).date())
        sem, sem_info = build_semantic(symbol, mk, df.index, scorer=scorer, start=start, cache_ttl=0)
    clf, meta = fit_model(df, ctx, mk, symbol, sem, sem_info)
    path = save_model(symbol, clf, meta)
    dclf, dmeta = fit_dip_model(df, ctx, mk, symbol, sem, sem_info)
    save_model(symbol + "__dip", dclf, dmeta)
    meta["dip"] = dmeta
    return symbol, meta, path
