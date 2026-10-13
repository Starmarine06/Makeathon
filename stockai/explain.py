"""Explain a prediction: which inputs pushed P(up) and by how much, in plain language.

Uses XGBoost's exact tree-SHAP contributions (log-odds). They are converted to approximate
percentage points of probability via the local slope p*(1-p), so the drivers roughly add up to
the distance between the model's base rate and its prediction.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import xgboost

from .model import LoadedModel

GROUPS = {"news": "News sentiment", "events": "Earnings", "analysts": "Analyst actions",
          "market": "Market backdrop", "technical": "Price & volume pattern"}


def _pct(v): return f"{v * 100:+.1f}%"
def _num(v): return f"{v:+.2f}"
def _days(v): return f"{v:.0f} days"
def _plain(v): return f"{v:.2f}"


# feature -> (label, group, formatter)
SPEC = {
    "sent_1d": ("Today's headline tone", "news", _num),
    "sent_3d": ("Headline tone, last 3 days", "news", _num),
    "sent_7d": ("Headline tone, last 7 days", "news", _num),
    "sent_shock": ("Tone vs. 30-day norm", "news", _num),
    "neg_share_7d": ("Share of negative headlines (7d)", "news", lambda v: f"{v * 100:.0f}%"),
    "days_to_earn": ("Days until earnings", "events", _days),
    "days_since_earn": ("Days since last earnings", "events", _days),
    "last_surprise": ("Last earnings surprise", "events", lambda v: f"{v:+.1f}%"),
    "analyst_net_30d": ("Net analyst upgrades (30d)", "analysts", lambda v: f"{v:+.0f}"),
    "target_chg_30d": ("Avg. price-target revision (30d)", "analysts", _pct),
    "mkt_ret_1": ("Market return today", "market", _pct),
    "mkt_ret_5": ("Market return, 5 days", "market", _pct),
    "rel_ret_5": ("Stock vs. market, 5 days", "market", _pct),
    "vix_rel": ("Volatility index vs. 1-yr avg", "market", _pct),
    "vix_chg_1": ("Volatility index change today", "market", _pct),
    "rsi14": ("RSI (14)", "technical", lambda v: f"{v:.0f}"),
    "stoch_k14": ("Stochastic %K (14)", "technical", lambda v: f"{v:.0f}"),
    "bb_pos": ("Position in Bollinger band", "technical", _plain),
    "vol_regime": ("Short vs. long volatility", "technical", _plain),
    "vol_rel20": ("Volume vs. 20-day avg (log)", "technical", _num),
    "dow": ("Day of week", "technical", lambda v: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(v) % 7]),
}
SPEC.update({
    "mom_12_1": ("12-month momentum (skip last month)", "technical", _pct),
    "ret_63": ("3-month return", "technical", _pct), "ret_126": ("6-month return", "technical", _pct),
    "vol_10": ("10-day volatility", "technical", _pct), "vol_20": ("20-day volatility", "technical", _pct),
    "vol_60": ("60-day volatility", "technical", _pct), "idio_vol": ("Stock-specific volatility", "technical", _pct),
    "atr14_pct": ("Average daily range", "technical", _pct), "bb_width": ("Bollinger band width", "technical", _pct),
    "dist_high_252": ("Distance below 1-year high", "technical", _pct),
    "dist_low_252": ("Distance above 1-year low", "technical", _pct),
})
for n in (1, 2, 3, 5, 10, 20):
    SPEC[f"ret_{n}"] = (f"{n}-day return", "technical", _pct)
for n in (10, 20, 50):
    SPEC[f"dist_sma{n}"] = (f"Distance from {n}-day average", "technical", _pct)


def feature_spec(name: str) -> tuple[str, str, callable]:
    return SPEC.get(name, (name.replace("_", " "), "technical", _plain))


def explain(model: LoadedModel, feats: pd.DataFrame, top: int = 6) -> dict | None:
    """Drivers of the prediction for the *last* row of `feats`."""
    if feats.empty:
        return None
    cols = model.features
    row = feats.reindex(columns=cols).iloc[[-1]]
    booster = model.clf.get_booster()
    # Same trees predict_proba uses: early stopping keeps only the first best_iteration trees.
    n_trees = int(model.meta.get("best_iteration") or 0)
    contrib = booster.predict(xgboost.DMatrix(row, feature_names=cols), pred_contribs=True,
                              iteration_range=(0, n_trees))[0]
    margin, bias = float(contrib.sum()), float(contrib[-1])
    p = 1 / (1 + math.exp(-margin))
    slope = p * (1 - p) * 100          # percentage points of probability per unit of log-odds

    drivers, groups = [], {g: 0.0 for g in GROUPS}
    for name, c, v in zip(cols, contrib[:-1], row.iloc[0]):
        label, group, fmt = feature_spec(name)
        effect = float(c) * slope
        groups[group] += effect
        if abs(effect) < 0.05:
            continue
        drivers.append({"feature": name, "label": label, "group": group,
                        "value": None if pd.isna(v) else float(v),
                        "value_text": "n/a" if pd.isna(v) else fmt(float(v)),
                        "effect_pp": round(effect, 2)})
    drivers.sort(key=lambda d: -abs(d["effect_pp"]))
    return {
        "base_probability": round(1 / (1 + math.exp(-bias)) * 100, 1),
        "probability": round(p * 100, 1),
        "drivers": drivers[:top],
        "groups": [{"group": g, "label": GROUPS[g], "effect_pp": round(e, 2)}
                   for g, e in sorted(groups.items(), key=lambda kv: -abs(kv[1])) if g in {
                       SPEC.get(c, ("", "technical", None))[1] for c in cols}],
    }
