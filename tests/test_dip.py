"""Dip-risk model: target definition, no leakage of the threshold, and it learns volatility clustering."""
import numpy as np
import pandas as pd
import pytest

from stockai.markets import US
from stockai.training import DIP_HORIZON, dip_target, fit_dip_model

from test_core import synthetic


def garch_like(n=2200, seed=11):
    """Returns with persistent calm (0.6%/day) and stormy (2.5%/day) regimes lasting ~1-3 months."""
    rng = np.random.default_rng(seed)
    vol = np.concatenate([np.full(rng.integers(20, 60), rng.choice([0.006, 0.025])) for _ in range(n // 20)])[:n]
    ret = rng.normal(0, 1, n) * vol
    close = 100 * np.exp(np.cumsum(ret))
    df = synthetic(n, seed)
    return df.assign(Close=close, Open=np.r_[close[0], close[:-1]], High=close * (1 + vol * 0.8), Low=close * (1 - vol * 0.8))


def test_dip_target_is_worst_future_low_vs_close():
    idx = pd.bdate_range("2026-01-05", periods=10)
    df = pd.DataFrame({"Close": 100.0, "Low": [99, 99, 98, 90, 99, 99, 99, 99, 99, 99.0]}, index=idx)
    t = dip_target(df, horizon=3)
    assert t.iloc[0] == pytest.approx(-0.10)       # day 0 looks at the lows of days 1..3: 99, 98, 90
    assert t.iloc[-3:].isna().all()                # last `horizon` rows have no complete future window


def test_dip_model_learns_volatility_clustering():
    df = garch_like()
    clf, meta = fit_dip_model(df, None, US, "CLUSTER")
    h = meta["holdout"]
    assert meta["horizon_days"] == DIP_HORIZON and meta["threshold"] <= -0.01
    assert h["auc"] > 0.60                          # stormy regimes keep producing deep drawdowns
    assert h["lift_top_quintile"] > 1.3
    assert meta["has_skill"]


def test_dip_model_has_no_skill_on_iid_noise():
    df = synthetic(1500, 8)
    _, meta = fit_dip_model(df, None, US, "NOISE")
    assert abs(meta["holdout"]["auc"] - 0.5) < 0.1
