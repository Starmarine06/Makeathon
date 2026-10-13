"""Cross-sectional engine: no look-ahead, correct cost accounting, and it finds a planted signal."""
import numpy as np
import pandas as pd
import pytest

from stockai import xsection as xs
from stockai.markets import US


def synthetic_panel(n_days=1100, n_names=24, signal=0.0, seed=0):
    """Random-walk stocks. With signal>0, tomorrow's relative return depends on today's 'mom_12_1'."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    names = [f"S{i}" for i in range(n_names)]
    idx = pd.MultiIndex.from_product([dates, names], names=[None, "symbol"])
    p = pd.DataFrame(index=idx)
    for c in xs.PRICE_FEATURES + ["days_to_earn", "days_since_earn", "last_surprise"]:
        p[c] = rng.normal(size=len(idx))
    p["mom_12_1"] = rng.normal(size=len(idx))
    daily = rng.normal(0, 0.012, len(idx))
    # forward 5d return = noise + signal * today's momentum score
    p["fwd"] = daily * np.sqrt(xs.H) + signal * p["mom_12_1"].to_numpy()
    return p


def test_ranking_is_cross_sectional_and_neutral_on_missing():
    p = synthetic_panel(30, 20)
    p.iloc[0, p.columns.get_loc("rsi14")] = np.nan
    r = xs.ranked(p, ["rsi14", "ret_5"])
    assert r.loc[p.index[0], "rsi14"] == 0.0
    per_date = r["ret_5"].groupby(level=0).agg(["min", "max"])
    assert (per_date["min"] >= -0.5).all() and (per_date["max"] <= 0.5).all()


def test_walk_forward_scores_are_out_of_sample_only():
    p = synthetic_panel(1000, 20)
    s = xs.walk_forward_scores(p, xs.PRICE_FEATURES)
    first = s.index.get_level_values(0).min()
    assert first == p.index.get_level_values(0).unique()[xs.MIN_TRAIN_DAYS]


def test_model_finds_planted_signal_and_not_noise():
    cols = ["mom_12_1", "rsi14", "ret_5"]
    good = synthetic_panel(signal=0.004)
    s = xs.walk_forward_scores(good, cols)
    assert xs.information_coefficient(s, good["fwd"].reindex(s.index))["mean_ic"] > 0.1
    bad = synthetic_panel(signal=0.0, seed=4)
    s = xs.walk_forward_scores(bad, cols)
    ic = xs.information_coefficient(s, bad["fwd"].reindex(s.index))
    assert abs(ic["ic_t"]) < 3


def test_future_returns_do_not_change_past_scores():
    """Scores for a block must not depend on returns realised after that block's training cutoff."""
    p = synthetic_panel(1000, 20, signal=0.003)
    cols = ["mom_12_1", "rsi14"]
    a = xs.walk_forward_scores(p, cols)
    q = p.copy()
    cutoff = p.index.get_level_values(0).unique()[xs.MIN_TRAIN_DAYS + xs.TEST_BLOCK - 1]
    late = q.index.get_level_values(0) > cutoff
    q.loc[late, "fwd"] = np.random.default_rng(1).normal(size=late.sum())   # scramble the future
    b = xs.walk_forward_scores(q, cols)
    first_block = a.index.get_level_values(0) <= cutoff
    pd.testing.assert_series_equal(a[first_block], b.reindex(a.index)[first_block])


def test_backtest_charges_turnover_and_matches_hand_calc():
    dates = pd.bdate_range("2024-01-01", periods=xs.H * 3 + 1)
    names = [f"N{i}" for i in range(20)]
    idx = pd.MultiIndex.from_product([dates, names])
    scores = pd.Series(np.tile(np.arange(20.0), len(dates)), index=idx)        # always picks N10..N19
    fwd = pd.Series(np.tile(np.r_[np.zeros(10), np.full(10, 0.01)], len(dates)), index=idx)
    free = xs.backtest_topk(scores, fwd, k=10, cost_bps=0)
    paid = xs.backtest_topk(scores, fwd, k=10, cost_bps=50)
    assert free["periods"] == 4 and free["net"]["total_return"] == pytest.approx(1.01 ** 4 - 1)
    # only the first rebalance has turnover (100%): 2 sides * 50 bps once
    assert paid["net"]["total_return"] == pytest.approx((1.01 - 0.01) * 1.01 ** 3 - 1, rel=1e-6)
    assert paid["turnover_per_rebalance"] == pytest.approx(0.25)
    assert free["benchmark"]["total_return"] == pytest.approx(1.005 ** 4 - 1)


def test_pooled_dip_labels_are_vol_scaled_and_model_finds_planted_risk():
    from stockai import pooled
    p = synthetic_panel(1100, 20, seed=2)
    rng = np.random.default_rng(5)
    p["vol_60"] = 0.01
    risk = p["bb_pos"].to_numpy()                       # planted: high bb_pos today -> deeper drop ahead
    p["dd"] = -np.abs(rng.normal(0, 0.02, len(p))) - 0.012 * np.clip(risk, 0, None)
    lab = pooled.labels(p)
    assert set(lab.dropna().unique()) <= {0.0, 1.0}
    cut = pooled.dip_threshold(0.01)
    assert cut == pytest.approx(-1.5 * 0.01 * 5 ** 0.5)
    assert ((p["dd"] <= cut).astype(float)[lab.notna()] == lab.dropna()).all()
    _, meta = pooled.fit_pooled(p, ["bb_pos", "rsi14", "ret_5", "vol_regime"], US)
    h = meta["holdout"]
    assert meta["kind"] == "dip_pooled" and h["auc"] > 0.6 and meta["has_skill"]
    assert h["dip_rate_when_top_quintile"] > h["dip_rate_when_bottom_quintile"]
