"""Semantic layer: causality, after-close handling, event timing, and that the model can actually use it."""
import numpy as np
import pandas as pd
import pytest

from stockai.explain import explain
from stockai.features import SEMANTIC_FEATURES, TECH_FEATURES, build_features
from stockai.markets import IN, US
from stockai.model import LoadedModel
from stockai.semantic import analyst_features, event_features, news_features, trading_day_position
from stockai.training import fit_model

from test_core import synthetic

IDX = pd.bdate_range("2026-03-02", periods=20)   # Mon 2 Mar ... Fri 27 Mar


def news(rows):
    return pd.DataFrame({"ts": pd.to_datetime([r[0] for r in rows], utc=True),
                         "score": [r[1] for r in rows]})


def test_after_close_headline_is_credited_to_next_trading_day():
    # US close is 16:00 New York = 20:00 UTC in March (EDT after 8 Mar -> use a pre-DST date: EST, 21:00 UTC)
    ts = pd.Series(pd.to_datetime(["2026-03-03 15:00", "2026-03-03 22:00", "2026-03-06 22:00", "2026-03-07 15:00"], utc=True))
    pos = trading_day_position(IDX, ts, US)
    assert [IDX[i].strftime("%a %d") for i in pos] == ["Tue 03", "Wed 04", "Mon 09", "Mon 09"]


def test_news_features_are_causal():
    rng = np.random.default_rng(1)
    days = pd.date_range("2026-03-02", periods=40, freq="D", tz="UTC") + pd.Timedelta(hours=15)
    n = news(list(zip(days, rng.uniform(-1, 1, len(days)))))
    full = news_features(IDX, n, US)
    cut = news_features(IDX[:10], n[n["ts"] < pd.Timestamp("2026-03-16", tz="UTC")], US)
    # rows before the cut only depend on headlines published before it
    pd.testing.assert_frame_equal(full.iloc[:9], cut.iloc[:9])


def test_news_features_values_and_missing():
    n = news([("2026-03-03 15:00", 0.8), ("2026-03-03 16:00", 0.4), ("2026-03-04 15:00", -0.6)])
    f = news_features(IDX, n, US)
    assert f.loc["2026-03-03", "sent_1d"] == pytest.approx(0.6)
    assert np.isnan(f.loc["2026-03-02", "sent_1d"])
    assert f.loc["2026-03-04", "sent_3d"] == pytest.approx((0.8 + 0.4 - 0.6) / 3)
    assert f.loc["2026-03-04", "neg_share_7d"] == pytest.approx(1 / 3)
    assert news_features(IDX, news([]), US).isna().all().all()


def test_event_features_do_not_leak_same_day_report():
    earn = pd.DataFrame({"surprise": [10.0, np.nan]},
                        index=pd.to_datetime(["2026-03-10", "2026-04-20"]))
    f = event_features(IDX, earn)
    assert f.loc["2026-03-09", "days_to_earn"] == 1
    assert f.loc["2026-03-10", "days_to_earn"] == 0
    assert np.isnan(f.loc["2026-03-10", "last_surprise"])         # report day itself: not yet known
    assert f.loc["2026-03-11", "last_surprise"] == 10.0 and f.loc["2026-03-11", "days_since_earn"] == 1
    assert f.loc["2026-03-26", "days_to_earn"] == 25


def test_analyst_features_exclude_today_and_missing_coverage_is_nan():
    acts = pd.DataFrame({"action": ["up", "up", "down"], "target_chg": [0.1, np.nan, -0.05]},
                        index=pd.to_datetime(["2026-02-02", "2026-03-10", "2026-03-10"]))
    f = analyst_features(IDX, acts)
    assert f.loc["2026-03-10", "analyst_net_30d"] == 0             # that day's actions not visible yet
    assert f.loc["2026-03-11", "analyst_net_30d"] == 0             # +1 up -1 down +1 up(Feb 2) ... within 30d
    assert analyst_features(IDX, pd.DataFrame()).isna().all().all()


def test_build_features_adds_semantic_columns():
    df = synthetic(300)
    assert build_features(df)[SEMANTIC_FEATURES].isna().all().all()
    sem = pd.DataFrame({"sent_1d": 0.5}, index=df.index)
    assert (build_features(df, None, sem)["sent_1d"] == 0.5).all()


def planted(n=1400, seed=3):
    """Returns whose sign is partly predicted by a 'headline tone' known at the previous close."""
    rng = np.random.default_rng(seed)
    tone = rng.normal(0, 1, n)
    ret = np.r_[0.0, 0.004 * tone[:-1]] + rng.normal(0, 0.012, n)
    close = 100 * np.exp(np.cumsum(ret))
    df = synthetic(n, seed)
    df = df.assign(Close=close, Open=np.r_[close[0], close[:-1]], High=close * 1.004, Low=close * 0.996)
    sem = pd.DataFrame(np.nan, index=df.index, columns=SEMANTIC_FEATURES)
    sem["sent_1d"] = tone
    return df, sem


def test_model_finds_signal_in_semantics_and_ablation_says_so():
    df, sem = planted()
    info = {"news_start": str(df.index[0].date()), "scorer": "test", "headlines": 1}
    clf, meta = fit_model(df, None, US, "PLANT", sem, info)
    sm = meta["semantic"]
    assert sm["enabled"] and sm["features"] == ["sent_1d"]          # sparse/empty semantic columns are dropped
    ab = sm["ablation"]
    assert ab["semantic_only"]["auc"] > 0.58
    assert ab["combined"]["auc"] > ab["technical_only"]["auc"] + 0.03
    assert ab["semantic_adds"] and meta["has_edge"]
    assert "sent_1d" in meta["features"] and meta["schema"] == 2


def test_no_signal_when_semantics_are_noise():
    df = synthetic(1400, 5)
    sem = pd.DataFrame(np.nan, index=df.index, columns=SEMANTIC_FEATURES)
    sem["sent_1d"] = np.random.default_rng(9).normal(size=len(df))
    info = {"news_start": str(df.index[0].date()), "scorer": "test", "headlines": 1}
    _, meta = fit_model(df, None, IN, "NOISE", sem, info)
    assert not meta["has_edge"] and not meta["semantic"]["ablation"]["semantic_adds"] or \
        meta["semantic"]["ablation"]["semantic_only"]["auc"] < 0.56


def test_explanation_sums_to_prediction_and_names_semantic_driver():
    df, sem = planted()
    sem.iloc[-1, sem.columns.get_loc("sent_1d")] = 2.5              # very bullish tone on the last day
    info = {"news_start": str(df.index[0].date()), "scorer": "test", "headlines": 1}
    clf, meta = fit_model(df, None, US, "PLANT", sem, info)
    feats = build_features(df, None, sem)
    ex = explain(LoadedModel(clf, meta), feats)
    assert ex["drivers"][0]["feature"] == "sent_1d" and ex["drivers"][0]["effect_pp"] > 0
    assert ex["probability"] > ex["base_probability"]
    p_model = clf.predict_proba(feats[meta["features"]].iloc[[-1]])[0, 1] * 100
    assert ex["probability"] == pytest.approx(p_model, abs=0.2)
