import numpy as np
import pandas as pd
import pytest

from stockai.features import FEATURES, SEMANTIC_FEATURES, TECH_FEATURES, WARMUP, build_features, make_target
from stockai.markets import IN, US, SymbolError, display_name, is_session_open, resolve_symbol
from stockai.signals import backtest, detect_signals
from stockai.training import fit_model


def synthetic(n=900, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    vol = rng.integers(1_000, 100_000, n).astype(float)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def test_features_are_causal():
    """Features at row t must not change when future rows are removed (no look-ahead)."""
    df = synthetic()
    cut = 600
    full = build_features(df).iloc[:cut]
    part = build_features(df.iloc[:cut])
    pd.testing.assert_frame_equal(full, part)


def test_feature_columns_and_scale_free():
    f = build_features(synthetic())
    assert list(f.columns) == FEATURES
    scaled = build_features(synthetic().assign(**{c: lambda d, c=c: d[c] * 50 for c in ["Open", "High", "Low", "Close"]}))
    # price-level changes must not change ratio features
    pd.testing.assert_series_equal(f["rsi14"], scaled["rsi14"])
    pd.testing.assert_series_equal(f["dist_sma20"], scaled["dist_sma20"], rtol=1e-9)


def test_target_last_row_unknown():
    t = make_target(synthetic())
    assert np.isnan(t.iloc[-1]) and t.iloc[:-1].isin([0.0, 1.0]).all()


@pytest.mark.parametrize("raw,market,expected,code", [
    ("reliance", "auto", "RELIANCE.NS", "IN"),
    ("TCS.NS", "auto", "TCS.NS", "IN"),
    ("500325.BO", "auto", "500325.BO", "IN"),
    ("^NSEI", "auto", "^NSEI", "IN"),
    ("AAPL", "auto", "AAPL", "US"),
    ("INFY", "us", "INFY", "US"),
    ("M&M", "auto", "M&M.NS", "IN"),
    ("BAJAJ-AUTO", "auto", "BAJAJ-AUTO.NS", "IN"),
    ("WIPRO", "in", "WIPRO.NS", "IN"),
])
def test_resolve_symbol(raw, market, expected, code):
    sym, mk = resolve_symbol(raw, market)
    assert (sym, mk.code) == (expected, code)


@pytest.mark.parametrize("bad", ["", "../etc/passwd", "A/B", "A B", "X" * 30, "a;rm", "..\\x", "%2e%2e"])
def test_resolve_rejects_bad_symbols(bad):
    with pytest.raises(SymbolError):
        resolve_symbol(bad)


def test_display_name():
    assert display_name("RELIANCE.NS") == "Reliance Industries"
    assert display_name("AAPL") == "Apple"


def test_detect_signals_alternate_and_respect_cooldown():
    probs = [0.5, 0.6, 0.62, 0.5, 0.4, 0.38, 0.6, 0.35]
    sigs = detect_signals(probs, 0.55, 0.45, cooldown=2)
    # i=7 would be a SELL but is only 1 bar after the BUY at i=6, so the cooldown suppresses it
    assert [(s["type"], s["index"]) for s in sigs] == [("BUY", 1), ("SELL", 4), ("BUY", 6)]


def test_cooldown_defers_not_drops_signal():
    sigs = detect_signals([0.6, 0.4, 0.4, 0.4], 0.55, 0.45, cooldown=2)
    assert [(s["type"], s["index"]) for s in sigs] == [("BUY", 0), ("SELL", 2)]


def test_backtest_costs_and_exposure():
    probs = np.array([0.6, 0.6, 0.4, 0.4])
    rets = np.array([0.01, 0.01, -0.02, -0.02])
    free = backtest(probs, rets, 0.55, 0.45, cost_bps=0)
    costly = backtest(probs, rets, 0.55, 0.45, cost_bps=50)
    assert free["trades"] == 1 and free["exposure"] == 0.5
    assert free["strategy_return"] == pytest.approx(1.01 * 1.01 - 1)
    assert costly["strategy_return"] < free["strategy_return"]
    assert free["buy_hold_return"] < 0 < free["strategy_return"]


def test_fit_model_uses_chronological_holdout():
    df = synthetic(1200)
    clf, meta = fit_model(df, None, US, "SYN")
    assert meta["rows"]["train"] > meta["rows"]["val"]
    assert meta["in_sample_until"] < meta["holdout_start"]
    assert 0.4 < meta["metrics"]["auc"] < 0.6          # random walk: no skill expected
    assert meta["sell_th"] <= 0.5 <= meta["buy_th"]
    assert meta["features"] == TECH_FEATURES and not meta["semantic"]["enabled"]
    assert not meta["has_edge"]


def test_fit_model_needs_enough_rows():
    with pytest.raises(ValueError):
        fit_model(synthetic(200), None, IN, "TINY")


def test_session_open_requires_fresh_bar():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Asia/Kolkata")
    mon_11 = datetime(2026, 10, 12, 11, 0, tzinfo=tz)   # a Monday
    assert is_session_open(IN, datetime(2026, 10, 12, 10, 58, tzinfo=tz), now=mon_11)
    assert not is_session_open(IN, datetime(2026, 10, 12, 9, 30, tzinfo=tz), now=mon_11)   # stale: holiday-like
    assert not is_session_open(IN, datetime(2026, 10, 12, 16, 0, tzinfo=tz), now=datetime(2026, 10, 12, 16, 1, tzinfo=tz))
    assert not is_session_open(IN, datetime(2026, 10, 11, 11, 0, tzinfo=tz), now=datetime(2026, 10, 11, 11, 1, tzinfo=tz))  # Sunday
