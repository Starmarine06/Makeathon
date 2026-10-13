"""Live news + dip-risk plumbing, fully offline (network and scorers are stubbed)."""
import numpy as np
import pandas as pd
import pytest

import app as app_mod
from stockai import data as data_mod
from stockai import semantic
from stockai.markets import US

NOW = pd.Timestamp("2026-10-14 12:00", tz="UTC")


def news(rows):
    return pd.DataFrame({"ts": [NOW - pd.Timedelta(hours=h) for h, _ in rows], "score": [s for _, s in rows],
                         "title": [f"t{i}" for i in range(len(rows))], "source": "x", "link": ""})


def test_alert_high_on_cluster_of_strong_negatives():
    a = semantic.news_alert(news([(1, -0.8), (2, -0.7), (5, -0.9), (7, 0.2)]), NOW)
    assert a["level"] == "high" and "3 strongly negative" in a["reason"]


def test_alert_elevated_on_sour_day_and_none_when_calm():
    sour = news([(h, -0.4) for h in (7, 9, 11, 13, 15, 17)])
    assert semantic.news_alert(sour, NOW)["level"] == "elevated"
    assert semantic.news_alert(news([(1, 0.5), (2, -0.2), (3, 0.1), (8, 0.3), (9, 0.0)]), NOW) is None
    assert semantic.news_alert(news([(30, -0.9), (31, -0.9), (32, -0.9)]), NOW) is None   # too old to matter
    assert semantic.news_alert(pd.DataFrame(), NOW) is None


class FakeScorer(semantic.Scorer):
    name = "fake"

    def score(self, texts):
        return [-0.9 if "plunge" in t else 0.4 for t in texts]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic, "CACHE_DIR", tmp_path)
    data_mod._cache.clear()
    yield
    data_mod._cache.clear()


def _items(*titles):
    now = pd.Timestamp.now(tz="UTC")
    return [{"ts": (now - pd.Timedelta(minutes=10 * i)).isoformat(), "title": t, "source": "S", "link": "https://e.x/a"}
            for i, t in enumerate(titles)]


def test_latest_news_scores_dedupes_persists_and_caches(isolated, monkeypatch):
    calls, feed = [], _items("Apple plunge", "Apple up")      # a real feed repeats the same publish times
    monkeypatch.setattr(semantic, "_fetch_query", lambda q, mk: calls.append(q) or feed)
    df = semantic.latest_news("AAPL", US, FakeScorer(), hours=48, ttl=60)
    assert list(df["score"]) == [-0.9, 0.4] and "when:2d" in calls[0]
    semantic.latest_news("AAPL", US, FakeScorer(), hours=48, ttl=60)
    assert len(calls) == 1                                       # second poll served from cache
    stored = semantic._store_path("AAPL").read_text()
    assert "Apple plunge" in stored
    data_mod._cache.clear()
    semantic.latest_news("AAPL", US, FakeScorer(), hours=48, ttl=60)
    import json
    assert len(json.loads(semantic._store_path("AAPL").read_text())["items"]) == 2   # no duplicates stored


def test_latest_news_survives_fetch_failure(isolated, monkeypatch):
    monkeypatch.setattr(semantic, "_fetch_query", lambda q, mk: None)
    assert semantic.latest_news("AAPL", US, FakeScorer()).empty


def test_news_endpoint_returns_headlines_alert_and_escapable_links(isolated, monkeypatch):
    now = pd.Timestamp.now(tz="UTC")
    df = pd.DataFrame({"ts": [now - pd.Timedelta(hours=h) for h in (1, 2, 3)],
                       "title": ["A plunge <b>x</b>", "B plunge", "C plunge"], "source": "S",
                       "link": ["javascript:alert(1)", "https://ok.example/x", ""], "score": [-0.9, -0.8, -0.7]})
    monkeypatch.setattr(semantic, "latest_news", lambda *a, **k: df)
    monkeypatch.setattr(semantic, "get_scorer", lambda name=None: FakeScorer())
    r = app_mod.app.test_client().get("/api/news/AAPL")
    j = r.get_json()
    assert r.status_code == 200 and j["alert"]["level"] == "high" and len(j["headlines"]) == 3
    assert j["headlines"][0]["link"] == "" and j["headlines"][1]["link"].startswith("https://")   # unsafe scheme dropped
    assert j["tone_24h"]["negative"] == 3 and len({h["id"] for h in j["headlines"]}) == 3


def test_news_endpoint_rejects_bad_symbol():
    assert app_mod.app.test_client().get("/api/news/..%2Fetc").status_code in (400, 404)


def test_dip_level_thresholds_news_override_and_pooled_threshold(monkeypatch):
    from types import SimpleNamespace
    h = {"base_rate": 0.12, "auc": 0.65, "auc_ci90": [0.61, 0.69], "volatility_regime_only_auc": 0.54,
         "lift_top_quintile": 1.7, "dip_rate_when_top_quintile": 0.2, "dip_rate_when_bottom_quintile": 0.05}
    meta = {"kind": "dip_pooled", "holdout": h, "train_base_rate": 0.12, "k_sigma": 1.5, "horizon_days": 5,
            "has_skill": True}
    model = SimpleNamespace(meta=meta)
    monkeypatch.setattr(app_mod, "explain", lambda *a, **k: {"drivers": []})
    for p, expected in ((0.13, "LOW"), (0.18, "ELEVATED"), (0.30, "HIGH")):
        monkeypatch.setattr(app_mod, "predict_proba", lambda m, f, p=p: np.array([0.1, p]))
        out = app_mod._dip(model, None, 0.02, None)
        assert out["level"] == expected
    assert out["threshold_pct"] == pytest.approx(1.5 * 0.02 * 5 ** 0.5 * 100, abs=0.06)   # 6.7% for a 2%/day stock
    monkeypatch.setattr(app_mod, "predict_proba", lambda m, f: np.array([0.1, 0.12]))
    assert app_mod._dip(model, None, 0.02, {"level": "high", "reason": "x"})["level"] == "ELEVATED"
    none = app_mod._dip(None, None, 0.02, None)
    assert none["level"] == "LOW" and not none["model"]
