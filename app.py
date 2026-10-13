"""Stock Trading AI: one Flask app for US and Indian (NSE/BSE) stocks.

    python app.py            # http://127.0.0.1:5000
    FLASK_DEBUG=1 python app.py
    HOST=0.0.0.0 python app.py   # only if you really want it reachable from your network
"""
import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from stockai import data as data_mod
from stockai import semantic, xsection
from stockai.explain import explain
from stockai.features import WARMUP, build_features
from stockai.markets import (IN_INDICES, NIFTY, US_NAMES, Market, SymbolError, display_name,
                             exchange_label, is_session_open, resolve_symbol)
from stockai.model import (HEURISTIC_BUY, HEURISTIC_SELL, heuristic_proba, load_model,
                           predict_proba, public_meta)
from stockai.signals import detect_signals, zone

app = Flask(__name__)
log = logging.getLogger("stockai")

CHART_BARS = 750          # ~3 years of daily candles
ANALYSIS_TTL = 90         # seconds; short so dip risk follows the live bar and fresh news
NEW_NEWS_WINDOWS = 8      # max uncached weekly news windows one web request may fetch
SHOWN_HEADLINES = 10
MARKER_COLORS = {"BUY": "0,255,136", "SELL": "255,68,68"}


class NoData(Exception):
    pass


def _headlines(news, mk: Market, limit: int = SHOWN_HEADLINES) -> list[dict]:
    if news is None or news.empty:
        return []
    tz = ZoneInfo(mk.tz)
    rows = []
    for r in news.sort_values("ts", ascending=False).head(limit).itertuples():
        link = r.link if urlparse(r.link).scheme in ("http", "https") else ""
        rows.append({"id": hashlib.sha1(r.title.encode("utf-8")).hexdigest()[:12],
                     "time": r.ts.tz_convert(tz).strftime("%d %b %H:%M"), "iso": r.ts.isoformat(),
                     "title": r.title, "source": r.source, "link": link, "score": round(float(r.score), 2)})
    return rows


DIP_RATIO_ELEVATED, DIP_RATIO_HIGH = 1.4, 2.0   # model probability vs the stock's normal dip rate


def _dip(model, feats, sigma, alert) -> dict:
    """Chance of a drop of at least `threshold` below the last close within `horizon` sessions.

    model: pooled (market-wide) dip model, or a per-symbol one as fallback. sigma: stock's daily volatility,
    which sets how big a "dip" is for this stock under the pooled model."""
    level = "LOW"
    out = {"model": False, "news_alert": alert}
    if model:
        m, h = model.meta, model.meta["holdout"]
        p = float(predict_proba(model, feats)[-1])
        base = float(h["base_rate"] or m["train_base_rate"])
        ratio = p / base if base > 0 else 1.0
        level = "HIGH" if ratio >= DIP_RATIO_HIGH else "ELEVATED" if ratio >= DIP_RATIO_ELEVATED else "LOW"
        if m.get("kind") == "dip_pooled":
            thr = m["k_sigma"] * sigma * (m["horizon_days"] ** 0.5) if sigma == sigma else float("nan")
            lo, hi = h["auc_ci90"]
            auc_text = f"{h['auc']:.3f} (90% CI {lo:.3f}-{hi:.3f})"
            vol_auc = h["volatility_regime_only_auc"]
        else:
            thr = abs(m["threshold"])
            auc_text, vol_auc = f"{h['auc']:.3f}", h["volatility_only_auc"]
        ex = explain(model, feats, top=4)
        out.update({"model": True, "pooled": m.get("kind") == "dip_pooled", "probability": round(p * 100, 1),
                    "base_rate": round(base * 100, 1),
                    "threshold_pct": None if thr != thr else round(thr * 100, 1), "horizon": m["horizon_days"],
                    "has_skill": m["has_skill"], "auc_text": auc_text, "volatility_only_auc": round(vol_auc, 3),
                    "lift": None if h["lift_top_quintile"] is None else round(h["lift_top_quintile"], 2),
                    "safest_rate": None if h.get("dip_rate_when_bottom_quintile") is None
                    else round(h["dip_rate_when_bottom_quintile"] * 100, 1),
                    "top_rate": None if h.get("dip_rate_when_top_quintile") is None
                    else round(h["dip_rate_when_top_quintile"] * 100, 1),
                    "drivers": (ex or {}).get("drivers", [])})
    if alert and alert["level"] == "high" and level == "LOW":
        level = "ELEVATED"          # a cluster of strongly negative headlines overrides a calm model
    out["level"] = level
    return out


def _events(earn) -> dict:
    out = {"next_earnings": None, "last_surprise": None, "last_earnings": None}
    if earn is None or earn.empty:
        return out
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    future = earn[earn.index >= today]
    if len(future):
        out["next_earnings"] = future.index[0].date().isoformat()
    past = earn[(earn.index < today) & earn["surprise"].notna()]
    if len(past):
        out["last_earnings"] = past.index[-1].date().isoformat()
        out["last_surprise"] = round(float(past["surprise"].iloc[-1]), 1)
    return out


def _tone(news) -> dict | None:
    if news is None or news.empty:
        return None
    recent = news[news["ts"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)]
    if recent.empty:
        return None
    return {"mean": round(float(recent["score"].mean()), 2), "count": int(len(recent)),
            "positive": int((recent["score"] > 0.3).sum()), "negative": int((recent["score"] < -0.3).sum())}


def analyze(symbol: str, mk: Market) -> dict:
    """Daily candles + model probabilities + signals + explanation for a symbol (cached)."""
    def build():
        df = data_mod.fetch_daily(symbol, years=5)
        if df.empty or len(df) < WARMUP + 20:
            raise NoData(f"No price data for {symbol}. Check the symbol and market.")
        ctx = data_mod.align_context(data_mod.fetch_context(mk), df.index)
        model = load_model(symbol)
        dip_model = load_model(f"POOLED_{mk.code}__dip") or load_model(symbol + "__dip")
        msem = (model.meta.get("semantic") or {}) if model else {}

        sem = news = earn = None
        try:
            if msem.get("enabled"):
                # Same scorer as training, or the sentiment scale would silently differ.
                start = max(df.index[0].date(), datetime.fromisoformat(msem["news_start"]).date())
                sem, info = semantic.build_semantic(symbol, mk, df.index, scorer=msem["scorer"],
                                                    start=start, max_new=NEW_NEWS_WINDOWS)
                news, earn = info["news"], info["earnings"]
            elif not symbol.startswith("^"):
                today = datetime.now(timezone.utc).date()
                news = semantic.load_news(symbol, mk, today - timedelta(days=21), today,
                                          semantic.get_scorer(), max_new=4)
                earn = data_mod.fetch_earnings(symbol)
        except Exception:
            log.exception("semantic layer failed for %s; continuing without it", symbol)

        if earn is None and not symbol.startswith("^"):
            earn = data_mod.fetch_earnings(symbol)
        alert = None
        try:   # freshest headlines drive the news-alert part of dip risk
            alert = semantic.news_alert(semantic.latest_news(symbol, mk, semantic.get_scorer(msem.get("scorer"))))
        except Exception:
            log.exception("latest news failed for %s", symbol)

        feats = build_features(df, ctx, sem)
        dip_feats, sigma = feats, float("nan")
        if dip_model:
            if dip_model.meta.get("kind") == "dip_pooled":
                events = sem if sem is not None else pd.concat(
                    [pd.DataFrame(np.nan, index=df.index, columns=semantic.NEWS_FEATURES),
                     semantic.event_features(df.index, earn), pd.DataFrame(np.nan, index=df.index, columns=semantic.ANALYST_FEATURES)],
                    axis=1)
                dip_feats = xsection.stock_frame(symbol, df, ctx, events).droplevel(1)
            sigma = float(df["Close"].pct_change().rolling(60).std().iloc[-1])
        dip = _dip(dip_model, dip_feats, sigma, alert)
        why = None
        if model:
            probs = predict_proba(model, feats)
            buy_th, sell_th, cooldown = model.buy_th, model.sell_th, model.meta.get("cooldown", mk.cooldown)
            why = explain(model, feats)
        else:
            probs = heuristic_proba(feats)
            buy_th, sell_th, cooldown = HEURISTIC_BUY, HEURISTIC_SELL, mk.cooldown
        probs = probs.copy()
        probs[:WARMUP] = 0.5  # incomplete rolling windows: never signal on them
        context = {"explanation": why, "headlines": _headlines(news, mk), "tone": _tone(news),
                   "events": _events(earn), "semantic_in_model": bool(msem.get("enabled"))}
        return {"df": df, "probs": probs, "model": model, "buy_th": buy_th, "sell_th": sell_th,
                "cooldown": cooldown, "why": context, "dip": dip}
    return data_mod.cached(("analysis", symbol), ANALYSIS_TTL, build)


def _prediction(a: dict) -> dict:
    p = float(a["probs"][-1])
    return {"probability": round(p * 100, 1), "signal": zone(p, a["buy_th"], a["sell_th"]),
            "buy_threshold": round(a["buy_th"] * 100, 1), "sell_threshold": round(a["sell_th"] * 100, 1),
            "source": "model" if a["model"] else "heuristic"}


def _quote(df) -> dict:
    last, prev = df.iloc[-1], df["Close"].iloc[-2]
    price = float(last["Close"])
    return {"price": round(price, 2), "prev_close": round(float(prev), 2),
            "change": round(price - float(prev), 2), "change_pct": round((price / float(prev) - 1) * 100, 2),
            "open": round(float(last["Open"]), 2), "high": round(float(last["High"]), 2),
            "low": round(float(last["Low"]), 2), "volume": int(last["Volume"]),
            "as_of": df.index[-1].date().isoformat()}


def _context(symbol: str, mk: Market) -> dict:
    return {"symbol": symbol, "name": display_name(symbol), "market": mk.code,
            "exchange": exchange_label(symbol, mk), "currency": mk.currency, "locale": mk.locale,
            "timezone": mk.tz}


def _error(message: str, status: int):
    return jsonify({"error": message}), status


@app.route("/")
def home():
    nifty = [{"value": s, "label": n} for s, n in NIFTY.items()]
    nifty += [{"value": s, "label": n} for s, n in IN_INDICES.items()]
    us = [{"value": s, "label": n} for s, n in US_NAMES.items()]
    return render_template("index.html", suggestions=nifty + us)


@app.route("/api/historical/<symbol>")
def historical(symbol):
    try:
        symbol, mk = resolve_symbol(symbol, request.args.get("market", "auto"))
        a = analyze(symbol, mk)
        df, probs = a["df"], a["probs"]
        n = min(CHART_BARS, len(df))
        df_w, probs_w = df.iloc[-n:], probs[-n:]

        in_sample_until = a["model"].meta["in_sample_until"] if a["model"] else None
        dates = [d.date().isoformat() for d in df_w.index]
        candles = [{"time": t, "open": round(float(o), 2), "high": round(float(h), 2),
                    "low": round(float(l), 2), "close": round(float(c), 2)}
                   for t, o, h, l, c in zip(dates, df_w["Open"], df_w["High"], df_w["Low"], df_w["Close"])]

        sigs = detect_signals(probs_w, a["buy_th"], a["sell_th"], a["cooldown"])
        markers = []
        for s in sigs:
            t = dates[s["index"]]
            oos = in_sample_until is None or t > in_sample_until
            rgb = MARKER_COLORS[s["type"]]
            markers.append({"time": t, "oos": oos, "type": s["type"],
                            "position": "belowBar" if s["type"] == "BUY" else "aboveBar",
                            "shape": "arrowUp" if s["type"] == "BUY" else "arrowDown",
                            "color": f"rgba({rgb},{1 if oos else 0.3})",
                            "text": f"{s['type']} {s['prob'] * 100:.0f}%"})

        return jsonify({
            **_context(symbol, mk),
            "candles": candles, "markers": markers,
            "quote": _quote(df), "prediction": _prediction(a),
            "signals": {"total": len(markers), "buy": sum(m["type"] == "BUY" for m in markers),
                        "sell": sum(m["type"] == "SELL" for m in markers),
                        "out_of_sample": sum(m["oos"] for m in markers)},
            "model": public_meta(a["model"].meta) if a["model"] else None,
            "why": a["why"], "dip": a["dip"],
        })
    except SymbolError as exc:
        return _error(str(exc), 400)
    except NoData as exc:
        return _error(str(exc), 404)
    except Exception:
        log.exception("historical failed for %s", symbol)
        return _error("Internal error while loading data", 500)


@app.route("/api/live/<symbol>")
def live(symbol):
    try:
        symbol, mk = resolve_symbol(symbol, request.args.get("market", "auto"))
        a = analyze(symbol, mk)
        df = a["df"]
        sess = data_mod.fetch_session(symbol, mk)
        tz = ZoneInfo(mk.tz)

        if sess is None:  # intraday feed unavailable: fall back to the last daily bar
            q, bar, is_open = _quote(df), None, False
        else:
            prior = df["Close"][df.index < pd.Timestamp(sess["date"])]
            prev = float(prior.iloc[-1]) if len(prior) else float(df["Close"].iloc[0])
            price = sess["close"]
            q = {"price": round(price, 2), "prev_close": round(prev, 2),
                 "change": round(price - prev, 2), "change_pct": round((price / prev - 1) * 100, 2),
                 "open": round(sess["open"], 2), "high": round(sess["high"], 2),
                 "low": round(sess["low"], 2), "volume": sess["volume"], "as_of": sess["date"]}
            bar = {"time": sess["date"], "open": q["open"], "high": q["high"], "low": q["low"], "close": q["price"]}
            is_open = is_session_open(mk, sess["last_ts"])

        return jsonify({**_context(symbol, mk), "quote": q, "bar": bar, "market_open": is_open,
                        "prediction": _prediction(a), "dip": a["dip"], "server_time": datetime.now(tz).strftime("%H:%M:%S")})
    except SymbolError as exc:
        return _error(str(exc), 400)
    except NoData as exc:
        return _error(str(exc), 404)
    except Exception:
        log.exception("live failed for %s", symbol)
        return _error("Internal error while loading data", 500)


@app.route("/api/news/<symbol>")
def news(symbol):
    """Freshest scored headlines for the company plus a bad-news alert. Cheap to poll (45s shared cache)."""
    try:
        symbol, mk = resolve_symbol(symbol, request.args.get("market", "auto"))
        m = load_model(symbol) or load_model(f"POOLED_{mk.code}__dip")
        scorer = semantic.get_scorer(((m.meta.get("semantic") or {}).get("scorer")) if m else None)
        df = semantic.latest_news(symbol, mk, scorer, hours=72)
        day = df[df["ts"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)] if len(df) else df
        tone = None if day.empty else {"mean": round(float(day["score"].mean()), 2), "count": int(len(day)),
                                       "positive": int((day["score"] > 0.3).sum()),
                                       "negative": int((day["score"] < -0.3).sum())}
        return jsonify({"symbol": symbol, "headlines": _headlines(df, mk, limit=25), "tone_24h": tone,
                        "alert": semantic.news_alert(df), "scorer": scorer.name,
                        "server_time": datetime.now(ZoneInfo(mk.tz)).strftime("%H:%M:%S")})
    except SymbolError as exc:
        return _error(str(exc), 400)
    except Exception:
        log.exception("news failed for %s", symbol)
        return _error("Internal error while loading news", 500)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG") == "1"
    print(f"Stock Trading AI -> http://{host}:{port}  (train models with: python train.py SYMBOL)")
    app.run(host=host, port=port, debug=debug)
