"""Model persistence (XGBoost JSON + metadata JSON, no pickle) and inference."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
HEURISTIC_BUY, HEURISTIC_SELL = 0.55, 0.45


@dataclass
class LoadedModel:
    clf: XGBClassifier
    meta: dict

    @property
    def features(self) -> list[str]:
        return self.meta["features"]

    @property
    def buy_th(self) -> float:
        return self.meta["buy_th"]

    @property
    def sell_th(self) -> float:
        return self.meta["sell_th"]


def _stem(symbol: str) -> Path:
    return MODELS_DIR / re.sub(r"[^A-Za-z0-9_-]", "_", symbol)


def save_model(symbol: str, clf: XGBClassifier, meta: dict) -> Path:
    MODELS_DIR.mkdir(exist_ok=True)
    stem = _stem(symbol)
    clf.save_model(str(stem.with_suffix(".json")))
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return stem.with_suffix(".json")


_loaded: dict[str, tuple[float, LoadedModel]] = {}


def load_model(symbol: str) -> LoadedModel | None:
    stem = _stem(symbol)
    model_path, meta_path = stem.with_suffix(".json"), stem.with_suffix(".meta.json")
    if not (model_path.exists() and meta_path.exists()):
        return None
    mtime = max(model_path.stat().st_mtime, meta_path.stat().st_mtime)
    hit = _loaded.get(symbol)
    if hit and hit[0] == mtime:
        return hit[1]
    clf = XGBClassifier()
    clf.load_model(str(model_path))
    loaded = LoadedModel(clf, json.loads(meta_path.read_text()))
    _loaded[symbol] = (mtime, loaded)
    return loaded


def predict_proba(model: LoadedModel, feats: pd.DataFrame) -> np.ndarray:
    X = feats.reindex(columns=model.features)
    return model.clf.predict_proba(X)[:, 1]


def heuristic_proba(feats: pd.DataFrame) -> np.ndarray:
    """Fallback when no model is trained: a squashed vol-adjusted 10-day momentum score."""
    z = feats["ret_10"] / (feats["vol_20"] * np.sqrt(10))
    return (0.5 + 0.12 * np.tanh(z.fillna(0.0))).to_numpy()


def public_meta(meta: dict) -> dict:
    """The subset of metadata the UI shows."""
    keys = ("trained_at", "in_sample_until", "holdout_start", "holdout_end", "metrics",
            "backtest", "has_edge", "interval", "top_features", "semantic")
    return {k: meta.get(k) for k in keys}
