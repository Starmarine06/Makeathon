"""Turn a probability series into BUY/SELL chart signals and a simple cost-aware backtest."""
from __future__ import annotations

import numpy as np


def zone(p: float, buy_th: float, sell_th: float) -> str:
    return "BUY" if p >= buy_th else "SELL" if p <= sell_th else "HOLD"


def detect_signals(probs, buy_th: float, sell_th: float, cooldown: int = 2) -> list[dict]:
    """Emit a signal when the probability enters the BUY or SELL zone.

    Signals strictly alternate (never BUY, BUY) and are at least `cooldown` bars apart;
    a signal suppressed by the cooldown fires later if the probability is still in-zone.
    """
    out: list[dict] = []
    state, last = None, -10**9
    for i, p in enumerate(probs):
        z = zone(p, buy_th, sell_th)
        if z != "HOLD" and z != state and i - last >= cooldown:
            out.append({"index": i, "type": z, "prob": float(p)})
            state, last = z, i
    return out


def backtest(probs, next_ret, buy_th: float, sell_th: float, cost_bps: float) -> dict:
    """Long/flat strategy: enter in the BUY zone, exit in the SELL zone, otherwise hold.

    probs[t] is known at close t and earns next_ret[t] (close t -> close t+1).
    cost_bps is charged per side on every position change.
    """
    probs, next_ret = np.asarray(probs, float), np.asarray(next_ret, float)
    pos = np.zeros(len(probs))
    cur = 0.0
    for i, p in enumerate(probs):
        if p >= buy_th:
            cur = 1.0
        elif p <= sell_th:
            cur = 0.0
        pos[i] = cur
    turns = np.abs(np.diff(np.r_[0.0, pos]))
    strat = pos * next_ret - turns * cost_bps / 1e4
    equity = np.cumprod(1 + strat)
    hold = np.cumprod(1 + next_ret)
    drawdown = float((equity / np.maximum.accumulate(equity) - 1).min()) if len(equity) else 0.0
    sd = strat.std()
    return {
        "strategy_return": float(equity[-1] - 1) if len(equity) else 0.0,
        "buy_hold_return": float(hold[-1] - 1) if len(hold) else 0.0,
        "max_drawdown": drawdown,
        "sharpe": float(strat.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
        "trades": int((np.diff(np.r_[0.0, pos]) > 0).sum()),
        "exposure": float(pos.mean()) if len(pos) else 0.0,
        "cost_bps_per_side": cost_bps,
    }
