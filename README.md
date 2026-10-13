# Stock Trading AI (US + India)

Flask app that charts daily candles for US and Indian (NSE/BSE) stocks, overlays BUY/SELL zones from a
per-symbol XGBoost model, and shows **how well that model actually did on data it never saw**.

## Run
    pip install -r requirements.txt
    python train.py RELIANCE TCS AAPL        # bare Nifty names get .NS; or: --preset nifty / --preset us
    python app.py                            # http://127.0.0.1:5000

Symbols: `RELIANCE`, `TCS.NS`, `500325.BO`, `^NSEI`, `AAPL`. Use the market selector for collisions
(e.g. `INFY` + US = the ADR). `HOST=0.0.0.0` / `FLASK_DEBUG=1` are opt-in.
Without a trained model the app falls back to a momentum heuristic and says so.

## Design
- `stockai/features.py` — one feature pipeline for training *and* serving (no train/serve skew);
  scale-free, causal features (tested: removing future rows never changes earlier features).
- `stockai/training.py` — chronological 70/10/20 split, early stopping, walk-forward CV, costed backtest,
  AUC significance test. Models saved as XGBoost JSON + metadata (no pickle).
- India specifics: IST handling, NIFTY-50 names, Nifty/India VIX context features, STT-aware costs,
  ₹ and lakh/crore formatting, session open check that respects holidays.
- Chart markers before `in_sample_until` are faded: the model trained on them.

## Semantic layer (why, not just pattern)
`stockai/semantic.py` adds what is being *said* and *scheduled* about a company: FinBERT-scored headlines
(Google News, weekly windows, disk-cached), earnings calendar + EPS surprise, analyst rating/target changes.
A headline published after the close is credited to the next trading day, so features never see the future
(tested). Each prediction comes with a SHAP-style breakdown (`stockai/explain.py`): which inputs moved
P(up), grouped as news / earnings / analysts / market / price pattern, plus the headlines behind it.
Every trained model reports an **ablation** (price-only vs semantic-only vs combined, same rows) so you can
see whether the semantics add information. `python train.py AAPL` (use `--no-news` to skip).

## Cross-sectional ranking (how systematic funds trade equities)
`python xs.py --market in|us [--news]` ranks a universe (Nifty 50 / 50 US large caps) on 12-1 momentum,
reversal, volatility, earnings surprise (and optionally news tone), holds the top 10, rebalances every
5 days, and is evaluated **walk-forward, out-of-sample, after costs** against an equal-weight benchmark and
a plain-momentum baseline, with rank-IC significance.

## Live news + dip risk
`python train_dip.py --market us|in` trains a **pooled** dip-risk model across ~50 stocks (one model per
market). A "dip" = the worst low in the next 5 sessions is at least 1.5 x the stock's own 5-day volatility
below today's close, so the model cannot score by merely noticing that volatile stocks fall further.
The app shows the probability next to the stock's normal rate, the main reasons, and its tested accuracy.

Held-out results (last ~2 years, 90% block-bootstrap CI on AUC; 0.5 = chance):

| Market | AUC | 90% CI | Riskiest 20% of days dipped | Safest 20% dipped | Base rate |
|---|---|---|---|---|---|
| US (50 large caps) | 0.654 | 0.616-0.688 | 21% | 5% | 12% |
| India (Nifty 48) | 0.628 | 0.582-0.662 | 22% | 7% | 13% |

Per-stock dip models (AAPL/NVDA/TSLA) were **not** reliable on their own (too little data), hence the
pooled model. Main drivers are earnings timing, volume and volatility regime. This is a *warning with
quantified odds*, not a prediction that a dip will happen; most flagged days do not dip.

`/api/news/<symbol>` returns freshest headlines (Google News RSS, FinBERT-scored, 45 s shared cache); the
UI polls it every 45 s, marks new items, and shows a banner on a cluster of strongly negative headlines.
That news alert is a transparent rule (3+ strongly negative headlines in 6 h, or a sour 24 h tone) and is
**not backtested**; it can raise a LOW model reading to ELEVATED but never lowers it.

## Results (honest)
Out-of-sample 2020 → Oct 2026, top-10 of ~50, weekly, after costs (2 bps/side US, 12 bps/side India):

| | Sharpe US | Sharpe India |
|---|---|---|
| ML ranking model | 0.98 | 0.84 |
| 12-1 momentum only | 1.28 | 0.74 |
| Equal-weight universe | 1.10 | 1.11 |

Per-stock next-day direction (AAPL/NVDA/TSLA): holdout AUC 0.48-0.49 with or without news. The India model
has a statistically positive rank-IC (t = 2.4) that does not survive costs. Universes are today's survivors,
so every number is optimistic. **No trading edge is claimed.** Educational/research use, not advice.
Selling trading signals or algos generally requires registration (SEBI Research Analyst in India,
SEC/state investment adviser in the US) - get legal advice first.

Test: `python -m pytest`
