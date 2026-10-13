"""Train per-symbol models for the US and Indian markets.

    python train.py AAPL
    python train.py RELIANCE.NS TCS HDFCBANK        # bare Nifty names get .NS automatically
    python train.py --preset nifty                  # all bundled Nifty 50 names
    python train.py INFY --market us                # the US ADR instead of the NSE listing
"""
import argparse
import sys

from stockai.markets import NIFTY, US_NAMES
from stockai.training import train_symbol

PRESETS = {"nifty": [f"{s}.NS" for s in NIFTY], "us": [s for s in US_NAMES if not s.startswith("^")]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--market", default="auto", choices=["auto", "in", "us"])
    ap.add_argument("--years", type=int, default=10, help="years of daily price history (default 10)")
    ap.add_argument("--news-years", type=int, default=4, help="years of headlines to train on (default 4)")
    ap.add_argument("--no-news", action="store_true", help="technical features only (skip the semantic layer)")
    ap.add_argument("--scorer", choices=["auto", "finbert", "vader"], default="auto", help="headline sentiment model")
    args = ap.parse_args()

    symbols = list(args.symbols) + (PRESETS[args.preset] if args.preset else [])
    if not symbols:
        ap.error("give at least one symbol or --preset")

    failed = 0
    for raw in symbols:
        try:
            symbol, meta, path = train_symbol(raw, args.market, args.years, news=not args.no_news,
                                           news_years=args.news_years, scorer=args.scorer)
        except Exception as exc:  # keep going through a batch
            failed += 1
            print(f"FAIL  {raw}: {exc}")
            continue
        m, b = meta["metrics"], meta["backtest"]
        verdict = "edge" if meta["has_edge"] else "no reliable edge"
        print(f"OK    {symbol:14s} holdout {meta['holdout_start']}..{meta['holdout_end']}  "
              f"AUC {m['auc']:.3f} (z={m['auc_z']:+.1f}, cv {m['cv_auc_mean']:.3f})  "
              f"acc {m['accuracy']:.1%} vs {m['baseline_accuracy']:.1%} baseline  "
              f"strategy {b['strategy_return']:+.1%} vs buy&hold {b['buy_hold_return']:+.1%}  [{verdict}]")
        d = meta["dip"]
        h = d["holdout"]
        print(f"      dip risk (>= {abs(d['threshold']):.1%} drop within {d['horizon_days']}d, base rate {h['base_rate']:.0%}): "
              f"holdout AUC {h['auc']:.3f} (z={h['auc_z']:+.1f}) vs volatility-only {h['volatility_only_auc']:.3f}; "
              f"top-quintile dip rate {h['dip_rate_when_top_quintile']:.0%} ({h['lift_top_quintile']:.1f}x)  "
              f"[{'skill' if d['has_skill'] else 'no reliable skill'}]")
        ab = meta["semantic"]["ablation"]
        if ab:
            print(f"      semantic ablation (holdout AUC / cv): technical {ab['technical_only']['auc']:.3f}/{ab['technical_only']['cv_auc']:.3f}"
                  f"  semantic-only {ab['semantic_only']['auc']:.3f}/{ab['semantic_only']['cv_auc']:.3f}"
                  f"  combined {ab['combined']['auc']:.3f}/{ab['combined']['cv_auc']:.3f}"
                  f"  -> {'semantics help' if ab['semantic_adds'] else 'semantics do not help here'}"
                  f"  ({meta['semantic']['headlines']} headlines, {meta['semantic']['scorer']})")
    return 1 if failed == len(symbols) else 0


if __name__ == "__main__":
    sys.exit(main())
