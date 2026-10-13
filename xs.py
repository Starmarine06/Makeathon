"""Cross-sectional ranking model: evaluate honestly out-of-sample.

    python xs.py --market in            # Nifty 50
    python xs.py --market us            # 50 US large caps
    python xs.py --market in --news     # also use headline sentiment (slow the first time: scrapes + FinBERT)
"""
import argparse
import json
import sys

from stockai import xsection as xs
from stockai.markets import IN, NIFTY, US, US_LARGE


def row(label, r):
    n, vt = r["net"], r.get("vol_targeted")
    print(f"  {label:<22} ann {n['ann_return']:+7.1%}  vol {n['ann_vol']:5.1%}  Sharpe {n['sharpe']:+5.2f}  "
          f"maxDD {n['max_drawdown']:6.1%}  turnover {r['turnover_per_rebalance']:4.0%}/rebal"
          + (f"  | vol-targeted: Sharpe {vt['sharpe']:+5.2f} ann {vt['ann_return']:+6.1%}" if vt else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", choices=["in", "us"], required=True)
    ap.add_argument("--k", type=int, default=10, help="stocks held (default 10)")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--news", action="store_true")
    args = ap.parse_args()
    mk = IN if args.market == "in" else US
    symbols = [f"{s}.NS" for s in NIFTY] if mk is IN else US_LARGE

    print(f"Loading {len(symbols)} {mk.code} stocks...")
    panel, cols = xs.load_panel(symbols, mk, args.years, news=args.news)
    print(f"Panel: {len(panel):,} stock-days, {len(cols)} features. Walk-forward evaluation...")
    res = xs.evaluate(panel, cols, mk, args.k)
    print(f"\nOut-of-sample {res['oos_start']} -> {res['oos_end']}, top-{args.k} of {res['names']}, "
          f"hold {xs.H}d, {mk.cost_bps:g} bps/side costs")
    row("ML ranking model", res["model"])
    row("12-1 momentum only", res["momentum_baseline"])
    b = res["model"]["benchmark"]
    print(f"  {'equal-weight universe':<22} ann {b['ann_return']:+7.1%}  vol {b['ann_vol']:5.1%}  "
          f"Sharpe {b['sharpe']:+5.2f}  maxDD {b['max_drawdown']:6.1%}")
    for lbl, key in (("model", "model"), ("momentum", "momentum_baseline")):
        ic = res[key]["ic"]
        print(f"  {lbl}: rank-IC {ic['mean_ic']:+.3f} (t={ic['ic_t']:+.1f}, {ic['pct_positive']:.0%} of {ic['n']} weeks positive); "
              f"excess-vs-universe t={res[key]['excess_per_period_t']:+.1f}")
    with open(f"xs_{args.market}{'_news' if args.news else ''}_report.json", "w") as fh:
        json.dump(res, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
