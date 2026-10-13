"""Train the pooled dip-risk model for a market.

    python train_dip.py --market us
    python train_dip.py --market in --news      # include headline sentiment where cached/scraped
"""
import argparse
import sys

from stockai.markets import IN, NIFTY, US, US_LARGE
from stockai.pooled import K, train_pooled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", choices=["in", "us"], required=True)
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--news", action="store_true")
    args = ap.parse_args()
    mk = IN if args.market == "in" else US
    symbols = [f"{s}.NS" for s in NIFTY] if mk is IN else US_LARGE
    meta, path = train_pooled(symbols, mk, args.years, news=args.news)
    h = meta["holdout"]
    print(f"{mk.code}: {meta['names']} stocks, dip = drop worse than {K} sigma over {meta['horizon_days']} days "
          f"(base rate {h['base_rate']:.0%}), holdout {h['start']}..{h['end']}")
    print(f"  holdout AUC {h['auc']:.3f}  90% CI [{h['auc_ci90'][0]:.3f}, {h['auc_ci90'][1]:.3f}]  "
          f"(volatility-regime alone {h['volatility_regime_only_auc']:.3f})")
    print(f"  riskiest 20% of days dipped {h['dip_rate_when_top_quintile']:.0%} ({h['lift_top_quintile']:.2f}x base); "
          f"safest 20% dipped {h['dip_rate_when_bottom_quintile']:.0%}")
    print(f"  top features: {', '.join(meta['top_features'])}")
    print(f"  => {'SKILL (CI above 0.5)' if meta['has_skill'] else 'no statistically reliable skill'}; saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
