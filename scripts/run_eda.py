"""
Phase 1 entrypoint: scan the raw dump, build pairs, profile brands, pick one.

    python scripts/run_eda.py                     # real data (data/raw/twcs.csv)
    python scripts/run_eda.py --fixture           # synthetic fixture, no download
    python scripts/run_eda.py --max-rows 400000   # faster scan

Writes:
    data/sample/pairs_<brand>.parquet   the working subsample for the chosen brand
    data/sample/brand_profiles.csv      the full selection table
    report/figures/brand_selection.png  the chart that justifies the pick
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.eda.brand_select import plot_selection, profile_brands, to_markdown
from src.ingest.threads import build_pairs, build_pairs_streaming, load_raw


def main() -> int:
    ap = argparse.ArgumentParser(description="Brand selection EDA")
    ap.add_argument("--fixture", action="store_true",
                    help="use the synthetic test fixture instead of the Kaggle dump")
    ap.add_argument("--max-rows", type=int, default=config.EDA_SCAN_ROWS,
                    help="cap rows scanned from the raw CSV")
    ap.add_argument("--min-pairs", type=int, default=200,
                    help="ignore brands with fewer pairs than this")
    ap.add_argument("--brand", type=str, default=None,
                    help="force a brand instead of taking the top-scoring one")
    ap.add_argument("--top-n", type=int, default=15)
    args = ap.parse_args()

    # ---------------- load ----------------
    if args.fixture:
        from tests.make_fixture import FIXTURE, generate
        if not FIXTURE.exists():
            generate()
        print(f"[eda] fixture mode -> {FIXTURE.name}")
        pairs = build_pairs(load_raw(FIXTURE))
        min_pairs = 5
    else:
        if not config.RAW_CSV.exists():
            print(
                f"\n  Raw dataset not found: {config.RAW_CSV}\n\n"
                "  Get it from Kaggle: thoughtvector/customer-support-on-twitter\n"
                "  Put twcs.csv in data/raw/ , or run with --fixture to smoke-test "
                "the pipeline.\n",
                file=sys.stderr,
            )
            return 1
        print(f"[eda] scanning up to {args.max_rows:,} rows of {config.RAW_CSV.name} ...")
        pairs = build_pairs_streaming(max_rows=args.max_rows)
        min_pairs = args.min_pairs

    print(f"[eda] reconstructed {len(pairs):,} (customer -> brand reply) pairs "
          f"across {pairs['brand'].nunique():,} brands")
    if pairs.empty:
        print("[eda] no pairs built - check the input file schema", file=sys.stderr)
        return 1

    # ---------------- profile ----------------
    profiles = profile_brands(pairs, min_pairs=min_pairs)
    if profiles.empty:
        print(f"[eda] no brand cleared min_pairs={min_pairs}", file=sys.stderr)
        return 1

    out_csv = config.DATA_SAMPLE / "brand_profiles.csv"
    profiles.to_csv(out_csv, index=False)

    print("\n" + "=" * 78)
    print("BRAND SELECTION  —  ranked by usable grounding material")
    print("=" * 78)
    print(to_markdown(profiles, top_n=args.top_n))

    chart = plot_selection(profiles, top_n=args.top_n)
    print(f"[eda] chart -> {chart}")

    # ---------------- pick + freeze the subsample ----------------
    chosen = args.brand or profiles.iloc[0]["brand"]
    row = profiles[profiles["brand"] == chosen]
    if row.empty:
        print(f"[eda] brand '{chosen}' not in profile table", file=sys.stderr)
        return 1
    row = row.iloc[0]

    sub = pairs[pairs["brand"] == chosen].copy()
    if len(sub) > config.SAMPLE_PAIRS_PER_BRAND:
        sub = sub.sample(config.SAMPLE_PAIRS_PER_BRAND, random_state=config.SEED)
    sub = sub.reset_index(drop=True)

    out_parquet = config.DATA_SAMPLE / f"pairs_{chosen}.parquet"
    try:
        sub.to_parquet(out_parquet, index=False)
    except Exception:  # pyarrow missing -> csv fallback keeps the pipeline runnable
        out_parquet = out_parquet.with_suffix(".csv")
        sub.to_csv(out_parquet, index=False)

    print("\n" + "-" * 78)
    print(f"CHOSEN BRAND: {chosen}")
    print(f"  pairs kept          {len(sub):,}")
    print(f"  resolution rate     {row['resolution_rate']:.1%}"
          f"   (deflection {row['deflection_rate']:.1%})")
    print(f"  specificity         {row['specificity_rate']:.1%}")
    print(f"  gratitude in f/u    {row['gratitude_rate']:.1%}")
    print(f"  subsample           {out_parquet}")
    print("-" * 78)
    print("\nnext: define the intent taxonomy from this subsample "
          "(python scripts/build_taxonomy.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
