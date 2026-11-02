"""Clean the fetched reports and mine drug / 3-drug signals.

  python process_and_mine.py                      defaults
  python process_and_mine.py --drug-scope suspect only drugs flagged suspect/interacting
  python process_and_mine.py --min-triplet 15     lower the support threshold (more triplets)

Outputs (data/results/): triplet_results.parquet (all tested), triplet_signals.csv (signals only,
with plain-language justification), drug_reaction_stats.parquet, drug_month.parquet,
excluded_terms.parquet, reports_clean.parquet, meta.json.   Read by app.py.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

import pharmasense_core as core

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "structured_fda_data.parquet"
OUT = ROOT / "data" / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug-scope", choices=["all", "suspect"], default="all",
                    help="all listed drugs (default) or only suspect/interacting ones")
    ap.add_argument("--min-triplet", type=int, default=None,
                    help="min reports containing all 3 drugs (default: max(10, 0.1%% of reports))")
    ap.add_argument("--min-events", type=int, default=3, help="min reports with the 3 drugs AND the reaction")
    ap.add_argument("--max-drugs", type=int, default=300, help="only the N most frequent drugs enter FP-Growth")
    ap.add_argument("--max-triplets", type=int, default=20000)
    args = ap.parse_args()

    if not RAW.exists():
        print("Data file not found. Run fetch_fda_api.py first.")
        sys.exit(1)
    raw = pd.read_parquet(RAW)
    try:
        reports, excluded, info = core.prepare_reports(raw, args.drug_scope)
    except ValueError as e:
        print(e)
        sys.exit(1)
    print(f"Raw reports: {info['raw_reports']:,} -> usable: {info['kept_reports']:,}")
    print(f"  dropped (fewer than 3 distinct drugs after name clean-up): {info['dropped_fewer_than_3_distinct_drugs']:,}")
    print(f"  dropped (no genuine reaction left): {info['dropped_no_genuine_reaction']:,}")
    print(f"  reaction mentions excluded as non-adverse-events/indications: {info['excluded_reaction_occurrences']:,}")
    if len(excluded):
        print("  most frequent excluded terms:", "; ".join(f"{r.reaction.title()} ({r.occurrences})"
                                                         for r in excluded.head(5).itertuples()))
    if len(reports) < 100:
        print("Too few usable reports to mine - fetch more data.")
        sys.exit(1)

    result = core.run_mining(reports, args.min_triplet, args.min_events, args.max_drugs, args.max_triplets)
    core.save_results(result, reports, excluded, info, OUT)

    m = result["meta"]
    print(f"\nTested {m['n_tested']:,} (triplet, reaction) combinations.")
    if m["n_tested"]:
        print(f"  median ROR over everything tested: {m['median_ror_all_tested']:.2f}   "
              f"(share >10: {100 * m['share_ror_over_10']:.1f}%, >100: {100 * m['share_ror_over_100']:.2f}%)")
        print("  verdicts:", m["verdict_counts"])
    else:
        print("  nothing reached the minimum support - fetch more data or lower --min-triplet / --min-events.")
    print(f"Results written to {OUT.relative_to(ROOT)}/  ->  streamlit run app.py")


if __name__ == "__main__":
    main()
