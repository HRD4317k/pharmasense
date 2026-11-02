"""Copy polypharmacy_candidates from faers.db into data/faers.duckdb as `triplet_signals`,
the table dashboard_v1/app.py reads (column names mapped to what the app expects)."""
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "processed" / "faers.db"
DST = ROOT / "data" / "faers.duckdb"


def main():
    src = duckdb.connect(str(SRC), read_only=True)
    try:
        df = src.execute("SELECT * FROM polypharmacy_candidates").fetchdf()
    except duckdb.Error:
        print("Table polypharmacy_candidates not found. Run polypharmacy_miner.py first.")
        return
    finally:
        src.close()
    if df.empty:
        print("polypharmacy_candidates is empty - nothing to export.")
        return
    out = pd.DataFrame({
        "drug_a": df["Drug 1"], "drug_b": df["Drug 2"], "drug_c": df["Drug 3"],
        "reaction": df["Adverse Event"], "n_reports": df["a"],
        "ror": df["ROR_ABC"], "ror_lcl": df["ROR_ABC_Lower_CI"],
        "max_pair_ror": df["Max_Pair_ROR"], "max_single_ror": df["Max_Single_ROR"],
        "interaction_coef": df["Beta_ABC"], "p_value": df["P_Value"], "q_value": df["Q_Value"]})
    dst = duckdb.connect(str(DST))
    dst.register("out_df", out)
    dst.execute("CREATE OR REPLACE TABLE triplet_signals AS SELECT * FROM out_df")
    dst.close()
    print(f"Exported {len(out)} candidate rows to {DST.relative_to(ROOT)} (table triplet_signals)")


if __name__ == "__main__":
    main()
