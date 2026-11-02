"""Load data/results/reports_clean.parquet (written by process_and_mine.py) into
data/processed/faers.db with the tables the src/analysis modules expect: demo, clean_drug, clean_reac."""
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent
PARQUET = ROOT / "data" / "results" / "reports_clean.parquet"
DB_PATH = ROOT / "data" / "processed" / "faers.db"


def main():
    if not PARQUET.exists():
        print("Run process_and_mine.py first (data/results/reports_clean.parquet is missing).")
        sys.exit(1)
    df = pd.read_parquet(PARQUET)
    demo = (df[["report_id", "date", "patient_sex", "patient_age"]]
            .rename(columns={"report_id": "caseid", "date": "receive_date"}).astype(str))
    drugs = (df[["report_id", "drugs"]].explode("drugs").rename(columns={"report_id": "caseid", "drugs": "drugname"})
             .dropna().drop_duplicates())
    reacs = (df[["report_id", "reactions"]].explode("reactions").rename(columns={"report_id": "caseid", "reactions": "reaction"})
             .dropna().drop_duplicates())
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    for name, frame in (("demo", demo), ("clean_drug", drugs), ("clean_reac", reacs)):
        con.register("tmp_frame", frame)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp_frame")
        con.unregister("tmp_frame")
        print(f"{name}: {len(frame):,} rows")
    con.close()
    print(f"Wrote {DB_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
