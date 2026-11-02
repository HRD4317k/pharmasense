"""Multi-gate 3-drug polypharmacy signal miner (FAERS / openFDA data in data/processed/faers.db).

Gates: 1-2 support | 3 ROR lower CI > 1 | 4 triplet beats every pair and single |
       5 three-way logistic interaction, Benjamini-Hochberg FDR < 0.05 and beta > 0.
Writes tables `polypharmacy_candidates` (everything that reached Gate 2, with the gate columns
filled as far as each row got) and `polypharmacy_signals` (rows that passed all gates).
"""
import argparse
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

DB_PATH = Path("../../data/processed/faers.db")


def sq(s):
    """Escape a value for use inside a single-quoted SQL string."""
    return str(s).replace("'", "''")


def calculate_ror(a, b, c, d):
    # Haldane-Anscombe correction: add 0.5 to every cell to avoid division by zero
    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    ror = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return ror, np.exp(np.log(ror) - 1.96 * se), np.exp(np.log(ror) + 1.96 * se)


def lower_order_rors(con, drugs, ae, n_ae, total):
    """ROR of every single drug and every pair inside the triplet, for the same adverse event."""
    def subset_ror(subset):
        has_drugs = " AND ".join(
            f"EXISTS (SELECT 1 FROM clean_drug dg WHERE dg.caseid = d.caseid AND dg.drugname = '{sq(x)}')"
            for x in subset)
        ab = con.execute(f"SELECT COUNT(*) FROM demo d WHERE {has_drugs}").fetchone()[0]
        a = con.execute(
            f"SELECT COUNT(*) FROM demo d WHERE {has_drugs} AND EXISTS "
            f"(SELECT 1 FROM clean_reac r WHERE r.caseid = d.caseid AND r.reaction = ?)", (ae,)).fetchone()[0]
        return calculate_ror(a, ab - a, n_ae - a, total - ab - n_ae + a)[0]

    d1, d2, d3 = drugs
    singles = [subset_ror([x]) for x in (d1, d2, d3)]
    pairs = [subset_ror(p) for p in ([d1, d2], [d1, d3], [d2, d3])]
    return max(singles), max(pairs)


def test_3way_interaction(con, d1, d2, d3, ae):
    query = f"""
        SELECT d.caseid,
               MAX(CASE WHEN dg.drugname = '{sq(d1)}' THEN 1 ELSE 0 END) AS d1,
               MAX(CASE WHEN dg.drugname = '{sq(d2)}' THEN 1 ELSE 0 END) AS d2,
               MAX(CASE WHEN dg.drugname = '{sq(d3)}' THEN 1 ELSE 0 END) AS d3,
               MAX(CASE WHEN r.reaction  = '{sq(ae)}' THEN 1 ELSE 0 END) AS y
        FROM demo d
        LEFT JOIN clean_drug dg ON d.caseid = dg.caseid AND dg.drugname IN ('{sq(d1)}', '{sq(d2)}', '{sq(d3)}')
        LEFT JOIN clean_reac r  ON d.caseid = r.caseid  AND r.reaction = '{sq(ae)}'
        GROUP BY d.caseid
    """
    df = con.execute(query).fetchdf()
    if df["y"].sum() < 5 or df["d1"].sum() == 0 or df["d2"].sum() == 0 or df["d3"].sum() == 0:
        return None, None
    df["d1_d2"] = df["d1"] * df["d2"]
    df["d1_d3"] = df["d1"] * df["d3"]
    df["d2_d3"] = df["d2"] * df["d3"]
    df["d1_d2_d3"] = df["d1"] * df["d2"] * df["d3"]
    try:
        model = smf.logit("y ~ d1 + d2 + d3 + d1_d2 + d1_d3 + d2_d3 + d1_d2_d3", data=df).fit(disp=0)
        return float(model.params.get("d1_d2_d3", 0.0)), float(model.pvalues.get("d1_d2_d3", 1.0))
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-drug-reports", type=int, default=100, help="a drug must appear in this many reports to enter FP-Growth")
    ap.add_argument("--min-triplet", type=int, default=20, help="min reports containing all 3 drugs (MIN_SUPPORT_ABS)")
    ap.add_argument("--min-ae", type=int, default=10, help="min reports with all 3 drugs + the AE (MIN_SUPPORT_AE)")
    ap.add_argument("--top-triplets", type=int, default=50, help="how many of the most frequent triplets to analyse")
    args = ap.parse_args()

    os.chdir(Path(__file__).parent)
    con = duckdb.connect(str(DB_PATH))

    total = con.execute("SELECT COUNT(DISTINCT caseid) FROM demo").fetchone()[0]
    if total == 0:
        print("Database is empty. Run fetch_fda_api.py and load_to_duckdb.py first.")
        return
    print(f"Total reports: {total}")

    freq_drugs = con.execute(
        "SELECT drugname FROM clean_drug GROUP BY drugname HAVING COUNT(DISTINCT caseid) > ?",
        (args.min_drug_reports,)).fetchdf()["drugname"].tolist()
    if len(freq_drugs) < 3:
        print(f"Only {len(freq_drugs)} drugs appear in more than {args.min_drug_reports} reports. "
              f"Fetch more data or lower --min-drug-reports.")
        return

    con.register("freq_drugs_df", pd.DataFrame({"drugname": freq_drugs}))
    tx_df = con.execute("""
        SELECT caseid, list(drugname) AS drugs FROM clean_drug
        WHERE drugname IN (SELECT drugname FROM freq_drugs_df)
        GROUP BY caseid HAVING COUNT(drugname) >= 3""").fetchdf()
    transactions = [list(x) for x in tx_df["drugs"]]
    print(f"FP-Growth over {len(transactions)} reports and {len(freq_drugs)} frequent drugs...")

    te = TransactionEncoder()
    sparse = te.fit(transactions).transform(transactions, sparse=True)
    df_sparse = pd.DataFrame.sparse.from_spmatrix(sparse, columns=te.columns_)
    min_sup = args.min_triplet / len(transactions)
    try:
        itemsets = fpgrowth(df_sparse, min_support=min_sup, use_colnames=True, max_len=3)
    except TypeError:
        itemsets = fpgrowth(df_sparse, min_support=min_sup, use_colnames=True)
    triplets = itemsets[itemsets["itemsets"].apply(len) == 3].copy()
    print(f"Found {len(triplets)} frequent 3-drug triplets.")
    triplets = triplets.sort_values("support", ascending=False).head(args.top_triplets)

    ae_totals = {}   # cache: reaction -> number of reports
    rows = []
    for _, trip in tqdm(triplets.iterrows(), total=len(triplets), desc="Processing triplets"):
        d1, d2, d3 = sorted(trip["itemsets"])
        join3 = (f"JOIN clean_drug g1 ON {{t}}.caseid = g1.caseid AND g1.drugname = '{sq(d1)}' "
                 f"JOIN clean_drug g2 ON {{t}}.caseid = g2.caseid AND g2.drugname = '{sq(d2)}' "
                 f"JOIN clean_drug g3 ON {{t}}.caseid = g3.caseid AND g3.drugname = '{sq(d3)}'")
        ab = con.execute(f"SELECT COUNT(DISTINCT d.caseid) FROM demo d {join3.format(t='d')}").fetchone()[0]
        aes = con.execute(f"""
            SELECT r.reaction, COUNT(DISTINCT r.caseid) AS a FROM clean_reac r {join3.format(t='r')}
            GROUP BY r.reaction HAVING COUNT(DISTINCT r.caseid) >= {int(args.min_ae)}""").fetchdf()

        for _, ae_row in aes.iterrows():
            ae, a = ae_row["reaction"], int(ae_row["a"])
            if ae not in ae_totals:
                ae_totals[ae] = con.execute(
                    "SELECT COUNT(DISTINCT caseid) FROM clean_reac WHERE reaction = ?", (ae,)).fetchone()[0]
            n_ae = ae_totals[ae]
            ror, lcl, _ = calculate_ror(a, ab - a, n_ae - a, total - ab - n_ae + a)
            row = {"Drug 1": d1, "Drug 2": d2, "Drug 3": d3, "Adverse Event": ae, "a": a,
                   "ROR_ABC": ror, "ROR_ABC_Lower_CI": lcl,
                   "Max_Single_ROR": np.nan, "Max_Pair_ROR": np.nan, "Max_Lower_Order_ROR": np.nan,
                   "Interaction_Strength": np.nan, "Beta_ABC": np.nan, "P_Value": np.nan}
            rows.append(row)

            if ror <= 1 or lcl <= 1:                                   # Gate 3
                continue
            max_single, max_pair = lower_order_rors(con, (d1, d2, d3), ae, n_ae, total)
            max_lower = max(max_single, max_pair)
            row.update(Max_Single_ROR=max_single, Max_Pair_ROR=max_pair, Max_Lower_Order_ROR=max_lower,
                       Interaction_Strength=np.log(ror) - np.log(max_lower))
            if row["Interaction_Strength"] <= 0:                       # Gate 4
                continue
            beta, p = test_3way_interaction(con, d1, d2, d3, ae)       # Gate 5 (regression part)
            if beta is not None:
                row.update(Beta_ABC=beta, P_Value=p)

    if not rows:
        print("No candidate (triplet, adverse event) pairs. Fetch more data or lower --min-triplet / --min-ae.")
        con.close()
        return

    cand = pd.DataFrame(rows)
    cand["Q_Value"] = np.nan
    tested = cand["P_Value"].notna()
    if tested.any():                                                   # Gate 5 (FDR part)
        cand.loc[tested, "Q_Value"] = multipletests(cand.loc[tested, "P_Value"], alpha=0.05, method="fdr_bh")[1]

    final = cand[(cand["Q_Value"] < 0.05) & (cand["Beta_ABC"] > 0)].copy()
    final["Emerging_Signal_Score"] = final["Interaction_Strength"] * final["ROR_ABC_Lower_CI"]
    final = final.sort_values("Emerging_Signal_Score", ascending=False)

    print(f"\nCandidates: {len(cand)} | Gate 3: {(cand['ROR_ABC_Lower_CI'] > 1).sum()} | "
          f"Gate 4: {(cand['Interaction_Strength'] > 0).sum()} | Gate 5 (final): {len(final)}")
    if len(final):
        print("\n--- FINAL 3-DRUG POLYPHARMACY SIGNALS ---")
        print(final.to_string(index=False))

    con.register("cand_df", cand)
    con.register("final_df", final)
    con.execute("CREATE OR REPLACE TABLE polypharmacy_candidates AS SELECT * FROM cand_df")
    con.execute("CREATE OR REPLACE TABLE polypharmacy_signals AS SELECT * FROM final_df")
    print("Saved tables 'polypharmacy_candidates' and 'polypharmacy_signals'.")
    con.close()


if __name__ == "__main__":
    main()
