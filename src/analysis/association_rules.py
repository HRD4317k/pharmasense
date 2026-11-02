import duckdb
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats as stats

DB_PATH = Path("../../data/processed/faers.db")

def main():
    os.chdir(Path(__file__).parent)
    con = duckdb.connect(str(DB_PATH))
    
    print("Computing Hybrid Apriori (PRR & Chi-Square) for Drug Pairs -> AE...")
    
    # Total reports in database
    N = con.execute("SELECT COUNT(DISTINCT caseid) FROM demo").fetchone()[0]
    if N == 0:
        print("No data found.")
        return
        
    print(f"Total reports (N): {N}")
    
    # 1. Find frequent drug pairs (Support >= 20)
    print("Finding frequent drug pairs...")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE frequent_pairs AS
        SELECT d1.drugname AS drug1, d2.drugname AS drug2, COUNT(DISTINCT d1.caseid) as pair_support
        FROM clean_drug d1
        JOIN clean_drug d2 ON d1.caseid = d2.caseid AND d1.drugname < d2.drugname
        GROUP BY d1.drugname, d2.drugname
        HAVING COUNT(DISTINCT d1.caseid) >= 20
    """)
    pair_count = con.execute("SELECT COUNT(*) FROM frequent_pairs").fetchone()[0]
    print(f"Found {pair_count} frequent drug pairs.")
    
    # 2. Join with reactions to find (Drug1, Drug2, AE) triples
    print("Finding associations with Adverse Events...")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE triple_counts AS
        SELECT 
            p.drug1, 
            p.drug2, 
            r.reaction as ae,
            p.pair_support,
            COUNT(DISTINCT r.caseid) as a
        FROM frequent_pairs p
        JOIN clean_drug d1 ON d1.drugname = p.drug1
        JOIN clean_drug d2 ON d2.drugname = p.drug2 AND d1.caseid = d2.caseid
        JOIN clean_reac r ON d1.caseid = r.caseid
        GROUP BY p.drug1, p.drug2, r.reaction, p.pair_support
        HAVING COUNT(DISTINCT r.caseid) >= 5
    """)
    
    # 3. Get overall AE support to calculate c and d
    con.execute("""
        CREATE OR REPLACE TEMP TABLE ae_counts AS
        SELECT reaction as ae, COUNT(DISTINCT caseid) as ae_support
        FROM clean_reac
        GROUP BY reaction
    """)
    
    # 4. Calculate PRR, 95% CI, and Chi-Square
    print("Calculating PRR and Chi-Square...")
    
    query = f"""
        SELECT 
            t.drug1,
            t.drug2,
            t.ae,
            t.a,
            (t.pair_support - t.a) as b,
            (ae.ae_support - t.a) as c,
            ({N} - t.pair_support - ae.ae_support + t.a) as d,
            ae.ae_support
        FROM triple_counts t
        JOIN ae_counts ae ON t.ae = ae.ae
    """
    df = con.execute(query).fetchdf()
    
    if len(df) == 0:
        print("No associations found.")
        return
        
    # Prevent division by zero
    df['b'] = df['b'].replace(0, 0.5)
    df['c'] = df['c'].replace(0, 0.5)
    df['d'] = df['d'].replace(0, 0.5)
    
    # PRR = (a / (a+b)) / (c / (c+d))
    df['prr'] = (df['a'] / (df['a'] + df['b'])) / (df['c'] / (df['c'] + df['d']))
    
    # PRR 95% CI lower bound (PRR05)
    # se = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    se = np.sqrt(1/df['a'] - 1/(df['a']+df['b']) + 1/df['c'] - 1/(df['c']+df['d']))
    df['prr_05'] = np.exp(np.log(df['prr']) - 1.96 * se)
    
    # Chi-square with Yates correction
    # X2 = (|ad - bc| - N/2)^2 * N / ((a+b)(c+d)(a+c)(b+d))
    n = df['a'] + df['b'] + df['c'] + df['d']
    num = (np.abs(df['a'] * df['d'] - df['b'] * df['c']) - n/2)**2 * n
    den = (df['a'] + df['b']) * (df['c'] + df['d']) * (df['a'] + df['c']) * (df['b'] + df['d'])
    df['chi_square'] = num / den
    
    # 5. Filter significant signals based on thresholds
    # support >= 20 is already applied to drug pairs, but let's apply to 'a' if needed. The paper says min_support >= 20 for the pattern
    significant = df[(df['a'] >= 20) & (df['prr_05'] >= 2.0) & (df['chi_square'] >= 4.0)]
    
    print(f"Extracted {len(significant)} significant DIAE patterns.")
    
    # Save to database
    con.register("significant_df", significant)
    con.execute("CREATE OR REPLACE TABLE diae_signals AS SELECT * FROM significant_df")
    print("Results saved to 'diae_signals' table.")
    con.close()

if __name__ == "__main__":
    main()
