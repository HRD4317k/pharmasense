import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import os

DB_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "faers.db"

def get_db():
    return duckdb.connect(str(DB_PATH), read_only=True)

def calculate_ror(a, b, c, d):
    # Haldane-Anscombe: add 0.5 to every cell ONLY if some cell is zero
    if min(a, b, c, d) == 0:
        a, b, c, d = a+0.5, b+0.5, c+0.5, d+0.5
    ror = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_low = np.exp(np.log(ror) - 1.96 * se)
    ci_high = np.exp(np.log(ror) + 1.96 * se)
    return round(ror, 4), round(ci_low, 4), round(ci_high, 4)

def calculate_prr(a, b, c, d):
    a0, b0, c0, d0 = a, b, c, d
    if a == 0 or c == 0:
        a, b, c, d = a+0.5, b+0.5, c+0.5, d+0.5
    prr = (a/(a+b)) / (c/(c+d))
    se = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    ci_low = np.exp(np.log(prr) - 1.96 * se)
    ci_high = np.exp(np.log(prr) + 1.96 * se)
    # Chi-square
    # Yates-corrected chi-square on the raw (uncorrected) counts
    n = a0+b0+c0+d0
    num = max(abs(a0*d0 - b0*c0) - n/2, 0)**2 * n
    den = (a0+b0)*(c0+d0)*(a0+c0)*(b0+d0)
    chi2 = num/den if den > 0 else 0.0
    return round(prr, 4), round(ci_low, 4), round(ci_high, 4), round(chi2, 4)

def get_drug_ae_signals(drug_name: str, top_n: int = 20) -> pd.DataFrame:
    """Compute ROR and PRR signals for a specific drug vs all AEs."""
    con = get_db()
    drug_name = drug_name.upper()
    
    total = con.execute("SELECT COUNT(DISTINCT caseid) FROM demo").fetchone()[0]
    
    # Reports containing the drug
    n_drug = con.execute(
        "SELECT COUNT(DISTINCT caseid) FROM clean_drug WHERE drugname LIKE ?",
        (f"%{drug_name}%",)
    ).fetchone()[0]
    
    if n_drug == 0:
        con.close()
        return pd.DataFrame()
    
    # Per-AE counts with the drug
    df = con.execute(f"""
        SELECT r.reaction,
               COUNT(DISTINCT r.caseid) as a,
               (SELECT COUNT(DISTINCT caseid) FROM clean_reac WHERE reaction = r.reaction) as n_ae
        FROM clean_reac r
        JOIN clean_drug d ON r.caseid = d.caseid
        WHERE d.drugname LIKE ?
        GROUP BY r.reaction
        HAVING COUNT(DISTINCT r.caseid) >= 3
        ORDER BY COUNT(DISTINCT r.caseid) DESC
        LIMIT {int(top_n)}
    """, (f"%{drug_name}%",)).fetchdf()
    
    con.close()
    
    results = []
    for _, row in df.iterrows():
        a = row['a']
        n_ae = row['n_ae']
        b = n_drug - a
        c = n_ae - a
        d = total - n_drug - n_ae + a
        
        ror, ror_l, ror_h = calculate_ror(a, b, c, d)
        prr, prr_l, prr_h, chi2 = calculate_prr(a, b, c, d)
        
        results.append({
            'Adverse Event': row['reaction'],
            'Reports': int(a),
            'ROR': ror,
            'ROR 95% CI': f"[{ror_l}, {ror_h}]",
            'PRR': prr,
            'PRR 95% CI': f"[{prr_l}, {prr_h}]",
            'Chi2': chi2,
            'Signal': 'HIGH' if (ror_l > 2 and chi2 >= 4) else ('MODERATE' if ror_l > 1 else 'LOW')
        })
    
    return pd.DataFrame(results)

def get_top_drug_pairs(top_n: int = 30) -> pd.DataFrame:
    """Get most frequently co-reported drug pairs."""
    con = get_db()
    df = con.execute(f"""
        SELECT d1.drugname as drug1, d2.drugname as drug2,
               COUNT(DISTINCT d1.caseid) as co_reports
        FROM clean_drug d1
        JOIN clean_drug d2 ON d1.caseid = d2.caseid AND d1.drugname < d2.drugname
        GROUP BY d1.drugname, d2.drugname
        HAVING COUNT(DISTINCT d1.caseid) >= 10
        ORDER BY co_reports DESC
        LIMIT {int(top_n)}
    """).fetchdf()
    con.close()
    return df

def get_temporal_trends(drug_name: str) -> pd.DataFrame:
    """Get quarterly report counts for a drug."""
    con = get_db()
    drug_name = drug_name.upper()
    df = con.execute(f"""
        SELECT SUBSTR(receive_date, 1, 4) as year,
               COUNT(DISTINCT d.caseid) as reports
        FROM clean_drug d
        JOIN demo dem ON d.caseid = dem.caseid
        WHERE d.drugname LIKE ?
          AND receive_date IS NOT NULL
        GROUP BY SUBSTR(receive_date, 1, 4)
        ORDER BY year
    """, (f"%{drug_name}%",)).fetchdf()
    con.close()
    return df

def get_ae_heatmap_data(top_drugs: int = 15, top_aes: int = 15) -> pd.DataFrame:
    """Returns pivot table of drug vs AE for heatmap."""
    con = get_db()
    drugs = con.execute(f"""
        SELECT drugname FROM clean_drug
        GROUP BY drugname ORDER BY COUNT(DISTINCT caseid) DESC LIMIT {top_drugs}
    """).fetchdf()['drugname'].tolist()
    
    aes = con.execute(f"""
        SELECT reaction FROM clean_reac
        GROUP BY reaction ORDER BY COUNT(DISTINCT caseid) DESC LIMIT {top_aes}
    """).fetchdf()['reaction'].tolist()
    
    if not drugs or not aes:
        con.close()
        return pd.DataFrame()
    dph = ",".join("?" * len(drugs))
    aph = ",".join("?" * len(aes))
    df = con.execute(f"""
        SELECT d.drugname, r.reaction, COUNT(DISTINCT d.caseid) as reports
        FROM clean_drug d
        JOIN clean_reac r ON d.caseid = r.caseid
        WHERE d.drugname IN ({dph})
          AND r.reaction IN ({aph})
        GROUP BY d.drugname, r.reaction
    """, drugs + aes).fetchdf()
    con.close()
    
    pivot = df.pivot(index='drugname', columns='reaction', values='reports').fillna(0)
    return pivot
