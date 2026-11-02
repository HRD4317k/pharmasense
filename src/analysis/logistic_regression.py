import duckdb
from pathlib import Path
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DB_PATH = Path("../../data/processed/faers.db")


def sq(x):
    return str(x).replace("'", "''")

def test_confounding(con, drug1, drug2, ae):
    """
    Fits a logistic regression model:
    Logit(P(AE=1)) = beta0 + beta1*D1 + beta2*D2 + beta3*(D1*D2)
    """
    # 1. Create a cohort matrix for these specific drugs and AE
    # We need a boolean matrix where each row is a caseid, 
    # columns are D1_present, D2_present, AE_present.
    
    query = f"""
        SELECT 
            d.caseid,
            MAX(CASE WHEN dg.drugname = '{sq(drug1)}' THEN 1 ELSE 0 END) as d1,
            MAX(CASE WHEN dg.drugname = '{sq(drug2)}' THEN 1 ELSE 0 END) as d2,
            MAX(CASE WHEN r.reaction = '{sq(ae)}' THEN 1 ELSE 0 END) as y
        FROM demo d
        LEFT JOIN clean_drug dg ON d.caseid = dg.caseid AND dg.drugname IN ('{sq(drug1)}', '{sq(drug2)}')
        LEFT JOIN clean_reac r ON d.caseid = r.caseid AND r.reaction = '{sq(ae)}'
        GROUP BY d.caseid
    """
    
    df = con.execute(query).fetchdf()
    
    # Check if we have enough variance
    if df['y'].sum() == 0 or df['d1'].sum() == 0 or df['d2'].sum() == 0:
        return None
        
    df['d1_d2'] = df['d1'] * df['d2']
    
    # Fit Logistic Regression
    try:
        model = smf.logit("y ~ d1 + d2 + d1_d2", data=df).fit(disp=0)
        
        # d1_d2 coefficient is beta3
        beta3 = model.params['d1_d2']
        p_value = model.pvalues['d1_d2']
        exp_beta3 = np.exp(beta3) # This is exp(d) in the paper
        
        return {
            'drug1': drug1,
            'drug2': drug2,
            'ae': ae,
            'exp_d': exp_beta3,
            'p_value': p_value,
            'significant_interaction': (p_value < 0.05 and exp_beta3 > 1)
        }
    except Exception as e:
        print(f"Failed to fit LR for {drug1}, {drug2}, {ae}: {e}")
        return None

def main():
    os.chdir(Path(__file__).parent)
    con = duckdb.connect(str(DB_PATH))
    
    # Check if diae_signals table exists
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    if 'diae_signals' not in tables:
        print("Please run association_rules.py first.")
        return
        
    # Pick top 10 potential signals to test
    signals = con.execute("SELECT drug1, drug2, ae FROM diae_signals ORDER BY prr_05 DESC LIMIT 10").fetchall()
    
    results = []
    print(f"Testing confounding for {len(signals)} patterns using Logistic Regression...")
    for drug1, drug2, ae in signals:
        print(f"Testing: {drug1} + {drug2} -> {ae}")
        res = test_confounding(con, drug1, drug2, ae)
        if res:
            results.append(res)
            
    if not results:
        print("No testable patterns (diae_signals is empty - try more data).")
        con.close()
        return
    res_df = pd.DataFrame(results)
    print("\nLogistic Regression Interaction Results:")
    print(res_df.to_string(index=False))
    
    con.close()

if __name__ == "__main__":
    main()
