import duckdb
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "faers.db"

def get_db_connection():
    # Streamlit runs from the root usually, but this ensures absolute path based on this script
    return duckdb.connect(str(DB_PATH), read_only=True)

def get_overall_stats():
    con = get_db_connection()
    try:
        total_reports = con.execute("SELECT count(DISTINCT caseid) FROM demo").fetchone()[0]
        total_drugs = con.execute("SELECT count(DISTINCT drugname) FROM clean_drug").fetchone()[0]
        total_reactions = con.execute("SELECT count(DISTINCT reaction) FROM clean_reac").fetchone()[0]
        
        return {
            "Total Reports": f"{total_reports:,}",
            "Unique Drugs": f"{total_drugs:,}",
            "Unique Reactions": f"{total_reactions:,}"
        }
    except Exception as e:
        return {"Error": str(e)}
    finally:
        con.close()

def search_drug_stats(drug_name: str):
    con = get_db_connection()
    drug_name = drug_name.upper()
    try:
        # Number of reports for this drug
        like = f"%{drug_name}%"
        reports = con.execute("SELECT count(DISTINCT caseid) FROM clean_drug WHERE drugname LIKE ?", [like]).fetchone()[0]
        
        if reports == 0:
            return None
            
        # Top adverse events
        q_adrs = """
            SELECT r.reaction, count(DISTINCT r.caseid) as report_count
            FROM clean_reac r
            JOIN clean_drug d ON r.caseid = d.caseid
            WHERE d.drugname LIKE ?
            GROUP BY r.reaction
            ORDER BY report_count DESC
            LIMIT 10
        """
        top_adrs = con.execute(q_adrs, [like]).fetchdf()
        
        return {
            "reports_count": reports,
            "top_adrs": top_adrs
        }
    except Exception as e:
        print(e)
        return None
    finally:
        con.close()
