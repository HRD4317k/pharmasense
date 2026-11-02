"""One-command pipeline runner.

  python run_pipeline.py                 fetch -> clean + mine   (feeds app.py)
  python run_pipeline.py --full          ... plus the DuckDB analysis (PRR rules, regression) + ML models
  python run_pipeline.py --skip-fetch    reuse data/structured_fda_data.parquet
  python run_pipeline.py --max-records 5000    quicker first test run
  python run_pipeline.py --skip-fetch --calibrate    also run the permutation null (how many false alarms with NO real signal?)

After mining, a data-quality check runs automatically (is the sample spread over the period, or bunched on a few dates?).
"""
import argparse
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGES = {"pandas": "pandas", "numpy": "numpy", "duckdb": "duckdb", "mlxtend": "mlxtend",
            "sklearn": "scikit-learn", "xgboost": "xgboost", "statsmodels": "statsmodels",
            "streamlit": "streamlit", "plotly": "plotly", "pyarrow": "pyarrow",
            "dotenv": "python-dotenv", "requests": "requests", "tqdm": "tqdm"}


def preflight():
    missing = [pip for mod, pip in PACKAGES.items() if not _importable(mod)]
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Install them with:  pip install -r requirements.txt")
        sys.exit(1)


def _importable(mod):
    try:
        importlib.import_module(mod)
        return True
    except ImportError:
        return False


def run(label, args):
    print(f"\n=== {label} ===", flush=True)
    r = subprocess.run([sys.executable, *args], cwd=ROOT)
    if r.returncode != 0:
        print(f"\nStep failed: {label}")
        sys.exit(r.returncode)


def quality_report(calibrate=False):
    """Non-fatal diagnostics on the mined results. A failure here never stops the pipeline."""
    try:
        import json

        import pandas as pd

        from pharmasense_ai import cluster_check, null_calibration
        res = ROOT / "data" / "results"
        reports = pd.read_parquet(res / "reports_clean.parquet")
        q = cluster_check.data_quality(reports)
        print(f"\n=== Data quality ===\n{q['n_reports']} reports, {q['distinct_dates']} distinct receive dates over {q['months']} month(s).")
        print(("WARNING: " + q["warning"]) if q["warning"] else "Sample is spread over the period.")
        if calibrate:
            print("\n=== Permutation null ===", flush=True)
            T = pd.read_parquet(res / "triplet_results.parquet")
            out = null_calibration.permutation_null(reports, T)
            out["null"] = out["null"].to_dict("records")
            (res / "null_calibration.json").write_text(json.dumps(out, indent=1, default=str))
            print(json.dumps({k: v for k, v in out.items() if k != "null"}, indent=1, default=str))
    except Exception as e:  # noqa: BLE001 - diagnostics must not break the pipeline
        print(f"(data-quality step skipped: {type(e).__name__}: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-records", type=int, default=20000)
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--full", action="store_true", help="also run the DuckDB analysis, ML and dashboard export")
    ap.add_argument("--calibrate", action="store_true", help="run the permutation-null false-alarm check (slower; writes data/results/null_calibration.json)")
    ap.add_argument("--target", default=None, help="reaction for the ML models (with --full); default: most frequent")
    args = ap.parse_args()

    preflight()
    if not args.skip_fetch:
        run("1. Fetch from openFDA", ["fetch_fda_api.py", "--max-records", str(args.max_records)])
    run("2. Clean + mine (FP-Growth, ROR, attribution)", ["process_and_mine.py"])
    quality_report(args.calibrate)
    if args.full:
        a = "src/analysis/"
        run("3. Load into DuckDB", ["load_to_duckdb.py"])
        run("4. Pair signals (PRR / chi-square)", [a + "association_rules.py"])
        run("5. Pair interaction regression", [a + "logistic_regression.py"])
        run("6. ML models", [a + "advanced_models.py"] + (["--target", args.target] if args.target else []))
    print("\nDone. Start a dashboard:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
