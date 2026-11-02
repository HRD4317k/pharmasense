"""Fetch polypharmacy reports (>= 3 drugs) from openFDA drug/event and save them to
data/structured_fda_data.parquet (one row per report).

The API key is read from the OPENFDA_API_KEY entry in the .env file next to this script
(or from --api-key). Run:  python fetch_fda_api.py --max-records 20000
"""
import argparse
import calendar
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BASE_URL = "https://api.fda.gov/drug/event.json"
PAGE_SIZE = 1000      # openFDA maximum for `limit`
MAX_SKIP = 25000      # openFDA maximum for `skip`
JUNK_DRUGS = {"UNKNOWN", "UNSPECIFIED", "UNK", "NOT SPECIFIED", "DRUG NOT SPECIFIED"}


def check_api_key(api_key):
    """One tiny request so a bad key / no internet is reported immediately."""
    params = {"search": "receivedate:[20230101 TO 20230107]", "limit": 1}
    if api_key:
        params["api_key"] = api_key
    try:
        r = requests.get(BASE_URL, params=params, timeout=30)
    except requests.RequestException as e:
        return False, f"Could not reach api.fda.gov ({e}). Check your internet connection."
    if r.status_code == 200:
        return True, "openFDA reachable" + (" and API key accepted." if api_key else " (no API key).")
    if r.status_code in (401, 403):
        return False, f"openFDA rejected the API key (HTTP {r.status_code}). Check OPENFDA_API_KEY in .env."
    return False, f"Unexpected response from openFDA: HTTP {r.status_code} {r.text[:200]}"


def get_page(params, retries=6):
    """GET with retry on 429 / 5xx / timeouts. Returns (status_code, json_or_None)."""
    for attempt in range(retries):
        try:
            r = requests.get(BASE_URL, params=params, timeout=60)
        except requests.RequestException as e:
            print(f"  network error ({e}); retry {attempt + 1}/{retries}")
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            print("  rate limited, waiting 15 s...")
            time.sleep(15)
            continue
        if r.status_code >= 500:
            print(f"  server error {r.status_code}; retry {attempt + 1}/{retries}")
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            return 200, r.json()
        return r.status_code, None
    return 599, None


def parse_report(item):
    """One record per report (>= 3 drug entries and >= 1 reaction), else None.
    Keeps, per drug: name, openFDA generic name (if matched), role (1 suspect / 2 concomitant /
    3 interacting) and indication - the app uses these to normalise names, drop reactions that are
    really the disease being treated, and optionally restrict to suspect drugs.
    NOTE: do NOT filter on the top-level `duplicate` field - openFDA sets it to "1" on essentially
    every record, so filtering on it would delete the whole dataset."""
    patient = item.get("patient") or {}
    names, gens, roles, inds = [], [], [], []
    for d in patient.get("drug") or []:
        name = str(d.get("medicinalproduct") or "").strip().upper()
        if not name:
            continue
        g = (d.get("openfda") or {}).get("generic_name")
        gen = str(g[0] if isinstance(g, list) and g else (g or "")).strip().upper()
        names.append(name)
        gens.append(gen)
        roles.append(str(d.get("drugcharacterization") or ""))
        inds.append(str(d.get("drugindication") or "").strip().upper())
    reactions = sorted({str(r.get("reactionmeddrapt") or "").strip().upper()
                        for r in patient.get("reaction") or []} - {""})
    if len(names) < 3 or not reactions:
        return None
    try:
        version = int(item.get("safetyreportversion") or 0)
    except ValueError:
        version = 0
    return {
        "report_id": item.get("safetyreportid"),
        "version": version,
        "company_numb": str(item.get("companynumb") or ""),
        "date": item.get("receivedate") or item.get("receiptdate"),
        "serious": str(item.get("serious") or ""),
        "patient_sex": str(patient.get("patientsex", "UNKNOWN")),
        "patient_age": str(patient.get("patientonsetage", "UNKNOWN")),
        "drug_names": names, "drug_generic": gens, "drug_roles": roles, "drug_indications": inds,
        "reactions": reactions,
    }


def fetch_data(api_key, max_records=20000, year=2023):
    # Spread the quota evenly over the 12 months so the sample covers the whole year
    per_month = math.ceil(max_records / 12)
    records = []
    print(f"Fetching up to {max_records} polypharmacy reports from {year} (~{per_month} per month).")

    for month in range(1, 13):
        last_day = calendar.monthrange(year, month)[1]
        start, end = f"{year}{month:02d}01", f"{year}{month:02d}{last_day}"
        month_records, skip = 0, 0
        print(f"Month {year}-{month:02d}")
        while month_records < per_month and skip < MAX_SKIP:
            params = {"search": f"receivedate:[{start} TO {end}]", "limit": PAGE_SIZE, "skip": skip}
            if api_key:
                params["api_key"] = api_key
            status, data = get_page(params)
            if status == 404:        # openFDA's "no more matches"
                break
            if status != 200 or not data:
                print(f"  stopping this month (HTTP {status})")
                break
            for item in data.get("results", []):
                rec = parse_report(item)
                if rec:
                    records.append(rec)
                    month_records += 1
                    if month_records >= per_month:
                        break
            skip += PAGE_SIZE
            print(f"  scanned {skip} raw reports, kept {month_records}")
            time.sleep(0.3)

    if not records:
        print("No usable records fetched.")
        return None

    df = pd.DataFrame(records)
    # One row per case: keep the latest version of each manufacturer case number
    df = df.sort_values("version", ascending=False)
    has_case = df["company_numb"] != ""
    df = pd.concat([df[has_case].drop_duplicates("company_numb"), df[~has_case]]).drop_duplicates("report_id")
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    df.to_parquet(out / "structured_fda_data.parquet", index=False)
    print(f"Saved {len(df)} reports to data/structured_fda_data.parquet "
          f"({len(records) - len(df)} duplicate case versions removed)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("OPENFDA_API_KEY"), help="openFDA key (default: .env)")
    ap.add_argument("--max-records", type=int, default=20000, help="max polypharmacy reports to keep")
    ap.add_argument("--year", type=int, default=2023)
    args = ap.parse_args()

    key = (args.api_key or "").strip()
    if key in ("", "your_openfda_key_here"):
        print("WARNING: no API key found - openFDA's lower anonymous limits apply.")
        key = None
    ok, msg = check_api_key(key)
    print(msg)
    if not ok:
        sys.exit(1)
    if fetch_data(key, args.max_records, args.year) is None:
        sys.exit(1)
