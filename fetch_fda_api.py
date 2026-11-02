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


def pick_days(year, month, k=None):
    """Days of the month to sample: all of them, or k evenly spaced ones (for quick test runs)."""
    last = calendar.monthrange(year, month)[1]
    if not k or k >= last:
        return list(range(1, last + 1))
    return sorted({round(1 + i * (last - 1) / max(k - 1, 1)) for i in range(k)})


def fetch_data(api_key, max_records=20000, year=2023, sample_days=None):
    """Sample DAY BY DAY, not month by month.

    openFDA returns a month's records in date order, so taking "the first N of each month" (what this function used to
    do) returned only reports received on the 1st - the shipped sample had 15 distinct receive dates in a year, and
    batch submissions made on those days looked like signals. One quota per day spreads the sample over the period."""
    days = [(m, d) for m in range(1, 13) for d in pick_days(year, m, sample_days)]
    per_day = max(1, math.ceil(max_records / len(days)))
    records = []
    print(f"Fetching up to {max_records} polypharmacy reports from {year}: {len(days)} days x ~{per_day} per day.")

    for m, d in days:
        day = f"{year}{m:02d}{d:02d}"
        day_records, skip = 0, 0
        while day_records < per_day and skip < MAX_SKIP:
            params = {"search": f"receivedate:[{day} TO {day}]", "limit": PAGE_SIZE, "skip": skip}
            if api_key:
                params["api_key"] = api_key
            status, data = get_page(params)
            if status == 404:        # openFDA's "no more matches"
                break
            if status != 200 or not data:
                print(f"  stopping {day} (HTTP {status})")
                break
            for item in data.get("results", []):
                rec = parse_report(item)
                if rec:
                    records.append(rec)
                    day_records += 1
                    if day_records >= per_day:
                        break
            skip += PAGE_SIZE
            time.sleep(0.3)
        if d == 1 or d % 10 == 0:
            print(f"  {day}: kept {day_records} (total {len(records)})")

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
          f"({len(records) - len(df)} duplicate case versions removed); {df['date'].nunique()} distinct receive dates")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("OPENFDA_API_KEY"), help="openFDA key (default: .env)")
    ap.add_argument("--max-records", type=int, default=20000, help="max polypharmacy reports to keep")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--sample-days", type=int, default=None,
                    help="days sampled per month (evenly spaced); default all days. Use e.g. 5 for a quick test run")
    args = ap.parse_args()

    key = (args.api_key or "").strip()
    if key in ("", "your_openfda_key_here"):
        print("WARNING: no API key found - openFDA's lower anonymous limits apply.")
        key = None
    ok, msg = check_api_key(key)
    print(msg)
    if not ok:
        sys.exit(1)
    if fetch_data(key, args.max_records, args.year, args.sample_days) is None:
        sys.exit(1)
