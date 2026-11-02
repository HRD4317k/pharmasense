"""Command line:  python -m pharmasense_ai <command> ...

  doctor     test every data source with one small real request and say what worked   <- run this first
  explain    Explain + Challenge: why might these drugs interact (and cause this reaction)?
  live-faers count a combination in the WHOLE openFDA database and judge it
  discover   screen the local FAERS results for combination signals that are not documented
  analogs    structure, look-alike drugs and what they show
  validate   benchmark against known interactions (--calibrate to re-fit the evidence weights)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import ROOT, Settings


def _svc():
    from .services import Services
    return Services.build()


def _dump(obj, path: str | None):
    text = json.dumps(obj, indent=1, default=str)
    if path:
        Path(path).write_text(text)
        print(f"wrote {path}")
    return text


# ---------------------------------------------------------------------------------------------- doctor
def cmd_doctor(a) -> int:
    """One tiny real request per source. Live behaviour of these APIs was NOT verifiable when this code was written
    (no network access in the build environment), so this is how you find out what works on your machine."""
    s = Settings.load()
    print("Configuration:")
    for k, v in s.describe().items():
        print(f"  {k}: {v}")
    svc = _svc()
    checks = []

    last = {}

    def check(name, fn, ok):
        t = time.time()
        try:
            r = last[name] = fn()
            good, detail = ok(r)
        except Exception as e:  # noqa: BLE001 - this is a diagnostic, show anything
            good, detail = False, f"{type(e).__name__}: {e}"
        checks.append((name, good))
        print(f"  [{'OK ' if good else 'FAIL'}] {name:<22} {detail}  ({time.time() - t:.1f}s)")

    print("\nData sources (aspirin / warfarin):")
    check("PubChem", lambda: svc.pubchem.compound("aspirin"), lambda r: (bool(r and (r.get("smiles") or r.get("connectivity_smiles"))), f"SMILES {(r or {}).get('smiles') or (r or {}).get('connectivity_smiles')}"))
    check("ChEMBL", lambda: svc.chembl.find("warfarin"), lambda r: (bool(r and r.get("chembl_id")), f"id {(r or {}).get('chembl_id')}"))
    cid = (last.get("ChEMBL") or {}).get("chembl_id")
    if cid:
        check("ChEMBL mechanisms", lambda: svc.chembl.mechanisms(cid), lambda r: (bool(r), f"{len(r)} mechanism(s)"))
        check("Open Targets", lambda: svc.opentargets.mechanisms(cid), lambda r: (bool(r), f"{len(r)} mechanism row(s) (empty can be normal)"))
    check("UniProt", lambda: svc.uniprot.protein("Q9BQB6"), lambda r: (bool(r and r.get("gene")), f"gene {(r or {}).get('gene')}"))
    check("openFDA label", lambda: svc.openfda.label("warfarin"), lambda r: (bool(r and r.get("sections")), f"{len((r or {}).get('sections', {}))} label sections"))
    check("openFDA events", lambda: svc.openfda.event_total(svc.openfda.drug_clause("warfarin")), lambda r: (r > 1000, f"{r:,} warfarin reports"))
    check("Europe PMC", lambda: svc.europepmc.search("warfarin fluconazole interaction", 5), lambda r: (bool(r), f"{len(r)} record(s)"))
    check("STRING", lambda: svc.string.network(["CYP2C9", "CYP3A4", "CYP2C19", "CYP2D6"], 0.4), lambda r: (bool(r), f"{len(r or [])} edge(s)"))
    check("ClinicalTrials.gov", lambda: svc.ctgov.studies("warfarin", 3), lambda r: (bool(r), f"{len(r)} study record(s)"))
    print("\nLLM (optional):", end=" ")
    if not s.llm_ready:
        print(f"not configured (LLM_PROVIDER={s.llm_provider or 'none'}). The workflow runs fully without it.")
    else:
        r = svc.llm.chat("Reply with the single word OK.", "ping", max_tokens=5)
        print("OK" if r else "FAILED - check the key / model / base URL in .env")
        checks.append(("LLM", bool(r)))
    bad = [n for n, g in checks if not g]
    print("\n" + ("All checks passed." if not bad else f"{len(bad)} check(s) failed: {', '.join(bad)}. Failed sources are skipped and shown as gaps in reports; "
                                                    "send the FAIL lines above if you want them fixed."))
    return 1 if bad else 0


# ---------------------------------------------------------------------------------------------- explain
def cmd_explain(a) -> int:
    from . import faers_context as fc
    from .agents.workflow import explain
    from .report import render_markdown
    svc = _svc()
    faers = None
    if a.reaction and a.live_faers:
        faers = fc.from_live(svc.openfda, a.drugs, a.reaction)
    elif a.reaction and a.from_results:
        faers = fc.from_results(a.drugs, a.reaction)
    rep = explain(a.drugs, a.reaction, faers, svc, use_calibration=not a.no_calibration)
    if a.json:
        print(_dump(rep, a.out))
    else:
        md = render_markdown(rep)
        if a.out:
            Path(a.out).write_text(md)
            print(f"wrote {a.out}")
        print(md)
    return 0


def cmd_live_faers(a) -> int:
    from . import faers_context as fc
    svc = _svc()
    out = fc.from_live(svc.openfda, a.drugs, a.reaction)
    print(json.dumps({k: v for k, v in out.items() if k != "groups"}, indent=1, default=str))
    print("\ngroups (events, reports):", json.dumps(out["groups"]))
    return 0


# ---------------------------------------------------------------------------------------------- discover
def cmd_discover(a) -> int:
    import pandas as pd

    from . import discover
    res_dir = ROOT / "data" / "results"
    if not (res_dir / "triplet_results.parquet").exists():
        print("No local results found. Run `python run_pipeline.py` first.")
        return 2
    T = pd.read_parquet(res_dir / "triplet_results.parquet")
    reports = pd.read_parquet(res_dir / "reports_clean.parquet") if (res_dir / "reports_clean.parquet").exists() else None
    cand = discover.candidates_from_results(T, reports, top=a.top)
    print(f"{len(cand)} combination-like candidates; explaining the top {min(a.n, len(cand))} ...")
    out = discover.run_discovery(_svc(), cand, a.n)
    if len(out):
        print(out.drop(columns=["report"]).to_string(index=False))
        if a.out:
            _dump([r["report"] for r in out.to_dict("records")], a.out)
    return 0


def cmd_analogs(a) -> int:
    import pandas as pd

    from . import similar
    svc = _svc()
    lib_names = a.library or ["ibuprofen", "naproxen", "ketoprofen", "diclofenac", "celecoxib", "aspirin", "paracetamol", "warfarin", "simvastatin",
                              "atorvastatin", "pravastatin", "rosuvastatin", "fluoxetine", "sertraline", "paroxetine", "citalopram", "amiodarone", "fluconazole"]
    lib = similar.build_library(sorted(set(lib_names) | {a.drug.lower()}), svc)
    smi = lib.get(a.drug.lower())
    if not smi:
        print(f"No structure found for {a.drug!r} (PubChem unreachable or name not recognised).")
        return 2
    stats = None
    f = ROOT / "data" / "results" / "drug_reaction_stats.parquet"
    if f.exists():
        stats = pd.read_parquet(f)
    print(json.dumps(similar.analog_report(a.drug, smi, lib, stats, a.top), indent=1, default=str))
    return 0


# ---------------------------------------------------------------------------------------------- validate
def cmd_validate(a) -> int:
    from . import validate as v
    svc = _svc()
    rows = v.load_benchmark()
    print(f"Benchmark: {len(rows)} rows ({int((rows.label == 1).sum())} positive, {int((rows.label == 0).sum())} negative). Hand-curated - see docs/VALIDATION.md.\n")
    res, reports = v.run_benchmark(svc, rows)
    summary = v.summarise(res)
    print("\n" + json.dumps(summary, indent=1, default=str))
    if v.GOLD_ROLES.exists():
        print("\nLabel-parser recall vs gold roles:", json.dumps(v.parser_recall(svc), indent=1))
    if a.calibrate:
        print("\nCalibration:", json.dumps(v.calibrate(rows, reports), indent=1, default=str))
    if a.out:
        _dump({"summary": summary, "rows": res.to_dict("records")}, a.out)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pharmasense_ai", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("doctor", help="test every data source").set_defaults(fn=cmd_doctor)

    p = sp.add_parser("explain", help="mechanism hypotheses for 2+ drugs")
    p.add_argument("drugs", nargs="+")
    p.add_argument("--reaction")
    p.add_argument("--live-faers", action="store_true", help="count the whole openFDA database for this combination + reaction")
    p.add_argument("--from-results", action="store_true", help="use the local pipeline results (3 drugs)")
    p.add_argument("--no-calibration", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_explain)

    p = sp.add_parser("live-faers", help="whole-database counts and verdict")
    p.add_argument("drugs", nargs="+")
    p.add_argument("--reaction", required=True)
    p.set_defaults(fn=cmd_live_faers)

    p = sp.add_parser("discover", help="screen local results for undocumented combination signals")
    p.add_argument("--top", type=int, default=30, help="candidates to consider (by ROR lower bound)")
    p.add_argument("-n", type=int, default=8, help="candidates to explain")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_discover)

    p = sp.add_parser("analogs", help="structure + similar drugs")
    p.add_argument("drug")
    p.add_argument("--library", nargs="*")
    p.add_argument("--top", type=int, default=6)
    p.set_defaults(fn=cmd_analogs)

    p = sp.add_parser("validate", help="benchmark against known interactions")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_validate)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
