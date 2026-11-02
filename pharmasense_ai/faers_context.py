"""FAERS evidence for the reasoning engine, from either source:

  from_results  the local sample analysed by process_and_mine.py (data/results/triplet_results.parquet)
  from_live     the WHOLE openFDA database, counted on demand (faers_live.py) and judged by the same statistical code

Both return the same dict: verdict, ror (+CI), n_events, n_reports, driver_drugs, flags, source, ... so the mechanism
workflow does not care where the numbers came from. Local numbers come from a small non-random sample; the live ones
cover everything openFDA can match by generic name (a lower bound - see faers_live caveats).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

import pharmasense_core as core

from . import faers_live
from .config import ROOT
from .mathx import interaction as ia

PAIR_TO_FAERS = {"Pair effect": "Pair effect", "Drug alone": "Single-drug effect", "Partner alone": "Single-drug effect",
                 "Both alone": "Single-drug effect", "Cannot separate": "Cannot separate", "No signal": "No signal"}


def from_results(drugs: list[str], reaction: str, results_dir: Path | None = None) -> dict | None:
    f = Path(results_dir or ROOT / "data" / "results") / "triplet_results.parquet"
    if len(drugs) != 3 or not f.exists():
        return None
    want = {core.normalize_drug(d) for d in drugs}
    T = pd.read_parquet(f)
    m = T[(T["reaction"] == reaction.upper()) & T[["drug_a", "drug_b", "drug_c"]].apply(lambda r: set(r) == want, axis=1)]
    if m.empty:
        return None
    r = m.iloc[0]
    meta = Path(f).parent / "meta.json"
    n = pd.read_json(meta, typ="series").get("n_reports", "?") if meta.exists() else "?"
    return {"verdict": r["verdict"], "ror": float(r["ror"]), "ror_lcl": float(r["ror_lcl"]), "ror_ucl": float(r["ror_ucl"]),
            "n_events": int(r["a_ABC"]), "n_reports": int(r["n_ABC"]), "driver_drugs": r.get("driver_drugs", ""),
            "flags": r.get("flags", ""), "q": float(r["q_ror"]), "source": f"local sample ({n} reports)", "justification": r.get("justification", "")}


def _additive(groups: dict, k: int) -> dict:
    try:
        return ia.reri_pair(groups) if k == 2 else ia.triple_contrast(groups)
    except Exception:
        return {}


def from_live(openfda, drugs: list[str], reaction: str, min_events: int = 3) -> dict | None:
    """Count the whole database for this exact combination and reaction and judge it with the repo's own statistics."""
    k = len(drugs)
    names = [core.normalize_drug(d) or d.upper() for d in drugs]
    groups = faers_live.live_group_counts(openfda, drugs, reaction)
    total_rx = sum(e for e, _ in groups.values())
    out = {"source": "openFDA whole database (generic-name match)", "groups": groups, "additive": _additive(groups, k)}
    if k == 3:
        cats = core.CATS
        n_k = np.array([groups[c][1] for c in cats])
        EV = np.array([[groups[c][0]] for c in cats])
        res = core._triplet_block(names, np.array([reaction.upper()]), np.array([total_rx]), n_k, EV)
        res = core.classify_triplets(core.add_exact_tests(res, min_events), min_events)
        r = res.iloc[0]
        out.update(verdict=r["verdict"], ror=float(r["ror_ABC"]), ror_lcl=float(r["lcl_ABC"]), ror_ucl=float(r["ucl_ABC"]),
                   n_events=int(r["a_ABC"]), n_reports=int(r["n_ABC"]), driver_drugs=r["driver_drugs"], flags=r["flags"],
                   q=float(r["q_ror"]), justification=r["justification"])
        return out
    if k == 2:
        g_n = {c: groups[c][1] for c in faers_live.CATS2}
        g_a = {c: groups[c][0] for c in faers_live.CATS2}
        e = {c: core.ror(g_a[c], g_n[c] - g_a[c], g_a["REF"], g_n["REF"] - g_a["REF"]) for c in ("A", "B", "AB")}
        h_a = core.ror(g_a["AB"], g_n["AB"] - g_a["AB"], g_a["A"], g_n["A"] - g_a["A"])
        h_b = core.ror(g_a["AB"], g_n["AB"] - g_a["AB"], g_a["B"], g_n["B"] - g_a["B"])

        def exact_p(c):
            M_, K_, n_ = g_n[c] + g_n["REF"], g_a[c] + g_a["REF"], g_n[c]
            p_obs, p_min = hypergeom.sf(g_a[c] - 1, M_, K_, n_), hypergeom.sf(min_events - 1, M_, K_, n_)
            return float(min(p_obs / p_min, 1.0)) if p_min > 0 else 1.0

        q = multipletests([exact_p("AB"), exact_p("A"), exact_p("B")], method="fdr_bh")[1]
        row = {"a_both": g_a["AB"], "lcl_AB": float(e["AB"][1]), "q": q[0], "a_A": g_a["A"], "lcl_A": float(e["A"][1]), "q_A": q[1],
               "a_B": g_a["B"], "lcl_B": float(e["B"][1]), "q_B": q[2], "h2h_lcl_A": float(h_a[1]), "h2h_lcl_B": float(h_b[1])}
        v = core._pair_verdict(row, min_events)
        who = {"Drug alone": names[0], "Partner alone": names[1], "Both alone": " | ".join(names)}.get(v, " | ".join(names) if v == "Pair effect" else "")
        out.update(verdict=PAIR_TO_FAERS[v], pair_verdict=v, ror=float(e["AB"][0]), ror_lcl=float(e["AB"][1]), ror_ucl=float(e["AB"][2]),
                   n_events=g_a["AB"], n_reports=g_n["AB"], driver_drugs=who, flags="", q=float(q[0]), justification="")
        return out
    return None
