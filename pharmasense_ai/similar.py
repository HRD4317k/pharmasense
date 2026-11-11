"""Structurally similar drugs and what is known about them - with the limits of that inference stated.

Given a drug: (1) its structure and descriptors, (2) look-alikes from a local library (RDKit, four fingerprints),
optionally widened with ChEMBL's similarity search, (3) for each look-alike: observed FAERS signals, label warnings
and mechanism - and (4) the reactions the look-alikes share that THIS drug does not show: reported as
"inferred from analogues", never as observed. Pairs that look alike but have clearly different observed reaction
profiles are flagged as possible activity cliffs.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd

from . import chem
from .config import Settings

CLIFF_SIM, CLIFF_OVERLAP = 0.6, 0.2


BUNDLED = Path(__file__).resolve().parents[1] / "data" / "reference" / "library_smiles.json"


def bundled_library() -> dict[str, str]:
    """Structures shipped with the repository, so the default comparison library needs no network at all."""
    try:
        return dict(json.loads(BUNDLED.read_text())["smiles"])
    except Exception:
        return {}


def build_library_timed(names: list[str], svc, budget_s: float = 30.0, cache_name: str = "library.json"):
    """name -> SMILES. Order of sources: bundled file, on-disk cache, then PubChem for the rest.

    PubChem lookups run in a background thread and the caller waits at most `budget_s` seconds, so a slow or
    unreachable PubChem can never hang the app. Lookups still running keep filling the cache for the next try.
    Returns (library, pending): pending = names PubChem did not answer for (too slow or unreachable), as opposed to
    names it answered "unknown" for, which are simply absent from the library."""
    path = svc.settings.cache_dir / cache_name
    lib = bundled_library()
    try:
        lib = {**json.loads(path.read_text()), **lib}
    except Exception:
        pass
    todo = [n for n in names if n not in lib]
    found: dict[str, str] = {}
    base = dict(lib)
    lock = threading.Lock()

    def work():
        for n in todo:
            try:
                c = svc.pubchem.compound(n)
            except Exception:  # noqa: BLE001 - unreachable or failing: leave it pending, never cache it as "unknown"
                continue
            with lock:
                found[n] = (c or {}).get("connectivity_smiles") or (c or {}).get("smiles") or ""
                snapshot = {**base, **found}
            _save(path, snapshot)

    if todo:
        t = threading.Thread(target=work, daemon=True)
        t.start()
        t.join(budget_s)
    with lock:
        lib.update(found)
    pending = [n for n in todo if n not in found]
    return {k: v for k, v in lib.items() if v}, pending


def _save(path, lib):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(lib))
    except Exception:
        pass


def build_library(names: list[str], svc, cache_name: str = "library.json", budget_s: float = 120.0) -> dict[str, str]:
    """name -> SMILES (see build_library_timed)."""
    return build_library_timed(names, svc, budget_s, cache_name)[0]


def _signals(stats: pd.DataFrame, drug: str, top: int = 25) -> list[str]:
    s = stats[(stats["drug"] == drug.upper()) & stats["signal"]].sort_values("ror_lcl", ascending=False)
    return list(s["reaction"].head(top))


def analog_report(drug: str, smiles: str, library: dict[str, str], stats: pd.DataFrame | None, top_k: int = 6,
                  min_sim: float = 0.3, chembl_hits: list[dict] | None = None) -> dict:
    sims = chem.rank_similar(smiles, library, top_k, min_sim, exclude={drug, drug.upper()})
    own = set(_signals(stats, drug)) if stats is not None else set()
    rows, cliff, share = [], [], {}
    for s in sims:
        theirs = set(_signals(stats, s["name"])) if stats is not None else set()
        overlap = len(own & theirs) / len(own | theirs) if (own | theirs) else float("nan")
        rows.append({"drug": s["name"], "similarity": s["mean"], "by_fingerprint": {k: s[k] for k in chem.FP_KINDS},
                     "scaffold_note": chem.shared_substructure_note(smiles, s["smiles"]), "observed_reactions": sorted(theirs)[:10],
                     "reaction_overlap": overlap, "smiles": s["smiles"]})
        if s["mean"] >= CLIFF_SIM and overlap == overlap and overlap < CLIFF_OVERLAP and (own or theirs):
            cliff.append(s["name"])
        for r in theirs:
            share[r] = share.get(r, 0) + 1
    inferred = sorted([r for r, k in share.items() if k >= 2 and r not in own], key=lambda r: -share[r])[:10]
    return {"drug": drug, "descriptors": chem.descriptors(smiles), "scaffold": chem.scaffold(smiles), "analogues": rows,
            "chembl_similar": chembl_hits or [], "observed_reactions": sorted(own)[:15],
            "inferred_from_analogues": [{"reaction": r, "analogues_with_signal": share[r], "status": "inferred, not observed for this drug"} for r in inferred],
            "possible_activity_cliffs": cliff,
            "caveat": ("Similar structures often share targets and side effects, but small changes can switch activity on or off "
                       "(activity cliffs). Inferred reactions are hypotheses to check against the drug's own label.")}
