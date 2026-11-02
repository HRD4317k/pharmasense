"""Structurally similar drugs and what is known about them - with the limits of that inference stated.

Given a drug: (1) its structure and descriptors, (2) look-alikes from a local library (RDKit, four fingerprints),
optionally widened with ChEMBL's similarity search, (3) for each look-alike: observed FAERS signals, label warnings
and mechanism - and (4) the reactions the look-alikes share that THIS drug does not show: reported as
"inferred from analogues", never as observed. Pairs that look alike but have clearly different observed reaction
profiles are flagged as possible activity cliffs.
"""
from __future__ import annotations

import json

import pandas as pd

from . import chem
from .config import Settings
from .http import ApiError

CLIFF_SIM, CLIFF_OVERLAP = 0.6, 0.2


def build_library(names: list[str], svc, cache_name: str = "library.json") -> dict[str, str]:
    """name -> SMILES via PubChem (cached on disk, so each drug is looked up once)."""
    path = svc.settings.cache_dir / cache_name
    lib = {}
    try:
        lib = json.loads(path.read_text())
    except Exception:
        pass
    for n in names:
        if n in lib:
            continue
        try:
            c = svc.pubchem.compound(n)
        except ApiError:
            c = None
        lib[n] = (c or {}).get("connectivity_smiles") or (c or {}).get("smiles") or ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(lib))
    except Exception:
        pass
    return {k: v for k, v in lib.items() if v}


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
