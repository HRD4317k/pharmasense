"""ChEMBL web services: molecule lookup, mechanism of action, targets, similarity. Free, no key.
Docs: https://www.ebi.ac.uk/chembl/api/data/docs   (data licence: CC BY-SA 3.0 - keep the attribution)
"""
from __future__ import annotations

import re
from urllib.parse import quote

from ..http import ApiError, Http

BASE = "https://www.ebi.ac.uk/chembl/api/data"


def strip_stereo(smiles: str) -> str:
    """Drop stereo markers so the SMILES is safe inside a URL path (a '/' would be read as a path separator)."""
    return re.sub(r"[/\\@]", "", smiles)


def _phase(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return -1.0


class ChEMBL:
    def __init__(self, http: Http):
        self.http = http

    def find(self, name: str) -> dict | None:
        data = self.http.get_json(f"{BASE}/molecule/search.json", {"q": name, "limit": 10})
        mols = (data or {}).get("molecules") or []
        if not mols:
            return None
        exact = [m for m in mols if str(m.get("pref_name") or "").lower() == name.lower()]
        m = (exact or sorted(mols, key=lambda m: -_phase(m.get("max_phase"))))[0]
        return {"chembl_id": m.get("molecule_chembl_id"), "name": m.get("pref_name"), "max_phase": m.get("max_phase"),
                "smiles": (m.get("molecule_structures") or {}).get("canonical_smiles")}

    def mechanisms(self, chembl_id: str) -> list[dict]:
        data = self.http.get_json(f"{BASE}/mechanism.json", {"molecule_chembl_id": chembl_id, "limit": 100})
        mechs = (data or {}).get("mechanisms") or []
        if not mechs:
            data = self.http.get_json(f"{BASE}/mechanism.json", {"parent_molecule_chembl_id": chembl_id, "limit": 100})
            mechs = (data or {}).get("mechanisms") or []
        out = []
        seen = set()
        for m in mechs:
            key = (m.get("mechanism_of_action"), m.get("target_chembl_id"))
            if key in seen:
                continue
            seen.add(key)
            out.append({"mechanism": m.get("mechanism_of_action"), "action_type": m.get("action_type"),
                        "target_chembl_id": m.get("target_chembl_id"), "refs": m.get("mechanism_refs") or []})
        return out

    def target(self, target_chembl_id: str) -> dict | None:
        data = self.http.get_json(f"{BASE}/target/{quote(target_chembl_id)}.json")
        if not data:
            return None
        comps = data.get("target_components") or []
        return {"chembl_id": target_chembl_id, "name": data.get("pref_name"), "type": data.get("target_type"),
                "organism": data.get("organism"), "uniprot": [c["accession"] for c in comps if c.get("accession")]}

    def indications(self, chembl_id: str) -> list[dict]:
        data = self.http.get_json(f"{BASE}/drug_indication.json", {"molecule_chembl_id": chembl_id, "limit": 100})
        inds = (data or {}).get("drug_indications") or []
        if not inds:
            data = self.http.get_json(f"{BASE}/drug_indication.json", {"parent_molecule_chembl_id": chembl_id, "limit": 100})
            inds = (data or {}).get("drug_indications") or []
        seen, out = set(), []
        for d in inds:
            term = d.get("mesh_heading") or d.get("efo_term")
            if term and term not in seen:
                seen.add(term)
                out.append({"indication": term, "max_phase": d.get("max_phase_for_ind")})
        return out

    def similar(self, smiles: str, cutoff: int = 70, limit: int = 15) -> list[dict]:
        """Molecules at least `cutoff` % (40-100) Tanimoto-similar to the SMILES, from ChEMBL's whole collection."""
        url = f"{BASE}/similarity/{quote(strip_stereo(smiles), safe='')}/{int(cutoff)}.json"
        try:
            data = self.http.get_json(url, {"limit": limit})
        except ApiError:
            return []
        out = []
        for m in (data or {}).get("molecules") or []:
            try:
                sim = float(m.get("similarity")) / 100.0
            except (TypeError, ValueError):
                sim = None
            out.append({"chembl_id": m.get("molecule_chembl_id"), "name": m.get("pref_name"), "similarity": sim,
                        "max_phase": m.get("max_phase"),
                        "smiles": (m.get("molecule_structures") or {}).get("canonical_smiles")})
        return out
