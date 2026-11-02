"""PubChem PUG-REST: structure (SMILES) and computed properties. Free, no key.
Docs: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
"""
from __future__ import annotations

from urllib.parse import quote

from ..http import ApiError, Http

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PROPS = "MolecularFormula,MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,InChIKey,IUPACName"
# PubChem renamed its SMILES properties; we try the new names first, then the old ones.
SMILES_SETS = ("SMILES,ConnectivitySMILES", "IsomericSMILES,CanonicalSMILES")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def normalize(row: dict) -> dict:
    iso = next((row[k] for k in ("SMILES", "IsomericSMILES") if row.get(k)), None)
    con = next((row[k] for k in ("ConnectivitySMILES", "CanonicalSMILES") if row.get(k)), None)
    return {"cid": row.get("CID"), "smiles": iso or con, "connectivity_smiles": con or iso,
            "formula": row.get("MolecularFormula"), "mw": _num(row.get("MolecularWeight")),
            "xlogp": _num(row.get("XLogP")), "tpsa": _num(row.get("TPSA")),
            "hbd": row.get("HBondDonorCount"), "hba": row.get("HBondAcceptorCount"),
            "inchikey": row.get("InChIKey"), "iupac": row.get("IUPACName")}


class PubChem:
    def __init__(self, http: Http):
        self.http = http

    def compound(self, name: str) -> dict | None:
        """Best match for a drug name (generic or brand) or None if PubChem does not know it."""
        for smi in SMILES_SETS:
            url = f"{BASE}/compound/name/{quote(name, safe='')}/property/{smi},{PROPS}/JSON"
            try:
                data = self.http.get_json(url)
            except ApiError:
                continue                       # maybe an unknown property name: try the other spelling
            if data is None:
                return None                    # 404: PubChem has no such compound
            rows = (data.get("PropertyTable") or {}).get("Properties") or []
            if rows:
                return normalize(rows[0])
        return None
