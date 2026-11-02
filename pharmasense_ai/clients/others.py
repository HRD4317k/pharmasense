"""Two small free clients: STRING (protein-protein links) and ClinicalTrials.gov v2 (human studies). No keys."""
from __future__ import annotations

from ..http import ApiError, Http


class StringDB:
    """https://string-db.org/help/api/  (CC BY 4.0). Asks callers to identify themselves and to go slowly."""

    def __init__(self, http: Http, caller: str = "pharmasense"):
        self.http = http
        self.caller = caller

    def network(self, genes: list[str], min_score: float = 0.7, species: int = 9606) -> list[dict]:
        genes = [g for g in dict.fromkeys(genes) if g]
        if len(genes) < 2:
            return []
        try:
            d = self.http.get_json("https://string-db.org/api/json/network",
                                   {"identifiers": "\r".join(genes), "species": species,
                                    "required_score": int(min_score * 1000), "caller_identity": self.caller})
        except ApiError:
            return []
        out = []
        for e in d or []:
            try:
                score = float(e.get("score"))
            except (TypeError, ValueError):
                continue
            a, b = e.get("preferredName_A"), e.get("preferredName_B")
            if a and b and a != b:
                out.append({"a": a, "b": b, "score": score})
        return out


class ClinicalTrials:
    """https://clinicaltrials.gov/data-api/api  (public domain)."""

    URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self, http: Http):
        self.http = http

    def studies(self, drug: str, limit: int = 20) -> list[dict]:
        try:
            d = self.http.get_json(self.URL, {"query.intr": drug, "pageSize": limit, "format": "json"})
        except ApiError:
            return []
        out = []
        for s in (d or {}).get("studies") or []:
            p = s.get("protocolSection") or {}
            out.append({"nct_id": (p.get("identificationModule") or {}).get("nctId"),
                        "title": (p.get("identificationModule") or {}).get("briefTitle"),
                        "status": (p.get("statusModule") or {}).get("overallStatus"),
                        "phases": (p.get("designModule") or {}).get("phases") or [],
                        "conditions": (p.get("conditionsModule") or {}).get("conditions") or []})
        return out
