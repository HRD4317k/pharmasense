"""openFDA: drug labels (for RAG + enzyme roles) and FAERS counts (whole-database numbers without downloading).
Free key (recommended): https://open.fda.gov/apis/authentication/   - 240 requests/min and 120,000/day with a key,
1,000/day per IP without one.
"""
from __future__ import annotations

from ..http import Http

BASE = "https://api.fda.gov"
LABEL_SECTIONS = ["boxed_warning", "indications_and_usage", "contraindications", "warnings", "warnings_and_cautions",
                  "precautions", "adverse_reactions", "drug_interactions", "drug_and_or_laboratory_test_interactions",
                  "clinical_pharmacology", "mechanism_of_action", "pharmacokinetics", "pharmacodynamics",
                  "description", "overdosage", "use_in_specific_populations"]
CLASS_FIELDS = ["pharm_class_epc", "pharm_class_moa", "pharm_class_pe", "pharm_class_cs"]


def _q(name: str) -> str:
    return name.replace('"', " ").strip()


class OpenFDA:
    def __init__(self, http: Http, api_key: str = ""):
        self.http = http
        self.key = api_key

    def _p(self, params: dict) -> dict:
        return {**params, **({"api_key": self.key} if self.key else {})}

    # ------------------------------------------------------------------ labels
    def label(self, name: str) -> dict | None:
        """The most complete human prescription label for a drug name, reduced to the sections we use."""
        n = _q(name)
        search = (f'openfda.generic_name:"{n}" OR openfda.substance_name:"{n}" OR openfda.brand_name:"{n}"')
        d = self.http.get_json(f"{BASE}/drug/label.json", self._p({"search": search, "limit": 10}))
        results = (d or {}).get("results") or []
        if not results:
            return None

        def score(r):
            of = r.get("openfda") or {}
            rx = any("PRESCRIPTION" in str(t).upper() for t in of.get("product_type") or [])
            return (10 if rx else 0) + sum(1 for s in LABEL_SECTIONS if r.get(s)) + (5 if of.get("pharm_class_moa") else 0)

        r = max(results, key=score)
        of = r.get("openfda") or {}
        sections = {s: " ".join(str(x) for x in r[s]) if isinstance(r[s], list) else str(r[s])
                    for s in LABEL_SECTIONS if r.get(s)}
        set_id = r.get("set_id") or (of.get("spl_set_id") or [None])[0]
        return {"set_id": set_id, "version": r.get("effective_time"),
                "url": f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}" if set_id else BASE,
                "generic_name": (of.get("generic_name") or [None])[0], "brand_names": of.get("brand_name") or [],
                "classes": {f: of.get(f) or [] for f in CLASS_FIELDS}, "sections": sections}

    # ------------------------------------------------------------------ FAERS counts
    @staticmethod
    def drug_clause(name: str) -> str:
        return f'patient.drug.openfda.generic_name:"{_q(name)}"'

    def event_total(self, search: str | None = None) -> int:
        params = {"limit": 1}
        if search:
            params["search"] = search
        d = self.http.get_json(f"{BASE}/drug/event.json", self._p(params))
        return int((((d or {}).get("meta") or {}).get("results") or {}).get("total") or 0)

    def event_counts(self, search: str, field: str = "patient.reaction.reactionmeddrapt.exact", limit: int = 100):
        d = self.http.get_json(f"{BASE}/drug/event.json", self._p({"search": search, "count": field, "limit": limit}))
        return [(r["term"], int(r["count"])) for r in (d or {}).get("results") or []]
