"""Open Targets Platform GraphQL: mechanisms of action and FAERS-derived adverse-event enrichment. Free, no key.
Docs: https://platform-docs.opentargets.org/data-access/graphql-api   (data licence: CC0)

The field names below follow the public v4 schema but were NOT verifiable from the build sandbox; each query is
separate and defensive so a changed field only loses that one block, and `python -m pharmasense_ai doctor`
tells you at once whether they still work.
"""
from __future__ import annotations

from ..http import ApiError, Http

URL = "https://api.platform.opentargets.org/api/v4/graphql"

Q_MOA = """query($id: String!) { drug(chemblId: $id) { id name
  mechanismsOfAction { rows { mechanismOfAction actionType targetName targets { id approvedSymbol } } } } }"""
Q_AE = """query($id: String!) { drug(chemblId: $id) {
  adverseEvents(page: {index: 0, size: 50}) { count criticalValue rows { name meddraCode count logLR } } } }"""


class OpenTargets:
    def __init__(self, http: Http):
        self.http = http

    def _q(self, query: str, chembl_id: str) -> dict | None:
        try:
            d = self.http.post_json(URL, {"query": query, "variables": {"id": chembl_id}})
        except ApiError:
            return None
        if not d or d.get("errors") or not (d.get("data") or {}).get("drug"):
            return None
        return d["data"]["drug"]

    def mechanisms(self, chembl_id: str) -> list[dict]:
        drug = self._q(Q_MOA, chembl_id) or {}
        rows = ((drug.get("mechanismsOfAction") or {}).get("rows")) or []
        return [{"mechanism": r.get("mechanismOfAction"), "action_type": r.get("actionType"),
                 "target_name": r.get("targetName"),
                 "targets": [{"ensembl": t.get("id"), "symbol": t.get("approvedSymbol")} for t in r.get("targets") or []]}
                for r in rows]

    def adverse_events(self, chembl_id: str) -> list[dict]:
        """Reactions significantly over-represented in FAERS for this drug (log-likelihood ratio, FDR-controlled)."""
        drug = self._q(Q_AE, chembl_id) or {}
        rows = ((drug.get("adverseEvents") or {}).get("rows")) or []
        return [{"reaction": r.get("name"), "meddra": r.get("meddraCode"), "reports": r.get("count"), "llr": r.get("logLR")}
                for r in rows]
