"""UniProt: protein name, gene, function text and Reactome pathway membership. Free, no key.
Pathways come straight from UniProt's Reactome cross-references, so no separate Reactome call is needed.
Docs: https://www.uniprot.org/help/api
"""
from __future__ import annotations

from urllib.parse import quote

from ..http import Http


def _dig(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


class UniProt:
    def __init__(self, http: Http):
        self.http = http

    def protein(self, accession: str) -> dict | None:
        d = self.http.get_json(f"https://rest.uniprot.org/uniprotkb/{quote(accession)}.json")
        if not d:
            return None
        desc = d.get("proteinDescription") or {}
        name = (_dig(desc, "recommendedName", "fullName", "value")
                or _dig((desc.get("submissionNames") or [{}])[0], "fullName", "value"))
        gene = _dig((d.get("genes") or [{}])[0], "geneName", "value")
        function = None
        for c in d.get("comments") or []:
            if c.get("commentType") == "FUNCTION" and c.get("texts"):
                function = c["texts"][0].get("value")
                break
        pathways = []
        for x in d.get("uniProtKBCrossReferences") or []:
            if x.get("database") == "Reactome":
                nm = next((p.get("value") for p in x.get("properties") or [] if p.get("key") == "PathwayName"), x.get("id"))
                pathways.append({"id": x.get("id"), "name": nm})
        return {"accession": accession, "name": name, "gene": gene, "function": function,
                "organism": _dig(d, "organism", "scientificName"), "pathways": pathways}
