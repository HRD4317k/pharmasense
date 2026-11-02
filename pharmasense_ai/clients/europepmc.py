"""Europe PMC REST: PubMed/MEDLINE abstracts (plus open-access full-text metadata). Free, no key.
Docs: https://europepmc.org/RestfulWebService
Europe PMC indexes all of PubMed, so no NCBI key is needed.
"""
from __future__ import annotations

import re

from ..http import ApiError, Http

URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_TAG = re.compile(r"<[^>]+>")


class EuropePMC:
    def __init__(self, http: Http):
        self.http = http

    def search(self, query: str, page_size: int = 25) -> list[dict]:
        try:
            d = self.http.get_json(URL, {"query": f"({query}) AND SRC:MED", "format": "json",
                                         "pageSize": page_size, "resultType": "core"})
        except ApiError:
            return []
        out = []
        for r in ((d or {}).get("resultList") or {}).get("result") or []:
            abstract = _TAG.sub(" ", r.get("abstractText") or "")
            abstract = re.sub(r"\s+", " ", abstract).strip()
            if not abstract:
                continue
            pmid = r.get("pmid") or r.get("id")
            out.append({"pmid": pmid, "title": _TAG.sub("", r.get("title") or "").strip(), "abstract": abstract,
                        "year": r.get("pubYear"), "journal": r.get("journalTitle"), "doi": r.get("doi"),
                        "url": f"https://europepmc.org/article/{r.get('source', 'MED')}/{r.get('id')}"})
        return out

    @staticmethod
    def interaction_query(drug_a: str, drug_b: str, extra: str = "") -> str:
        q = f'"{drug_a}" AND "{drug_b}" AND (interaction OR "drug interaction" OR pharmacokinetics OR coadministration)'
        return f"{q} AND ({extra})" if extra else q
