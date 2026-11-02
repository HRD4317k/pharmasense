"""Offline fakes for the workflow tests.

IMPORTANT: everything here is a hand-written TEST FIXTURE shaped like the documented API responses. It is not data
captured from the live services, and the label sentences are paraphrases written for these tests, not FDA text.
"""
from __future__ import annotations

from pathlib import Path

from pharmasense_ai.http import ApiError
from pharmasense_ai.clients.europepmc import EuropePMC


class FakeLLM:
    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.ready = bool(replies)
        self.last_error = ""
        self.calls = 0

    def chat(self, system, user, **kw):
        self.calls += 1
        return self.replies.pop(0) if self.replies else None


WORLD = {
    "warfarin": {
        "pubchem": {"cid": 54678486, "smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O", "mw": 308.3},
        "chembl": {"chembl_id": "CHEMBL1464", "name": "WARFARIN", "max_phase": 4, "smiles": None},
        "mechanisms": [{"mechanism": "Vitamin K epoxide reductase complex subunit 1 inhibitor", "action_type": "INHIBITOR", "target_chembl_id": "CHEMBL1957", "refs": []}],
        "targets": {"CHEMBL1957": {"chembl_id": "CHEMBL1957", "name": "Vitamin K epoxide reductase complex subunit 1", "type": "SINGLE PROTEIN",
                                   "organism": "Homo sapiens", "uniprot": ["Q9BQB6"]}},
        "uniprot": {"Q9BQB6": {"accession": "Q9BQB6", "name": "Vitamin K epoxide reductase complex subunit 1", "gene": "VKORC1", "function": "",
                               "organism": "Homo sapiens", "pathways": [{"id": "R-HSA-159740", "name": "Gamma-carboxylation of protein precursors"}]}},
        "label": {"set_id": "TEST-WARF", "url": "https://dailymed.example/warf", "generic_name": "warfarin sodium", "brand_names": ["Coumadin"],
                  "classes": {"pharm_class_epc": ["Vitamin K Antagonist [EPC]"], "pharm_class_moa": [], "pharm_class_pe": [], "pharm_class_cs": []},
                  "sections": {
                      "clinical_pharmacology": "Warfarin is a racemic mixture of two enantiomers. The S-enantiomer is metabolized primarily by CYP2C9. "
                                               "The R-enantiomer is metabolized by CYP1A2 and CYP3A4.",
                      "drug_interactions": "Drugs that inhibit CYP2C9 may increase warfarin exposure and the INR. Concomitant antiplatelet agents or NSAIDs raise the bleeding risk.",
                      "warnings": "Warfarin can cause major or fatal bleeding. Monitor the INR regularly and adjust the dose.",
                      "adverse_reactions": "Hemorrhage is the most common adverse reaction. International normalized ratio increased has been reported."}},
    },
    "fluconazole": {
        "pubchem": {"cid": 3365, "smiles": "C1=CC(=C(C=C1F)F)C(CN2C=NC=N2)(CN3C=NC=N3)O", "mw": 306.3},
        "chembl": {"chembl_id": "CHEMBL238525", "name": "FLUCONAZOLE", "max_phase": 4, "smiles": None},
        "mechanisms": [{"mechanism": "Lanosterol 14-alpha demethylase inhibitor", "action_type": "INHIBITOR", "target_chembl_id": "CHEMBL1978", "refs": []}],
        "targets": {"CHEMBL1978": {"chembl_id": "CHEMBL1978", "name": "Lanosterol 14-alpha demethylase", "type": "SINGLE PROTEIN",
                                   "organism": "Candida albicans", "uniprot": ["P10613"]}},
        "uniprot": {},
        "label": {"set_id": "TEST-FLU", "url": "https://dailymed.example/flu", "generic_name": "fluconazole", "brand_names": ["Diflucan"],
                  "classes": {"pharm_class_epc": ["Azole Antifungal [EPC]"],
                              "pharm_class_moa": ["Cytochrome P450 2C9 Inhibitors [MoA]", "Cytochrome P450 3A4 Inhibitors [MoA]"],
                              "pharm_class_pe": [], "pharm_class_cs": []},
                  "sections": {
                      "clinical_pharmacology": "Fluconazole is a potent inhibitor of CYP2C9 and CYP3A4 and a moderate inhibitor of CYP2C19.",
                      "drug_interactions": "Fluconazole increases plasma concentrations of drugs metabolized by CYP2C9, including warfarin. "
                                           "Prolongation of the QT interval and torsade de pointes have been reported.",
                      "warnings": "Cases of QT prolongation and torsade de pointes have occurred; use caution with other QT prolonging drugs."}},
    },
    "pravastatin": {
        "pubchem": {"cid": 54687, "smiles": "CCC(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC(CC(CC(=O)O)O)O)O", "mw": 424.5},
        "chembl": {"chembl_id": "CHEMBL1144", "name": "PRAVASTATIN", "max_phase": 4, "smiles": None},
        "mechanisms": [], "targets": {}, "uniprot": {},
        "label": {"set_id": "TEST-PRAVA", "url": "https://dailymed.example/prava", "generic_name": "pravastatin sodium", "brand_names": [],
                  "classes": {"pharm_class_epc": ["HMG-CoA Reductase Inhibitor [EPC]"], "pharm_class_moa": [], "pharm_class_pe": [], "pharm_class_cs": []},
                  "sections": {"clinical_pharmacology": "Pravastatin undergoes minimal metabolism by the cytochrome P450 system and is cleared mainly by the liver and kidney.",
                               "warnings": "Myopathy and rhabdomyolysis have been reported."}},
    },
    "ketoconazole": {
        "pubchem": {"cid": 456201, "smiles": "CC(=O)N1CCN(CC1)C2=CC=C(C=C2)OCC3COC(O3)(CN4C=CN=C4)C5=C(C=C(C=C5)Cl)Cl", "mw": 531.4},
        "chembl": {"chembl_id": "CHEMBL75", "name": "KETOCONAZOLE", "max_phase": 4, "smiles": None},
        "mechanisms": [], "targets": {}, "uniprot": {},
        "label": {"set_id": "TEST-KETO", "url": "https://dailymed.example/keto", "generic_name": "ketoconazole", "brand_names": [],
                  "classes": {"pharm_class_epc": ["Azole Antifungal [EPC]"], "pharm_class_moa": ["Cytochrome P450 3A4 Inhibitors [MoA]"],
                              "pharm_class_pe": [], "pharm_class_cs": []},
                  "sections": {"clinical_pharmacology": "Ketoconazole is a potent inhibitor of CYP3A4."}},
    },
    # two invented drugs that share a pharmacological target but no enzyme: exercises the PD path and the loop-back edge
    "alphadrug": {
        "pubchem": {"cid": 1, "smiles": "CCO", "mw": 46.07}, "chembl": {"chembl_id": "CHEMBLA", "name": "ALPHADRUG", "max_phase": 4, "smiles": None},
        "mechanisms": [{"mechanism": "Cyclooxygenase-2 inhibitor", "action_type": "INHIBITOR", "target_chembl_id": "CHEMBLT1", "refs": []}],
        "targets": {"CHEMBLT1": {"chembl_id": "CHEMBLT1", "name": "Cyclooxygenase-2", "type": "SINGLE PROTEIN", "organism": "Homo sapiens", "uniprot": ["P35354"]}},
        "uniprot": {"P35354": {"accession": "P35354", "name": "Prostaglandin G/H synthase 2", "gene": "PTGS2", "function": "", "organism": "Homo sapiens", "pathways": []}},
        "label": None},
    "betadrug": {
        "pubchem": {"cid": 2, "smiles": "CCCO", "mw": 60.1}, "chembl": {"chembl_id": "CHEMBLB", "name": "BETADRUG", "max_phase": 4, "smiles": None},
        "mechanisms": [{"mechanism": "Cyclooxygenase-2 inhibitor", "action_type": "INHIBITOR", "target_chembl_id": "CHEMBLT1", "refs": []}],
        "targets": {"CHEMBLT1": {"chembl_id": "CHEMBLT1", "name": "Cyclooxygenase-2", "type": "SINGLE PROTEIN", "organism": "Homo sapiens", "uniprot": ["P35354"]}},
        "uniprot": {"P35354": {"accession": "P35354", "name": "Prostaglandin G/H synthase 2", "gene": "PTGS2", "function": "", "organism": "Homo sapiens", "pathways": []}},
        "label": None},
}

PAPERS_SUPPORT = {"pmid": "1001", "title": "Fluconazole potentiates warfarin", "year": "2010", "journal": "J Test", "url": "https://europepmc.example/1001",
                  "abstract": "Fluconazole inhibits CYP2C9, the main route of S-warfarin clearance. Co-administration of fluconazole and warfarin increased the INR in all patients."}
PAPERS_NULL = {"pmid": "1002", "title": "Fluconazole and warfarin in volunteers", "year": "2012", "journal": "J Test", "url": "https://europepmc.example/1002",
               "abstract": "In healthy volunteers fluconazole did not significantly alter the pharmacokinetics of warfarin. The study was small."}


class _Base:
    def __init__(self, world, fail=False):
        self.w, self.fail = world, fail

    def _chk(self):
        if self.fail:
            raise ApiError("fake outage")

    def _d(self, name):
        return self.w.get(name.lower())


class FakePubChem(_Base):
    def compound(self, name):
        self._chk()
        d = self._d(name)
        return d["pubchem"] if d else None


class FakeChEMBL(_Base):
    def similar(self, smiles, cutoff=70, limit=15):
        return []

    def find(self, name):
        self._chk()
        d = self._d(name)
        return d["chembl"] if d else None

    def _by_id(self, cid):
        return next((d for d in self.w.values() if d["chembl"]["chembl_id"] == cid), None)

    def mechanisms(self, cid):
        self._chk()
        d = self._by_id(cid)
        return d["mechanisms"] if d else []

    def target(self, tid):
        self._chk()
        for d in self.w.values():
            if tid in d["targets"]:
                return d["targets"][tid]
        return None

    def indications(self, cid):
        return []


class FakeUniProt(_Base):
    def protein(self, acc):
        self._chk()
        for d in self.w.values():
            if acc in d["uniprot"]:
                return d["uniprot"][acc]
        return None


class FakeOT:
    def adverse_events(self, cid):
        return []

    def mechanisms(self, cid):
        return []


class FakeOpenFDA(_Base):
    def label(self, name):
        self._chk()
        d = self._d(name)
        return d["label"] if d else None


class FakeEuropePMC(_Base):
    interaction_query = staticmethod(EuropePMC.interaction_query)

    def __init__(self, world, papers=None, fail=False):
        super().__init__(world, fail)
        self.papers, self.queries = papers or [], []

    def search(self, query, page_size=25):
        self.queries.append(query)
        # only answer if both drug names are in the query, like the real search would
        ql = query.lower()
        if self.fail:
            return []
        return [p for p in self.papers if all(n in ql for n in p.get("_needs", []))]


class FakeString:
    def __init__(self, edges=None):
        self.edges = edges or []

    def network(self, genes, **kw):
        return self.edges


class FakeCT:
    def studies(self, drug, limit=20):
        return []


class FakeServices:
    def __init__(self, papers=None, fail=False, llm=None, edges=None, world=None):
        w = world or WORLD
        self.pubchem, self.chembl, self.uniprot = FakePubChem(w, fail), FakeChEMBL(w, fail), FakeUniProt(w, fail)
        self.opentargets, self.openfda = FakeOT(), FakeOpenFDA(w, fail)
        self.europepmc = FakeEuropePMC(w, papers, fail)
        self.string, self.ctgov, self.llm = FakeString(edges), FakeCT(), llm or FakeLLM()
        import tempfile
        from types import SimpleNamespace
        self.settings = SimpleNamespace(cache_dir=Path(tempfile.mkdtemp(prefix="ps_fake_cache_")))
