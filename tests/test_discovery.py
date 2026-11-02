"""Offline tests for the knowledge graph, structural analogues and the discovery loop."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fakes import PAPERS_SUPPORT, FakeServices  # noqa: E402

from pharmasense_ai import discover, similar  # noqa: E402
from pharmasense_ai.kg import KnowledgeGraph  # noqa: E402
from pharmasense_ai.models import EvidenceStore  # noqa: E402
from pharmasense_ai.profile import build_profile  # noqa: E402


def _kg():
    svc, store = FakeServices(), EvidenceStore()
    kg = KnowledgeGraph()
    for d in ("warfarin", "fluconazole", "pravastatin"):
        kg.add_profile(build_profile(d, svc, store))
    return kg


def test_kg_has_typed_nodes_and_enzyme_bridge():
    kg = _kg()
    kinds = kg.stats()["by_kind"]
    assert kinds.get("drug") == 3 and kinds.get("enzyme", 0) >= 1
    paths = kg.paths("drug:FLUCONAZOLE", "drug:WARFARIN", max_hops=3)
    assert paths and any("CYP2C9" in p["nodes"] for p in paths)          # the shared enzyme is the bridge
    assert all(0 < p["score"] <= 1 for p in paths)
    assert paths == sorted(paths, key=lambda p: -p["score"])


def test_noisy_or_is_bounded():
    kg = _kg()
    s = kg.path_support(kg.paths("drug:FLUCONAZOLE", "drug:WARFARIN", max_hops=3))
    assert 0 < s < 1


def test_link_prediction_never_returns_existing_edges():
    kg = _kg()
    stats = pd.DataFrame({"drug": ["WARFARIN", "FLUCONAZOLE", "FLUCONAZOLE", "PRAVASTATIN"], "reaction": ["HAEMORRHAGE", "QT PROLONGED", "HAEMORRHAGE", "MYALGIA"],
                          "a": [20, 15, 6, 9], "ror": [5.0, 4.0, 2.0, 3.0], "signal": [True] * 4})
    kg.add_faers(stats)
    for r in kg.predict_links("fluconazole", "reaction"):
        assert not kg.G.has_edge("drug:FLUCONAZOLE", "reaction:" + r["node"].upper())
    assert kg.predict_links("not-a-drug") == []


def test_holdout_auc_on_structured_graph_beats_chance():
    # a graph with two dense communities: hidden edges are recoverable, so AUROC must be clearly above 0.5
    import networkx as nx
    G = nx.Graph()
    rng = np.random.default_rng(0)
    for c in range(2):
        nodes = [f"n{c}_{i}" for i in range(14)]
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                if rng.random() < 0.55:
                    G.add_edge(u, v, weight=1.0)
    kg = KnowledgeGraph()
    kg.G = G
    assert kg.holdout_auc("ppr", seed=1) > 0.7
    # and a random graph should hover near chance (not a guarantee for any one seed, so use a loose band)
    R = nx.gnp_random_graph(60, 0.1, seed=3)
    nx.set_edge_attributes(R, 1.0, "weight")
    kg2 = KnowledgeGraph()
    kg2.G = R
    assert 0.3 < kg2.holdout_auc("ppr", seed=2) < 0.7


def test_analogues_flag_inferred_not_observed():
    lib = {"ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "naproxen": "COc1ccc2cc(ccc2c1)C(C)C(=O)O", "ketoprofen": "CC(C(=O)O)c1cccc(c1)C(=O)c1ccccc1",
           "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C"}
    stats = pd.DataFrame({"drug": ["NAPROXEN"] * 2 + ["KETOPROFEN"] * 2 + ["IBUPROFEN"], "reaction": ["DYSPEPSIA", "OEDEMA"] * 2 + ["HEADACHE"],
                          "a": [9] * 5, "ror": [3.0] * 5, "ror_lcl": [2.0] * 5, "signal": [True] * 5})
    rep = similar.analog_report("ibuprofen", lib["ibuprofen"], lib, stats, top_k=3, min_sim=0.2)
    names = [a["drug"] for a in rep["analogues"]]
    assert "ibuprofen" not in names and set(names) & {"naproxen", "ketoprofen"}
    assert "caffeine" not in names[:2]                                      # an unrelated xanthine ranks below the profens
    inferred = {i["reaction"]: i for i in rep["inferred_from_analogues"]}
    assert {"DYSPEPSIA", "OEDEMA"} <= set(inferred) and all("not observed" in v["status"] for v in inferred.values())
    assert "HEADACHE" not in inferred                                       # the drug's own signal is observed, not inferred


def test_discovery_labels_documented_vs_undocumented():
    svc = FakeServices(papers=[{**PAPERS_SUPPORT, "_needs": ["warfarin", "fluconazole"]}])
    cand = pd.DataFrame([{"drug_a": "WARFARIN", "drug_b": "FLUCONAZOLE", "drug_c": "PRAVASTATIN", "reaction": "INTERNATIONAL NORMALISED RATIO INCREASED",
                          "verdict": "Probable combination", "ror": 9.0, "ror_lcl": 2.5, "ror_ucl": 30.0, "a_ABC": 6, "n_ABC": 20,
                          "driver_drugs": "WARFARIN | FLUCONAZOLE", "flags": ""}])
    out = discover.run_discovery(svc, cand, log=lambda *_: None)
    assert len(out) == 1 and out.iloc[0]["documentation"] in {"documented", "reaction known for a component drug; interaction undocumented", "candidate novel"}
    assert out.iloc[0]["documentation"] == "documented"                      # label_ddi / literature name warfarin + fluconazole
    assert isinstance(out.iloc[0]["report"], dict)
