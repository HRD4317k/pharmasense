"""End-to-end tests of Discover/Explain/Challenge logic with offline fakes.  Run: python -m pytest tests -q"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fakes import PAPERS_NULL, PAPERS_SUPPORT, FakeLLM, FakeServices  # noqa: E402

from pharmasense_ai import report as rp  # noqa: E402
from pharmasense_ai.agents.workflow import build_graph  # noqa: E402

FAERS_COMBO = {"verdict": "Combination effect", "ror": 12.0, "ror_lcl": 4.0, "ror_ucl": 36.0, "n_events": 9, "n_reports": 40,
               "source": "test sample", "driver_drugs": "WARFARIN | FLUCONAZOLE"}


def run(drugs, reaction=None, faers=None, svc=None, **kw):
    svc = svc or FakeServices(papers=[{**PAPERS_SUPPORT, "_needs": ["warfarin", "fluconazole"]}])
    app, store = build_graph(svc, **kw)
    out = app.invoke({"query": {"drugs": drugs, "reaction": reaction, "faers": faers}, "loops": 0, "warnings": [], "trace": []})
    return out["report"], svc, store


def hyp(rep, mtype, term=None):
    return next(h for h in rep["hypotheses"] if h["mechanism_type"] == mtype and (term is None or h.get("title", "").find(term) >= 0))


# ------------------------------------------------------------------ Explain
def test_known_pk_interaction_is_found_and_supported():
    rep, _, _ = run(["warfarin", "fluconazole"], "INTERNATIONAL NORMALISED RATIO INCREASED", FAERS_COMBO)
    h = hyp(rep, "PK-enzyme", "CYP2C9")
    assert h["perpetrator"] == "fluconazole" and h["victim"] == "warfarin"
    assert h["verdict"] == "supported" and h["posterior"] > 0.8
    assert h["interval_90"][0] < h["posterior"] < h["interval_90"][1]
    assert h["explains_reaction"] is True                       # warfarin's label ties it to bleeding / INR
    kinds = {rep["evidence"][e]["kind"] for e in h["supporting"]}
    assert {"label_class", "label_ddi", "literature", "faers"} <= kinds


def test_every_cited_evidence_id_resolves_to_a_source():
    rep, _, _ = run(["warfarin", "fluconazole"], "INTERNATIONAL NORMALISED RATIO INCREASED", FAERS_COMBO)
    for h in rep["hypotheses"]:
        for e in h["supporting"] + h["contradicting"] + h["reaction_evidence"]:
            ev = rep["evidence"][e]
            assert ev["text"] and ev["source"] and ev["ref"]
    md = rp.render_markdown(rep)
    assert "Caveats" in md and "CYP2C9" in md


def test_mechanism_negative_control_is_not_invented():
    """Pravastatin is barely metabolised by CYP, so ketoconazole (a CYP3A4 inhibitor) must not yield a PK hypothesis."""
    rep, _, _ = run(["pravastatin", "ketoconazole"])
    assert not [h for h in rep["hypotheses"] if h["mechanism_type"] in ("PK-enzyme", "PK-transporter")]


def test_pd_shared_target_found_for_invented_drugs():
    rep, _, _ = run(["alphadrug", "betadrug"], svc=FakeServices())
    h = hyp(rep, "PD-shared-target")
    assert "PTGS2" in h["title"] and "effects add" in h["claim"]


# ------------------------------------------------------------------ Challenge
def test_contradicting_literature_turns_supported_into_contested():
    papers = [{**PAPERS_SUPPORT, "_needs": ["warfarin", "fluconazole"]}, {**PAPERS_NULL, "_needs": ["warfarin", "fluconazole"]}]
    rep, _, _ = run(["warfarin", "fluconazole"], "INTERNATIONAL NORMALISED RATIO INCREASED", FAERS_COMBO, FakeServices(papers=papers))
    h = hyp(rep, "PK-enzyme", "CYP2C9")
    assert h["contradicting"], "the null-result abstract must be recorded against the hypothesis"
    assert h["verdict"] == "contested"
    assert any(rep["evidence"][e]["kind"] == "contradiction" for e in h["contradicting"])


def test_faers_single_drug_effect_counts_against_interaction():
    faers = {**FAERS_COMBO, "verdict": "Single-drug effect", "driver_drugs": "WARFARIN"}
    rep, _, _ = run(["warfarin", "fluconazole"], "INTERNATIONAL NORMALISED RATIO INCREASED", faers)
    h = hyp(rep, "PK-enzyme", "CYP2C9")
    assert any(rep["evidence"][e]["kind"] == "faers_against" for e in h["contradicting"])
    assert any("WARFARIN" in a for a in h["alternatives"])


def test_missing_evidence_is_reported_not_hidden():
    rep, _, _ = run(["warfarin", "fluconazole"], "HEADACHE", None, FakeServices(papers=[]))
    h = hyp(rep, "PK-enzyme", "CYP2C9")
    text = " ".join(h["gaps"])
    assert "No FAERS evidence" in text and "no abstracts" in text.lower()
    assert h["explains_reaction"] is False and "does not list headache" in text.lower()


def test_untested_hypothesis_triggers_a_targeted_second_search():
    svc = FakeServices(papers=[])
    rep, svc, _ = run(["alphadrug", "betadrug"], svc=svc)
    h = hyp(rep, "PD-shared-target")
    assert h["verdict"] == "untested"
    assert len(svc.europepmc.queries) == 2                          # first pass + targeted second pass (max_loops=1)
    assert "PTGS2" in svc.europepmc.queries[1] and "PTGS2" not in svc.europepmc.queries[0]
    assert any("searching the literature again" in t for t in rep["trace"])


# ------------------------------------------------------------------ robustness
def test_total_outage_degrades_to_gaps_not_a_crash():
    rep, _, _ = run(["warfarin", "fluconazole"], "HEADACHE", None, FakeServices(fail=True))
    assert rep["hypotheses"] == []
    assert "No mechanism could be proposed" in rep["narrative"] and "not 'no interaction'" in rep["narrative"]
    assert rep["sources_unavailable"] and rep["warnings"]


def test_report_is_json_serialisable():
    rep, _, _ = run(["warfarin", "fluconazole"], "INTERNATIONAL NORMALISED RATIO INCREASED", FAERS_COMBO)
    json.dumps(rep, default=str)


# ------------------------------------------------------------------ the LLM cannot add unsupported facts
def test_llm_hypothesis_without_valid_evidence_is_rejected():
    good = json.dumps([{"title": "Additive bleeding", "claim": "Both raise bleeding risk.", "evidence_ids": ["E001"]}])
    bad = json.dumps([{"title": "Made up mechanism", "claim": "Unsupported.", "evidence_ids": ["E999"]}])
    llm = FakeLLM([bad, json.dumps({}), "Not cited at all."])
    rep, _, store = run(["warfarin", "fluconazole"], svc=FakeServices(papers=[], llm=llm))
    assert not [h for h in rep["hypotheses"] if h["origin"] == "llm-proposed"]
    assert rep["llm_rejected_outputs"] >= 2                            # bad hypothesis + uncited narrative

    llm = FakeLLM([good, json.dumps({}), "Bleeding risk may add up [E001]."])
    rep, _, _ = run(["warfarin", "fluconazole"], svc=FakeServices(papers=[], llm=llm))
    added = [h for h in rep["hypotheses"] if h["origin"] == "llm-proposed"]
    assert added and added[0]["supporting"] == ["E001"]
    assert added[0]["posterior"] < 0.5                                  # probability comes from evidence + low prior, not from the model
    assert rep["narrative"] == "Bleeding risk may add up [E001]."


def test_extract_json_handles_lists_objects_and_fences():
    from pharmasense_ai.llm import extract_json
    assert extract_json('[{"a": 1}, {"a": 2}]') == [{"a": 1}, {"a": 2}]            # list of objects stays a list
    assert extract_json('Sure! ```json\n{"H01": {"missing": ["x"]}}\n``` hope that helps') == {"H01": {"missing": ["x"]}}
    assert extract_json('prefix [1, 2, 3] suffix') == [1, 2, 3]
    assert extract_json("no json here") is None and extract_json(None) is None
