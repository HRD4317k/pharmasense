"""DISCOVER: find relationships that are NOT already recorded, and say exactly what "not recorded" was checked against.

Two routes (each candidate then goes through the Explain + Challenge workflow):
  1. FAERS signal vs documentation   a statistically clear combination signal whose reaction appears in no component
     label and whose drugs are not named in each other's labels, and that no retrieved abstract links -> "candidate novel"
  2. Knowledge-graph link prediction  drug -> reaction links ranked by personalised PageRank over the profile graph,
     evaluated by hiding edges first (`KnowledgeGraph.holdout_auc`)

"Novel" only ever means "not found in: FDA labels of the component drugs, the Europe PMC abstracts retrieved". It does not
mean unknown to science. Absence of evidence is reported as such.
"""
from __future__ import annotations

import pandas as pd

from . import cluster_check, reasoning as rs
from .agents.workflow import build_graph
from .models import EvidenceStore

COMBINATION_LIKE = ("Combination effect", "Probable combination", "Pair effect")


def candidates_from_results(triplets: pd.DataFrame, reports: pd.DataFrame | None, verdicts=COMBINATION_LIKE, top: int = 20) -> pd.DataFrame:
    T = triplets[triplets["verdict"].isin(verdicts)].sort_values("ror_lcl", ascending=False).head(top)
    return cluster_check.annotate(T, reports) if reports is not None and len(T) else T


def documentation_status(report: dict) -> dict:
    """Is the relationship already documented? Uses only evidence that was actually retrieved."""
    hyps = report["hypotheses"]
    ev = report["evidence"]
    kinds_by_h = [{ev[e]["kind"] for e in h["supporting"] if e in ev} for h in hyps]
    label_pair = any("label_ddi" in k for k in kinds_by_h)
    literature = any("literature" in k for k in kinds_by_h)
    reaction = report["query"].get("reaction")
    label_reaction = any(h["explains_reaction"] for h in hyps) if reaction else None
    if label_pair or literature:
        status = "documented"
    elif label_reaction:
        status = "reaction known for a component drug; interaction undocumented"
    else:
        status = "candidate novel"
    return {"status": status, "label_names_partner": label_pair, "in_literature": literature, "label_lists_reaction": label_reaction}


def run_discovery(svc, cand: pd.DataFrame, max_candidates: int = 10, log=print) -> pd.DataFrame:
    """Explain + Challenge each candidate with ONE shared evidence store and profile cache (drugs recur across triplets)."""
    store, cache = EvidenceStore(), {}
    rows = []
    for r in cand.head(max_candidates).itertuples():
        drugs = [r.drug_a.title(), r.drug_b.title(), r.drug_c.title()]
        faers = {"verdict": r.verdict, "ror": r.ror, "ror_lcl": r.ror_lcl, "ror_ucl": r.ror_ucl, "n_events": int(r.a_ABC), "n_reports": int(r.n_ABC),
                 "driver_drugs": r.driver_drugs, "flags": getattr(r, "cluster_flag", "") or r.flags, "source": "local sample"}
        app, _ = build_graph(svc, store=store, cache=cache)
        out = app.invoke({"query": {"drugs": drugs, "reaction": r.reaction, "faers": faers}, "loops": 0, "warnings": [], "trace": []})
        rep = out["report"]
        doc = documentation_status(rep)
        best = rep["hypotheses"][0] if rep["hypotheses"] else None
        rows.append({"drugs": " + ".join(drugs), "reaction": r.reaction, "faers_verdict": r.verdict, "ror_lcl": r.ror_lcl,
                     "cluster_flag": getattr(r, "cluster_flag", ""), "documentation": doc["status"],
                     "best_hypothesis": best["title"] if best else "none found", "best_verdict": best["verdict"] if best else "",
                     "posterior": best["posterior"] if best else float("nan"), "n_gaps": sum(len(h["gaps"]) for h in rep["hypotheses"]),
                     "report": rep})
        log(f"  {rows[-1]['drugs']} -> {r.reaction}: {doc['status']}; best mechanism: {rows[-1]['best_hypothesis']} ({rows[-1]['best_verdict']})")
    out = pd.DataFrame(rows)
    order = {"candidate novel": 0, "reaction known for a component drug; interaction undocumented": 1, "documented": 2}
    return out.assign(_o=out["documentation"].map(order)).sort_values(["_o", "posterior"], ascending=[True, False]).drop(columns="_o") if len(out) else out
