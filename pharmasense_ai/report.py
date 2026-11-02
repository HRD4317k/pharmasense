"""Turn the final graph state into a report that a person can audit: every claim carries evidence ids, and every
evidence id resolves to a source sentence and a link. Also renders Markdown."""
from __future__ import annotations

import datetime as dt

from .models import DrugProfile, EvidenceStore, Hypothesis

STANDING_CAVEATS = [
    "FAERS reports are voluntary and unverified: they show what was reported, not how often something happens, and an association is not proof of causation.",
    "Mechanisms are hypotheses assembled from labels, curated databases and abstracts. A 'supported' verdict means the documents agree, not that an experiment confirmed it.",
    "Enzyme roles are text-mined from FDA labels with rules. `python -m pharmasense_ai validate` checks recall against a small hand-written list; precision has not been measured, so check the quoted sentence.",
    "Confounding by indication, co-medication outside the listed drugs, and stimulated reporting can each create or hide an apparent interaction.",
    "Research tool only. Not medical advice.",
]


def _ev(store: EvidenceStore, eid: str) -> dict:
    e = store.get(eid)
    return {"id": eid, "kind": e.kind, "source": e.source, "ref": e.ref, "section": e.section, "group": e.group, "text": e.text} if e else {"id": eid}


def drug_card(p: DrugProfile) -> dict:
    return {
        "smiles": p.smiles, "structure": p.structure, "chembl_id": p.chembl_id,
        "targets": [{"name": t.name, "action": t.action_type, "genes": t.genes, "n_pathways": len(t.pathways), "evidence": t.evidence} for t in p.targets],
        "enzyme_roles": [{"enzyme": r.enzyme, "role": r.role, "confidence": r.confidence, "evidence": r.evidence} for r in p.roles],
        "safety_axes": sorted(p.axes), "classes": p.classes, "label_url": (p.label or {}).get("url"),
        "faers_adverse_events_open_targets": p.ot_adverse_events[:8], "indications": p.indications,
        "human_studies": {"n_listed": len(p.trials), "examples": p.trials[:5]},
        "gaps": p.gaps, "sources": p.sources,
    }


def build_report(query: dict, profiles: dict, hyps: list[Hypothesis], store: EvidenceStore, warnings: list, lit_counts: dict,
                 trace: list, llm_used: bool, calibrated: bool, narrative: str) -> dict:
    cited = set()
    out_h = []
    for h in hyps:
        for e in h.supporting + h.contradicting + h.reaction_evidence:
            cited.add(e)
        out_h.append({
            "id": h.id, "mechanism_type": h.mechanism_type, "title": h.title, "claim": h.claim, "drugs": h.drugs,
            "victim": h.victim, "perpetrator": h.perpetrator, "term": h.term, "direction": h.direction, "origin": h.origin,
            "posterior": round(h.posterior, 3), "interval_90": [round(h.interval[0], 3), round(h.interval[1], 3)],
            "prior": h.prior, "verdict": h.verdict, "explains_reaction": h.explains_reaction,
            "supporting": h.supporting, "contradicting": h.contradicting, "reaction_evidence": h.reaction_evidence,
            "gaps": h.gaps, "alternatives": h.alternatives, "conformal": h.conformal})
    srcs = sorted({s for p in profiles.values() for s in p.sources})
    return {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "query": {k: v for k, v in query.items() if k != "faers"},
        "faers": query.get("faers"), "drugs": {n: drug_card(p) for n, p in profiles.items()}, "hypotheses": out_h,
        "evidence": {e: _ev(store, e) for e in sorted(cited)}, "sources_used": srcs,
        "sources_unavailable": sorted({g for p in profiles.values() for g in p.gaps}), "literature_abstracts": {f"{a} + {b}": n for (a, b), n in lit_counts.items()},
        "warnings": warnings, "trace": trace, "llm_used": llm_used,
        "calibration": "calibrated" if calibrated else "uncalibrated: evidence weights are defaults (run `python -m pharmasense_ai validate`)",
        "narrative": narrative, "caveats": STANDING_CAVEATS}


def default_narrative(hyps: list[Hypothesis], reaction: str | None) -> str:
    if not hyps:
        return ("No mechanism could be proposed: the profiles share no enzyme, target, pathway or safety axis, and neither label "
                "names the other drug. This is 'no evidence found', not 'no interaction'.")
    top = [h for h in hyps if h.verdict in ("supported", "contested", "untested")][:3] or hyps[:3]
    parts = []
    for h in top:
        cites = "".join(f"[{e}]" for e in (h.supporting[:3]))
        lo, hi = h.interval
        parts.append(f"{h.id} ({h.verdict}, posterior {h.posterior:.0%}, 90% interval {lo:.0%}-{hi:.0%}): {h.claim} {cites}".strip())
    tail = ""
    if reaction:
        explained = [h.id for h in hyps if h.explains_reaction]
        tail = (f" Hypotheses whose drug labels connect them to '{reaction.lower()}': {', '.join(explained)}." if explained else
                f" None of the hypotheses is linked to '{reaction.lower()}' by a label statement.")
    return " ".join(parts) + tail


def render_markdown(r: dict) -> str:
    L = [f"# Mechanism report: {' + '.join(r['query']['drugs'])}" + (f" -> {r['query']['reaction']}" if r["query"].get("reaction") else ""), "",
         f"*Generated {r['generated']} - {r['calibration']}*", "", r["narrative"], ""]
    if r.get("faers"):
        f = r["faers"]
        L += ["## FAERS context", f"{f.get('verdict')} - ROR {f.get('ror', float('nan')):.1f} (95% CI {f.get('ror_lcl', float('nan')):.1f}-{f.get('ror_ucl', float('nan')):.1f}), source: {f.get('source', 'local sample')}", ""]
    L.append("## Hypotheses")
    for h in r["hypotheses"]:
        L += [f"### {h['id']} - {h['title']}", f"**{h['verdict'].upper()}** - posterior {h['posterior']:.0%} (90% interval {h['interval_90'][0]:.0%}-{h['interval_90'][1]:.0%}; prior {h['prior']:.0%}) - type: {h['mechanism_type']}"
              + (f" - explains the reaction: {h['explains_reaction']}" if h["explains_reaction"] is not None else ""), "", h["claim"], ""]
        for title, ids in (("Supporting", h["supporting"]), ("Contradicting", h["contradicting"])):
            if ids:
                L.append(f"*{title}*")
                for e in ids:
                    ev = r["evidence"].get(e, {"id": e})
                    L.append(f"- [{e}] ({ev.get('kind')}) {ev.get('text', '')[:300]} - {ev.get('source', '')} {ev.get('ref', '')}")
                L.append("")
        if h["gaps"]:
            L += ["*Missing evidence*"] + [f"- {g}" for g in h["gaps"]] + [""]
        if h["alternatives"]:
            L += ["*Alternative explanations*"] + [f"- {a}" for a in h["alternatives"]] + [""]
    L += ["## Sources", ", ".join(r["sources_used"]) or "none reachable", ""]
    if r["sources_unavailable"]:
        L += ["## Not available"] + [f"- {g}" for g in r["sources_unavailable"]] + [""]
    L += ["## Caveats"] + [f"- {c}" for c in r["caveats"]]
    return "\n".join(L)
