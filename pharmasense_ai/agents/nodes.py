"""LangGraph nodes. Each node is a small function over the shared state; the deterministic logic lives in
`reasoning.py` / `profile.py` so it can be tested without the graph.

LLM use is optional and fenced: the model may *propose* hypotheses and *suggest* gaps/alternatives, but
  - a proposed hypothesis must cite evidence ids that exist (otherwise it is dropped and counted), and its
    probability still comes only from that evidence;
  - critic suggestions are shown as "model-suggested" text and never change a number;
  - the narrative is rejected if it cites an id that does not exist.
"""
from __future__ import annotations

import re

from .. import reasoning as rs
from ..llm import extract_json
from ..models import EvidenceStore, Hypothesis
from ..profile import build_profile
from ..report import build_report, default_narrative

SYSTEM = ("You are a careful clinical pharmacology assistant. Use ONLY the evidence items provided. "
          "Cite evidence ids like E012. If the evidence does not support a statement, say nothing. Reply with JSON only.")


def make_nodes(svc, store: EvidenceStore, table: dict | None = None, conformal=None, max_loops: int = 1, cache: dict | None = None):
    # `cache` may be shared across runs (with the SAME store, since profiles hold evidence ids that live in the store)
    cache = cache if cache is not None else {}
    stats = {"llm_rejected": 0, "llm_added": 0}

    # ---------------------------------------------------------------- profile
    def profile(state):
        q = state["query"]
        profiles, warns = {}, []
        for d in q["drugs"]:
            key = d.lower().strip()
            if key not in cache:
                cache[key] = build_profile(d.strip(), svc, store)
            p = cache[key]
            profiles[d] = p
            warns += [f"{d}: {g}" for g in p.gaps]
        n_t = sum(len(p.targets) for p in profiles.values())
        n_r = sum(len(p.roles) for p in profiles.values())
        return {"profiles": profiles, "warnings": warns,
                "trace": [f"profile: {len(profiles)} drugs, {n_t} targets, {n_r} enzyme/transporter roles, "
                          f"{sum(len(p.axes) for p in profiles.values())} safety axes"]}

    # ---------------------------------------------------------------- protein network
    def network(state):
        genes = sorted({g for p in state["profiles"].values() for t in p.targets for g in t.genes})
        edges = svc.string.network(genes) if len(genes) >= 2 else []
        return {"string_edges": edges, "trace": [f"network: {len(genes)} target genes, {len(edges)} STRING links"]}

    # ---------------------------------------------------------------- hypotheses
    def hypothesize(state):
        q = state["query"]
        hyps = rs.generate(state["profiles"], store, q.get("reaction"), state.get("string_edges"))
        msg = f"hypothesize: {len(hyps)} rule-based hypotheses"
        if svc.llm.ready and hyps is not None:
            added = _llm_propose(state, hyps)
            msg += f"; model proposed {added} more (rejected {stats['llm_rejected']} that cited no valid evidence)"
        return {"hypotheses": hyps, "trace": [msg]}

    def _llm_propose(state, hyps) -> int:
        ev = [{"id": e.id, "kind": e.kind, "drugs": e.drugs, "text": e.text[:220]} for e in list(store.items.values())[:60]]
        out = extract_json(svc.llm.chat(SYSTEM, "Drugs: %s. Reaction: %s.\nEvidence:\n%s\nExisting hypotheses: %s\n"
                                        "Propose up to 3 ADDITIONAL mechanisms by which these drugs could interact or cause the reaction. "
                                        'Return a JSON list of {"title","claim","evidence_ids"}; evidence_ids must come from the list above; '
                                        "return [] if nothing is supported." % (state["query"]["drugs"], state["query"].get("reaction"),
                                                                              ev, [h.title for h in hyps])))
        n = 0
        for item in out if isinstance(out, list) else []:
            ids = [i for i in (item.get("evidence_ids") or []) if store.get(i)]
            if not ids or not item.get("title"):
                stats["llm_rejected"] += 1
                continue
            drugs = state["query"]["drugs"][:2] if len(state["query"]["drugs"]) >= 2 else state["query"]["drugs"]
            h = Hypothesis(id=f"H{len(hyps) + 1:02d}", mechanism_type="llm-proposed", title=str(item["title"])[:160],
                           claim=str(item.get("claim", item["title"]))[:400], drugs=list(drugs), prior=0.05, origin="llm-proposed")
            h.supporting = ids
            hyps.append(h)
            n += 1
        stats["llm_added"] += n
        return n

    # ---------------------------------------------------------------- retrieval (literature + FAERS)
    def retrieve(state):
        hyps = state["hypotheses"]
        loops = state.get("loops", 0)
        extra = {}
        if loops:                                           # second pass: targeted terms for hypotheses still untested
            for h in hyps:
                if h.verdict == "untested" and h.term and h.mechanism_type != "documented":
                    key = tuple(h.drugs)
                    extra[key] = (extra.get(key, "") + (" OR " if key in extra else "") + f'"{h.term}"')
        counts = rs.attach_literature(hyps, state["profiles"], svc.europepmc, store, extra=extra)
        if not loops:
            rs.attach_faers(hyps, state["query"].get("faers"), store)
        n_lit = sum(len([e for e in h.supporting if store.get(e) and store.get(e).kind == "literature"]) for h in hyps)
        return {"lit_counts": counts, "hypotheses": hyps,
                "trace": [f"retrieve{' (targeted second pass)' if loops else ''}: {sum(counts.values())} abstracts, {n_lit} supporting sentences kept"]}

    # ---------------------------------------------------------------- challenge
    def challenge(state):
        hyps, q = state["hypotheses"], state["query"]
        rs.finalize(hyps, state["profiles"], store, q.get("reaction"), state.get("lit_counts", {}), q.get("faers"))
        n_con = sum(len(h.contradicting) for h in hyps)
        n_gap = sum(len(h.gaps) for h in hyps)
        return {"hypotheses": hyps, "trace": [f"challenge: {n_con} contradicting items, {n_gap} recorded gaps"]}

    def critic(state):
        """Optional model critique, run ONCE after scoring (not on every loop pass). Text only - never changes a number."""
        hyps = state["hypotheses"]
        if not (svc.llm.ready and hyps):
            return {"trace": ["critic: skipped (no LLM configured)"]}
        top = hyps[:4]                                          # already sorted by posterior
        pack = [{"id": h.id, "claim": h.claim, "support": [{"id": e, "text": store.get(e).text[:200]} for e in h.supporting[:5] if store.get(e)]} for h in top]
        out = extract_json(svc.llm.chat(SYSTEM, "Critique these mechanism hypotheses. For each give 'missing' (evidence a careful reviewer "
                                                "would still want) and 'alternatives' (other explanations for the same observation). "
                                                'Return {"H01": {"missing": [...], "alternatives": [...]}}.\n' + str(pack)))
        added = 0
        for h in top:
            c = out.get(h.id) if isinstance(out, dict) else None
            if isinstance(c, dict):
                m = [f"(model-suggested) {x}" for x in (c.get("missing") or [])[:2] if isinstance(x, str)]
                a = [f"(model-suggested) {x}" for x in (c.get("alternatives") or [])[:2] if isinstance(x, str)]
                h.gaps += m
                h.alternatives += a
                added += len(m) + len(a)
        return {"hypotheses": hyps, "trace": [f"critic: {added} model-suggested gaps/alternatives (text only)"]}

    # ---------------------------------------------------------------- score
    def score(state):
        hyps = state["hypotheses"]
        rs.score(hyps, store, table, conformal=conformal)
        top = hyps[0] if hyps else None
        return {"hypotheses": hyps, "trace": [f"score: {len(hyps)} hypotheses; top = {top.id} {top.verdict} {top.posterior:.0%}" if top else "score: no hypotheses"]}

    def route(state):
        hyps = state.get("hypotheses") or []
        weak = [h for h in hyps if h.verdict == "untested" and h.term and h.mechanism_type != "documented"
                and not any(store.get(e) and store.get(e).kind == "literature" for e in h.supporting)]
        return "retrieve" if (weak and state.get("loops", 0) < max_loops) else "critic"

    def bump(state):
        return {"loops": state.get("loops", 0) + 1, "trace": ["loop: some hypotheses are untested - searching the literature again with their specific terms"]}

    # ---------------------------------------------------------------- report
    def report(state):
        q, hyps = state["query"], state["hypotheses"]
        narrative = default_narrative(hyps, q.get("reaction"))
        used_llm = False
        if svc.llm.ready and hyps:
            txt = svc.llm.chat(SYSTEM.replace("Reply with JSON only.", "Write 3-5 plain sentences."),
                               "Summarise for a pharmacist which mechanisms are best supported and what is missing. Every sentence must end with "
                               "citation ids in square brackets, e.g. [E003][E010].\nHypotheses: %s\nEvidence: %s" % (
                                   [(h.id, h.verdict, round(h.posterior, 2), h.claim, h.supporting[:3]) for h in hyps[:4]],
                                   {e: store.get(e).text[:180] for h in hyps[:4] for e in h.supporting[:3] if store.get(e)}))
            if txt:
                cites = re.findall(r"\[(E\d+)\]", txt)
                sentences = [s for s in re.split(r"(?<=[.!?])\s+", txt.strip()) if s]
                if cites and all(store.get(c) for c in cites) and all(re.search(r"\[E\d+\]", s) for s in sentences):
                    narrative, used_llm = txt.strip(), True
                else:
                    stats["llm_rejected"] += 1
        rep = build_report(q, state["profiles"], hyps, store, state.get("warnings", []), state.get("lit_counts", {}),
                           state.get("trace", []), used_llm or stats["llm_added"] > 0, bool(table and "_calibrated" in table), narrative)
        rep["llm_rejected_outputs"] = stats["llm_rejected"]
        return {"report": rep, "trace": ["report: assembled"]}

    return {"profile": profile, "network": network, "hypothesize": hypothesize, "retrieve": retrieve, "challenge": challenge,
            "score": score, "critic": critic, "bump": bump, "report": report, "route": route}
