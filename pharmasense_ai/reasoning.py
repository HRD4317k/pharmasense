"""The reasoning core (pure functions, no LangGraph, no network): generate competing mechanism hypotheses from drug
profiles, attach supporting AND contradicting evidence, record what is missing, and score them.

Hypothesis families (each needs concrete, citable facts from BOTH drugs - nothing is invented):
  PK-enzyme / PK-transporter   one drug is a substrate of an enzyme/transporter the other inhibits or induces
  PD-shared-target             both act on the same protein (UniProt accession)
  PD-effect-axis               both labels warn about the same safety axis (QT, bleeding, serotonin syndrome ...)
  PD-pathway                   targets sit in the same (non-generic) Reactome pathway
  indirect-network             targets are linked in the STRING protein network, nothing stronger
  documented                   one label names the other drug but gives no mechanism we could extract
"""
from __future__ import annotations

import itertools
import re

import numpy as np

from . import label_parse as lp
from .mathx import evidence as em
from .models import DrugProfile, EvidenceStore, Hypothesis
from .rag import BM25, literature_chunks

GENERIC_PATHWAYS = {"Metabolism", "Signal Transduction", "Disease", "Immune System", "Metabolism of proteins",
                    "Post-translational protein modification", "Gene expression (Transcription)", "Developmental Biology",
                    "Cell Cycle", "Cellular responses to stimuli", "Transport of small molecules", "Vesicle-mediated transport",
                    "Innate Immune System", "Adaptive Immune System", "Cytokine Signaling in Immune system"}
INHIBIT_LIKE = {"INHIBITOR", "ANTAGONIST", "BLOCKER", "NEGATIVE ALLOSTERIC MODULATOR", "INVERSE AGONIST", "DEGRADER",
                "NEGATIVE MODULATOR", "ANTISENSE INHIBITOR", "RNAI INHIBITOR"}
ACTIVATE_LIKE = {"AGONIST", "PARTIAL AGONIST", "ACTIVATOR", "POSITIVE ALLOSTERIC MODULATOR", "OPENER", "STIMULATOR",
                 "POSITIVE MODULATOR", "RELEASING AGENT"}
AUTHORITATIVE = {"label_ddi", "label_class"}        # one such source is enough to count as "supported"
KIND_MAP = {"label_text": "label_context"}
LABEL_LINK_SECTIONS = ("boxed_warning", "warnings", "warnings_and_cautions", "adverse_reactions", "precautions", "drug_interactions")
DDI_SECTIONS = ("drug_interactions", "contraindications", "warnings", "warnings_and_cautions", "precautions", "boxed_warning")
SUPPORTED_P, UNSUPPORTED_P = 0.60, 0.15


# ------------------------------------------------------------------ helpers
def _word(term: str):
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", re.I)


def mentions(text: str, profile: DrugProfile) -> bool:
    terms = set(profile.names) | {profile.name.lower()}
    return any(_word(t).search(text) for t in terms if len(t) >= 4)


def action_sign(action: str) -> int:
    a = (action or "").upper()
    return -1 if a in INHIBIT_LIKE else (1 if a in ACTIVATE_LIKE else 0)


def reaction_link(profile: DrugProfile, reaction: str | None, store: EvidenceStore) -> list[str]:
    """Evidence ids for label statements tying this drug to the reaction (by safety axis, else by term match)."""
    if not reaction:
        return []
    ids = []
    for axis in lp.reaction_axes(reaction):
        ids += profile.axes.get(axis, [])[:2]
    if not ids and profile.label:
        toks = [t for t in re.findall(r"[a-z]{4,}", reaction.lower()) if t not in ("increased", "decreased", "abnormal", "disorder")]
        if toks:
            for sec in LABEL_LINK_SECTIONS:
                for s in lp.sentences(profile.label["sections"].get(sec, "")):
                    low = s.lower()
                    if all(t in low for t in toks):
                        ids.append(store.add("label_text", f"openFDA label: {profile.name}", profile.label["url"], s[:500],
                                             [profile.name], sec, f"label:{profile.name}", reaction_link=True))
                        break
                if ids:
                    break
    return list(dict.fromkeys(ids))[:3]


def ddi_mentions(a: DrugProfile, b: DrugProfile, store: EvidenceStore):
    """Sentences in one label that name the other drug (or its class). Returns (support_ids, contradicting_ids)."""
    sup, con = [], []
    for owner, other in ((a, b), (b, a)):
        if not owner.label:
            continue
        terms = sorted(set(other.names) | {other.name.lower()})
        found = 0
        for sec in DDI_SECTIONS:
            for s in lp.find_mentions(owner.label["sections"].get(sec, ""), terms, limit=2):
                if found >= 3:
                    break
                found += 1
                negated = lp.is_negated(s)
                eid = store.add("contradiction" if negated else "label_ddi", f"openFDA label: {owner.name}", owner.label["url"], s,
                                [owner.name, other.name], sec, f"label:{owner.name}", weight=0.8 if negated else 1.0)
                (con if negated else sup).append(eid)
    return sup, con


# ------------------------------------------------------------------ generation
def generate(profiles: dict[str, DrugProfile], store: EvidenceStore, reaction: str | None = None,
             string_edges: list[dict] | None = None) -> list[Hypothesis]:
    hyps: list[Hypothesis] = []

    def new(mtype, title, claim, drugs, **kw) -> Hypothesis:
        h = Hypothesis(id=f"H{len(hyps) + 1:02d}", mechanism_type=mtype, title=title, claim=claim, drugs=list(drugs),
                       prior=em.PRIORS[mtype], **kw)
        hyps.append(h)
        return h

    for A, B in itertools.combinations(profiles, 2):
        pa, pb = profiles[A], profiles[B]
        before = len(hyps)
        # ---- pharmacokinetic: substrate x inhibitor/inducer of the same enzyme or transporter
        seen = set()
        for v, p in ((A, B), (B, A)):
            for rv in profiles[v].roles:
                if rv.role != "substrate":
                    continue
                for rp in profiles[p].roles:
                    if rp.enzyme != rv.enzyme or rp.role not in ("inhibitor", "inducer") or (v, p, rv.enzyme, rp.role) in seen:
                        continue
                    seen.add((v, p, rv.enzyme, rp.role))
                    up = rp.role == "inhibitor"
                    mt = "PK-transporter" if rv.enzyme in lp.TRANSPORTERS else "PK-enzyme"
                    h = new(mt, f"{p} {'inhibits' if up else 'induces'} {rv.enzyme} -> {v} exposure {'rises' if up else 'falls'}",
                            f"{p} {'inhibits' if up else 'induces'} {rv.enzyme}, which handles {v}; "
                            f"{v} exposure is therefore expected to {'increase' if up else 'decrease'}.",
                            (A, B), victim=v, perpetrator=p, term=rv.enzyme,
                            direction=f"{v} exposure {'increases' if up else 'decreases'}")
                    h.supporting += [rv.evidence, rp.evidence]
        # ---- pharmacodynamic: shared target
        ta = {u: t for t in pa.targets for u in t.uniprot}
        tb = {u: t for t in pb.targets for u in t.uniprot}
        shared_pathway_blocklist = set()
        for u in sorted(set(ta) & set(tb)):
            x, y = ta[u], tb[u]
            gene = (x.genes or [x.name])[0]
            sa, sb = action_sign(x.action_type), action_sign(y.action_type)
            how = ("both inhibit/antagonise it - effects add" if sa == sb == -1 else
                   "both activate it - effects add" if sa == sb == 1 else
                   "one activates and one blocks it - effects oppose" if sa * sb == -1 else "both act on it")
            h = new("PD-shared-target", f"{A} and {B} both act on {gene}", f"{A} and {B} both act on {gene} ({how}).",
                    (A, B), term=gene, direction=how)
            h.supporting += x.evidence + y.evidence
            shared_pathway_blocklist.update(p["id"] for p in x.pathways)
        # ---- pharmacodynamic: shared safety axis
        for axis in sorted(set(pa.axes) & set(pb.axes)):
            h = new("PD-effect-axis", f"Both labels warn about {axis.lower()}",
                    f"Both {A} and {B} carry a labelled {axis.lower()} warning; used together the risk may add up.",
                    (A, B), term=axis, direction="additive risk")
            h.supporting += pa.axes[axis][:2] + pb.axes[axis][:2]
        # ---- pathway-level (specific pathways only)
        pw_a = {p["id"]: p["name"] for t in pa.targets for p in t.pathways}
        pw_b = {p["id"]: p["name"] for t in pb.targets for p in t.pathways}
        common = [(i, n) for i, n in pw_a.items() if i in pw_b and n not in GENERIC_PATHWAYS and i not in shared_pathway_blocklist]
        for i, n in sorted(common, key=lambda x: -len(x[1]))[:2]:
            ev_a = [e for t in pa.targets if any(p["id"] == i for p in t.pathways) for e in t.evidence][:1]
            ev_b = [e for t in pb.targets if any(p["id"] == i for p in t.pathways) for e in t.evidence][:1]
            h = new("PD-pathway", f"Targets converge on the pathway '{n}'", f"{A} and {B} act on different proteins in the Reactome pathway '{n}'.",
                    (A, B), term=n, direction="convergent pathway effect")
            h.supporting += ev_a + ev_b
        # ---- protein-network proximity (weakest)
        if string_edges and not any(h.mechanism_type == "PD-shared-target" for h in hyps[before:]):
            ga = {g for t in pa.targets for g in t.genes}
            gb = {g for t in pb.targets for g in t.genes}
            for e in string_edges:
                if (e["a"] in ga and e["b"] in gb) or (e["b"] in ga and e["a"] in gb):
                    eid = store.add("network", "STRING protein network", f"https://string-db.org/network/{e['a']}%0d{e['b']}",
                                    f"STRING: {e['a']} and {e['b']} are functionally linked (combined score {e['score']:.2f})", [A, B],
                                    "network", f"string:{e['a']}-{e['b']}")
                    h = new("indirect-network", f"Targets {e['a']} and {e['b']} are linked in the protein network",
                            f"{A} (via {e['a'] if e['a'] in ga else e['b']}) and {B} (via {e['b'] if e['a'] in ga else e['a']}) may influence each other indirectly.",
                            (A, B), term=f"{e['a']}-{e['b']}", direction="indirect")
                    h.supporting.append(eid)
                    break
        # ---- what the labels themselves say about this pair
        sup, con = ddi_mentions(pa, pb, store)
        pair_h = hyps[before:]
        if not pair_h and sup:
            h = new("documented", f"{A} and {B}: a label names the other drug", "A label names the other drug; no mechanism could be extracted from the text.",
                    (A, B), term="", direction="documented")
            pair_h = [h]
        for h in pair_h:
            h.supporting += sup
            h.contradicting += con
    return hyps


# ------------------------------------------------------------------ literature
def _term_in(h: Hypothesis, text: str) -> bool:
    if h.mechanism_type in ("PK-enzyme", "PK-transporter"):
        return h.term in {lp.norm_enzyme(t) for t in lp.ENZ_RE.findall(text)}
    if h.mechanism_type == "PD-effect-axis":
        return bool(lp._AXIS_RE[h.term].search(text))
    if h.mechanism_type == "PD-shared-target":
        return bool(_word(h.term).search(text))
    if h.mechanism_type == "PD-pathway":
        return h.term.lower() in text.lower()
    return True                                             # network / documented: any joint mention of the two drugs


def attach_literature(hyps: list[Hypothesis], profiles: dict[str, DrugProfile], europepmc, store: EvidenceStore,
                      page_size: int = 25, extra: dict | None = None) -> dict:
    """Search Europe PMC once per drug pair; keep abstract sentences that name BOTH drugs. A sentence that also names the
    hypothesis' mechanism supports it; with a negation ("did not alter ...") it contradicts it."""
    counts = {}
    pairs = sorted({tuple(h.drugs) for h in hyps})
    for A, B in pairs:
        q = europepmc.interaction_query(A, B, (extra or {}).get((A, B), ""))
        papers = europepmc.search(q, page_size)
        counts[(A, B)] = len(papers)
        chunks = [c for c in literature_chunks(papers, [A, B]) if mentions(c.text, profiles[A]) and mentions(c.text, profiles[B])]
        if not chunks:
            continue
        index = BM25(chunks)
        pair_hyps = [h for h in hyps if tuple(h.drugs) == (A, B)]
        for h in pair_hyps:
            ranked = [c for _, c in index.search(f"{A} {B} {h.term} {h.direction}", k=len(chunks))]
            n_sup = n_con = 0
            for c in ranked:
                neg = lp.is_negated(c.text)
                has = _term_in(h, c.text)
                if has and not neg and n_sup < 3:
                    n_sup += 1
                    h.supporting.append(store.add("literature", c.source, c.ref, c.text, [A, B], "abstract", c.group, term=h.term))
                elif neg and n_con < 2:
                    n_con += 1
                    h.contradicting.append(store.add("contradiction", c.source, c.ref, c.text, [A, B], "abstract", c.group,
                                                     weight=1.0 if has else 0.6, term=h.term))
    return counts


# ------------------------------------------------------------------ FAERS
FAERS_COMBINATION = {"Combination effect", "Probable combination", "Pair effect"}


def attach_faers(hyps: list[Hypothesis], faers: dict | None, store: EvidenceStore) -> None:
    if not faers:
        for h in hyps:
            h.pre_gaps.append("No FAERS evidence was supplied for this combination.")
        return
    v = faers.get("verdict", "")
    txt = (f"FAERS ({faers.get('source', 'local sample')}): {v}. Reporting odds ratio {faers.get('ror', float('nan')):.1f} "
           f"(95% CI {faers.get('ror_lcl', float('nan')):.1f}-{faers.get('ror_ucl', float('nan')):.1f}); "
           f"{faers.get('n_events', '?')} reports with the reaction among {faers.get('n_reports', '?')} listing all the drugs.")
    drivers = {d.strip().upper() for d in str(faers.get("driver_drugs") or "").split("|") if d.strip()}
    for h in hyps:
        involved = {d.upper() for d in h.drugs}
        if v in FAERS_COMBINATION and (v != "Pair effect" or not drivers or involved == drivers):
            h.supporting.append(store.add("faers", f"FAERS {faers.get('source', 'local sample')}", "https://open.fda.gov/data/faers/", txt, h.drugs,
                                          "faers", "faers", ror_lcl=faers.get("ror_lcl")))
        elif v == "Single-drug effect":
            h.contradicting.append(store.add("faers_against", f"FAERS {faers.get('source', 'local sample')}", "https://open.fda.gov/data/faers/",
                                             txt + " One drug on its own already explains the signal.", h.drugs, "faers", "faers"))
        else:
            h.pre_gaps.append(f"FAERS cannot separate the drugs here (verdict: {v or 'none'}).")
        if faers.get("flags"):
            h.pre_gaps.append("FAERS caveat: " + str(faers["flags"]))


# ------------------------------------------------------------------ gaps, alternatives, scoring
def finalize(hyps: list[Hypothesis], profiles: dict[str, DrugProfile], store: EvidenceStore, reaction: str | None,
             lit_counts: dict, faers: dict | None) -> None:
    for h in hyps:
        h.gaps = list(h.pre_gaps)                         # recomputed on every pass, so a loop never duplicates
        for d in h.drugs:
            p = profiles[d]
            if not p.label:
                h.gaps.append(f"No FDA label was retrieved for {d}, so its enzyme roles and warnings are unknown.")
        kinds = [store.get(e).kind for e in h.supporting if store.get(e)]
        if h.mechanism_type in ("PK-enzyme", "PK-transporter") and kinds and all(k in ("label_context",) for k in kinds[:2]):
            h.gaps.append("Enzyme roles rest on indirect label wording only.")
        n_lit = lit_counts.get(tuple(h.drugs))
        if n_lit is None:
            h.gaps.append("Literature was not searched.")
        elif n_lit == 0:
            h.gaps.append("Europe PMC returned no abstracts that discuss both drugs (or the search was unavailable).")
        elif not any(store.get(e).kind == "literature" for e in h.supporting if store.get(e)):
            h.gaps.append(f"None of {n_lit} retrieved abstracts links these drugs to {h.term or 'this mechanism'}.")
        if reaction:
            if h.mechanism_type in ("PK-enzyme", "PK-transporter"):
                ids = reaction_link(profiles[h.victim], reaction, store)
                h.explains_reaction = True if ids else (False if profiles[h.victim].label else None)
                if not ids and profiles[h.victim].label:
                    h.gaps.append(f"{h.victim}'s label does not list {reaction.lower()}; a change in exposure does not obviously explain it.")
            elif h.mechanism_type == "PD-effect-axis":
                ids = [] if h.term not in lp.reaction_axes(reaction) else \
                    [e for d in h.drugs for e in profiles[d].axes.get(h.term, [])[:1]]
                h.explains_reaction = h.term in lp.reaction_axes(reaction)
                if not h.explains_reaction:
                    h.gaps.append(f"The reported reaction ({reaction.lower()}) is not on the '{h.term}' axis.")
            else:
                ids = [e for d in h.drugs for e in reaction_link(profiles[d], reaction, store)]
                h.explains_reaction = True if ids else None
            h.reaction_evidence = list(dict.fromkeys(ids))
    # alternatives: other mechanisms that would explain the same reaction, and what FAERS says about single drugs
    for h in hyps:
        alt = [o.id for o in hyps if o is not h and o.explains_reaction and o.mechanism_type != h.mechanism_type
               and set(o.drugs) == set(h.drugs)]
        h.alternatives = [f"{o.id}: {o.title}" for o in hyps if o.id in alt]
        if faers and faers.get("verdict") == "Single-drug effect" and faers.get("driver_drugs"):
            h.alternatives.append(f"FAERS attributes the signal to {faers['driver_drugs']} alone (no interaction needed).")


def score(hyps: list[Hypothesis], store: EvidenceStore, table: dict | None = None, rho: float = 0.7,
          conformal=None, sigma: float = 0.5) -> None:
    table = {**em.DEFAULT_LOG_LR, **{k: v for k, v in (table or {}).items() if not k.startswith("_")}}

    def items(ids):
        out = []
        for e in dict.fromkeys(ids):
            ev = store.get(e)
            if ev:
                out.append(em.Item(KIND_MAP.get(ev.kind, ev.kind), ev.group or ev.source, float(ev.meta.get("weight", 1.0))))
        return out

    for h in hyps:
        h.supporting = list(dict.fromkeys(h.supporting))
        h.contradicting = list(dict.fromkeys(h.contradicting))
        it = items(h.supporting + h.contradicting)
        res = em.combine(it, h.prior, rho, table)
        h.log_odds, h.posterior = res["log_odds"], res["posterior"]
        h.interval = em.posterior_interval(it, h.prior, rho, table, sigma=sigma)
        sup_kinds = {KIND_MAP.get(store.get(e).kind, store.get(e).kind) for e in h.supporting if store.get(e)}
        sup_groups = {store.get(e).group for e in h.supporting if store.get(e)}
        if h.contradicting and h.posterior < UNSUPPORTED_P:
            h.verdict = "unsupported"
        elif h.posterior >= SUPPORTED_P and (len(sup_groups) >= 2 or sup_kinds & AUTHORITATIVE):
            h.verdict = "contested" if h.contradicting else "supported"
        elif h.contradicting:
            h.verdict = "contested"
        else:
            h.verdict = "untested"
        if conformal is not None and conformal.n:
            h.conformal = {"calibrated": True, "p_value": conformal.pvalue(h.posterior), "accepted": conformal.accepts(h.posterior),
                           "alpha": conformal.alpha, "n_calibration": conformal.n}
        else:
            h.conformal = {"calibrated": False}
    hyps.sort(key=lambda h: -h.posterior)             # ids stay as generated: alternatives refer to them
