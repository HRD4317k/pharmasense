"""Build a DrugProfile: structure -> targets -> pathways, label-derived enzyme roles and safety axes.

Each external call is wrapped: a source that fails (offline, rate-limited, schema changed) becomes an entry in
`profile.gaps`, and the rest of the profile is still built. "We could not look" is different from "there is nothing" -
the report keeps that difference.
"""
from __future__ import annotations

from . import label_parse as lp
from .http import ApiError
from .models import DrugProfile, EnzymeRole, EvidenceStore, Target

MAX_TARGETS = 8
MAX_UNIPROT_PER_TARGET = 2


def _try(fn, gaps, label, default=None):
    try:
        return fn()
    except ApiError as e:
        gaps.append(f"{label}: {e}")
        return default


def build_profile(name: str, svc, store: EvidenceStore) -> DrugProfile:
    p = DrugProfile(name=name)
    gaps, ok = p.gaps, p.sources

    # ---- structure
    pc = _try(lambda: svc.pubchem.compound(name), gaps, "PubChem")
    if pc:
        p.structure, p.smiles = pc, pc.get("smiles") or pc.get("connectivity_smiles") or ""
        ok.append("PubChem")
    else:
        gaps.append(f"PubChem: no structure found for '{name}'")

    # ---- ChEMBL: id, mechanisms -> targets -> UniProt -> Reactome pathways
    ch = _try(lambda: svc.chembl.find(name), gaps, "ChEMBL")
    if ch and ch.get("chembl_id"):
        ok.append("ChEMBL")
        p.chembl_id, p.max_phase = ch["chembl_id"], ch.get("max_phase")
        if not p.smiles and ch.get("smiles"):
            p.smiles = ch["smiles"]
        mechs = _try(lambda: svc.chembl.mechanisms(p.chembl_id), gaps, "ChEMBL mechanisms", [])
        seen_t = set()
        for m in mechs:
            tid = m.get("target_chembl_id")
            if not tid or tid in seen_t or len(seen_t) >= MAX_TARGETS:
                continue
            seen_t.add(tid)
            t = _try(lambda tid=tid: svc.chembl.target(tid), gaps, f"ChEMBL target {tid}")
            tgt = Target(name=(t or {}).get("name") or tid, chembl_id=tid, action_type=m.get("action_type") or "",
                         mechanism=m.get("mechanism") or "")
            human = (t or {}).get("organism") in (None, "Homo sapiens")
            for acc in ((t or {}).get("uniprot") or [])[:MAX_UNIPROT_PER_TARGET] if human else []:
                u = _try(lambda acc=acc: svc.uniprot.protein(acc), gaps, f"UniProt {acc}")
                if u:
                    tgt.uniprot.append(acc)
                    if u.get("gene"):
                        tgt.genes.append(u["gene"])
                    tgt.pathways.extend(u.get("pathways") or [])
            eid = store.add("database", "ChEMBL mechanism of action",
                            f"https://www.ebi.ac.uk/chembl/compound_report_card/{p.chembl_id}/",
                            f"{name}: {tgt.mechanism or 'mechanism not stated'} ({tgt.action_type or 'action unknown'}) - target {tgt.name}"
                            + (f" [{', '.join(tgt.genes)}]" if tgt.genes else ""), [name], "mechanism", f"chembl:{name}")
            tgt.evidence.append(eid)
            p.targets.append(tgt)
            # CYP / transporter "targets" are really perpetrator roles (e.g. cobicistat, ritonavir)
            if lp.ENZ_RE.search(tgt.name) and tgt.action_type.upper() in ("INHIBITOR", "INDUCER", "ACTIVATOR"):
                role = "inducer" if tgt.action_type.upper() in ("INDUCER", "ACTIVATOR") else "inhibitor"
                p.roles.append(EnzymeRole(lp.norm_enzyme(lp.ENZ_RE.search(tgt.name).group(0)), role, "structured", eid, tgt.mechanism))
        p.indications = _try(lambda: svc.chembl.indications(p.chembl_id), gaps, "ChEMBL indications", [])[:10]
        # Open Targets: independent FAERS-derived enrichment (a cross-check, not a hypothesis source)
        p.ot_adverse_events = svc.opentargets.adverse_events(p.chembl_id)[:30]
        if p.ot_adverse_events:
            ok.append("Open Targets")
    else:
        gaps.append(f"ChEMBL: no molecule found for '{name}'")

    # ---- openFDA label: enzyme roles + safety axes + class
    label = _try(lambda: svc.openfda.label(name), gaps, "openFDA label")
    if label:
        ok.append("openFDA label")
        p.label = {"url": label["url"], "set_id": label["set_id"], "sections": label["sections"]}
        p.classes = label["classes"]
        p.names = lp.name_variants(label.get("generic_name"), label.get("brand_names"), label["classes"], name)
        src = f"openFDA label: {name}"
        for r in lp.extract_roles(p.names or [name.lower()], label["sections"], label["classes"]):
            kind = {"structured": "label_class", "explicit": "label_role", "context": "label_context"}[r["confidence"]]
            eid = store.add(kind, src, label["url"], r["sentence"], [name], r["section"], f"label:{name}",
                            enzyme=r["enzyme"], role=r["role"])
            p.roles.append(EnzymeRole(r["enzyme"], r["role"], r["confidence"], eid, r["sentence"]))
        for axis, hits in lp.effect_axes(label["sections"]).items():
            p.axes[axis] = [store.add("label_text", src, label["url"], s, [name], sec, f"label:{name}", axis=axis) for sec, s in hits]
    else:
        gaps.append(f"openFDA label: no label found for '{name}' (its enzyme roles and warnings are unknown)")
        p.names = [name.lower()]

    # ---- human studies (context for "overall effects in people")
    p.trials = svc.ctgov.studies(name, 10)
    if p.trials:
        ok.append("ClinicalTrials.gov")
    return p
