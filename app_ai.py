"""Streamlit pages for the mechanism engine: Mechanism explorer, Structure & analogues, Discovery, Validation.

Imported by app.py (which owns st.set_page_config and the sidebar). Every page degrades gracefully: when a data source
is unreachable the page says which one and what is therefore missing, instead of showing an empty or fabricated result.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import ui

ROOT = Path(__file__).resolve().parent
RES = ROOT / "data" / "results"
INK, TEAL, RISK, MUTE, AMBER, GREEN = "#12303A", "#0E7C86", "#C2185B", "#6B8791", "#E0A030", "#2E8B57"
VERDICT_COLOR = {"supported": GREEN, "contested": AMBER, "untested": MUTE, "unsupported": RISK}
VERDICT_PLAIN = {
    "supported": "Evidence for is strong enough and nothing contradicts it.",
    "contested": "Some evidence for, but also evidence against.",
    "untested": "Plausible by mechanism, but little or no direct evidence was found.",
    "unsupported": "Evidence against outweighs evidence for.",
}
DEMO = ["warfarin", "fluconazole"]


# ------------------------------------------------------------------------------------------------ plumbing
@st.cache_resource(show_spinner=False)
def services():
    from pharmasense_ai.services import Services
    return Services.build()


def _pill(text, color):
    return f'<span style="display:inline-block;padding:.12rem .65rem;border-radius:999px;background:{color};color:#fff;font-weight:600;font-size:.82rem">{text}</span>'


def _bar(post, lo, hi, color):
    """Posterior marker with its 90% interval on a 0-100% track."""
    return (f'<div style="position:relative;height:12px;background:#E3ECEE;border-radius:6px;margin:.35rem 0 .1rem">'
            f'<div style="position:absolute;left:{lo * 100:.1f}%;width:{max((hi - lo) * 100, .8):.1f}%;top:0;bottom:0;background:{color}55;border-radius:6px"></div>'
            f'<div style="position:absolute;left:calc({post * 100:.1f}% - 3px);width:6px;top:-3px;bottom:-3px;background:{color};border-radius:3px"></div></div>'
            f'<div style="font-size:.78rem;color:{MUTE}">posterior {post:.0%} · 90% interval {lo:.0%}-{hi:.0%}</div>')


def _evidence_list(ids, ev):
    for e in ids:
        x = ev.get(e, {"id": e})
        ref = x.get("ref") or ""
        link = f" — [{x.get('source', '')}]({ref})" if ref.startswith("http") else f" — {x.get('source', '')}"
        st.markdown(f"**[{e}]** `{x.get('kind', '?')}` {x.get('text', '')[:420]}{link}")


def _svg(smiles, size=(300, 210)):
    try:
        from pharmasense_ai import chem
        s = chem.svg(smiles, size)
        return s or None
    except Exception:  # RDKit missing or SMILES invalid
        return None


def _show_svg(svg_text, caption=None):
    if svg_text:
        svg_text = svg_text.replace("<svg ", '<svg style="width:100%;height:auto;max-width:360px" ', 1)   # scale to the column
        st.markdown(f'<div style="background:#fff;border-radius:12px;border:1px solid #DDE8EA;padding:.3rem;text-align:center">{svg_text}</div>', unsafe_allow_html=True)
    else:
        st.caption("Structure image not available (no SMILES, or the drawing library is missing on this server).")
    if caption:
        st.caption(caption)


def _availability_banner(rep):
    if rep.get("sources_unavailable"):
        with st.expander(f"⚠ {len(rep['sources_unavailable'])} source problem(s): results below may be incomplete", expanded=not rep.get("sources_used")):
            for g in rep["sources_unavailable"]:
                st.markdown(f"- {g}")
            st.caption("A missing source is shown as a gap. It never counts as evidence against a mechanism. Run `python -m pharmasense_ai doctor` to see which sources work on this machine.")
    if not rep.get("sources_used"):
        st.error("No data source was reachable, so no mechanism can be assessed. Check your internet connection.")


# ------------------------------------------------------------------------------------------------ hypothesis rendering
def render_hypotheses(rep, key=""):
    ev = rep["evidence"]
    hyps = rep["hypotheses"]
    if not hyps:
        st.info(rep["narrative"])
        return
    for h in hyps:
        col = VERDICT_COLOR.get(h["verdict"], MUTE)
        with st.container(border=True):
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown(f"**{h['id']} · {h['title']}**  \n{_pill(h['verdict'].upper(), col)} &nbsp; <span style='color:{MUTE};font-size:.85rem'>{h['mechanism_type']}"
                            + (" · LLM-proposed" if h["origin"] == "llm-proposed" else "") + "</span>", unsafe_allow_html=True)
                st.write(h["claim"])
                if h["explains_reaction"] is True:
                    st.caption("✓ A drug label connects the affected drug to the reported reaction.")
                elif h["explains_reaction"] is False:
                    st.caption("✗ No label statement connects this mechanism to the reported reaction.")
            with c2:
                st.markdown(_bar(h["posterior"], h["interval_90"][0], h["interval_90"][1], col), unsafe_allow_html=True)
                st.caption(f"prior {h['prior']:.0%} → evidence → posterior. {VERDICT_PLAIN.get(h['verdict'], '')}")
                cf = h.get("conformal", {})
                if cf.get("calibrated"):
                    st.caption(f"Conformal check (n={cf['n_calibration']}): {'accepted' if cf['accepted'] else 'not accepted'} at α={cf['alpha']}")
            if h["supporting"]:
                with st.expander(f"Evidence for ({len(h['supporting'])})"):
                    _evidence_list(h["supporting"], ev)
            if h["contradicting"]:
                with st.expander(f"Evidence against ({len(h['contradicting'])})", expanded=True):
                    _evidence_list(h["contradicting"], ev)


def render_challenge(rep):
    hyps = rep["hypotheses"]
    st.markdown("##### Contradictions, missing evidence and alternatives")
    st.caption("A gap means *not found in the sources searched*. It is not evidence that a mechanism is false.")
    if not hyps:
        st.info("Nothing to challenge: no hypothesis was proposed.")
        return
    rows = []
    for h in hyps:
        for e in h["contradicting"]:
            rows.append({"Hypothesis": f"{h['id']} {h['title']}", "Kind": "Contradiction", "Detail": rep["evidence"].get(e, {}).get("text", e)[:300]})
        for g in h["gaps"]:
            rows.append({"Hypothesis": f"{h['id']} {h['title']}", "Kind": "Missing evidence", "Detail": g})
        for a in h["alternatives"]:
            rows.append({"Hypothesis": f"{h['id']} {h['title']}", "Kind": "Alternative", "Detail": a})
    if rows:
        d = pd.DataFrame(rows)
        for kind in ("Contradiction", "Missing evidence", "Alternative"):
            sub = d[d.Kind == kind]
            if len(sub):
                with st.expander(f"{kind} ({len(sub)})", expanded=kind == "Contradiction"):
                    st.dataframe(sub[["Hypothesis", "Detail"]], hide_index=True, width="stretch")
    else:
        st.success("No contradictions, gaps or alternatives were recorded.")
    f = rep.get("faers")
    if f and f.get("flags"):
        st.warning(f"FAERS data-quality flags for this signal: {f['flags']}")


def render_drugs(rep):
    names = list(rep["drugs"])
    key = "|".join(f"{n}:{rep['drugs'][n].get('smiles') or ''}" for n in names)
    if st.session_state.get("_drug_svgs_key") != key:            # draw once per report, in a separate process
        from pharmasense_ai import chem
        st.session_state["_drug_svgs"] = chem.svgs_isolated([(rep["drugs"][n].get("smiles") or "", 300, 210) for n in names])
        st.session_state["_drug_svgs_key"] = key
    pics = st.session_state["_drug_svgs"]
    cols = st.columns(len(names))
    for c, n, pic in zip(cols, names, pics):
        d = rep["drugs"][n]
        with c, st.container(border=True):
            st.markdown(f"**{n}**")
            _show_svg(pic or None)
            s = d.get("structure") or {}
            if s:
                st.caption(" · ".join(f"{k} {v}" for k, v in s.items() if v not in (None, "") and k in ("formula", "mw", "logp", "tpsa", "hbd", "hba", "molecular_weight", "molecular_formula"))[:200] or "")
            if d["enzyme_roles"]:
                st.markdown("**Enzyme / transporter roles**")
                st.dataframe(pd.DataFrame(d["enzyme_roles"])[["enzyme", "role", "confidence"]], hide_index=True, width="stretch")
            if d["targets"]:
                st.markdown("**Molecular targets**")
                for t in d["targets"][:5]:
                    st.markdown(f"- {t['name']} ({t['action'] or 'acts on'}){' — ' + ', '.join(t['genes']) if t['genes'] else ''}")
            if d["safety_axes"]:
                st.markdown("**Label warns about:** " + ", ".join(d["safety_axes"]))
            if d["indications"]:
                st.markdown("**Used for:** " + ", ".join(i if isinstance(i, str) else str(i.get("name", i)) for i in d["indications"][:5]))
            if d.get("faers_adverse_events_open_targets"):
                ae = d["faers_adverse_events_open_targets"]
                st.markdown("**Reported adverse events (Open Targets / FAERS):** " + ", ".join(a if isinstance(a, str) else str(a.get("name", a)) for a in ae[:6]))
            if d["human_studies"]["n_listed"]:
                st.caption(f"{d['human_studies']['n_listed']} registered human studies on ClinicalTrials.gov")
            if d.get("label_url"):
                st.markdown(f"[FDA label]({d['label_url']})")
            for g in d["gaps"]:
                st.caption(f"⚠ {g}")


# ------------------------------------------------------------------------------------------------ page 1
def page_mechanism():
    st.title("Mechanism explorer")
    st.caption("Why might these drugs interact? Targets → metabolic enzymes → shared proteins → literature → competing hypotheses → evidence for and against each.")
    with st.form("mech"):
        c1, c2 = st.columns([3, 2])
        drugs_txt = c1.text_input("Drugs (2 or 3, comma-separated, generic names)", value=", ".join(DEMO))
        reaction = c2.text_input("Reaction to explain (optional, MedDRA wording)", value="INTERNATIONAL NORMALISED RATIO INCREASED")
        src = st.radio("FAERS context", ["None", "Whole openFDA database (live counts)", "My local pipeline results (3 drugs only)"], horizontal=True,
                       help="Live counting asks openFDA for 16 counts (cached afterwards). Local results come from the sample you fetched.")
        go = st.form_submit_button("Explain and challenge", type="primary")
    if go:
        drugs = [d.strip() for d in drugs_txt.split(",") if d.strip()]
        if not 2 <= len(drugs) <= 3:
            st.error("Enter two or three drugs.")
            return
        try:
            from pharmasense_ai import faers_context as fc
            from pharmasense_ai.agents.workflow import explain
            svc = services()
            faers = None
            with st.status("Running the evidence workflow …", expanded=True) as status:
                if reaction.strip() and src.startswith("Whole"):
                    st.write("Counting the whole FAERS database …")
                    faers = fc.from_live(svc.openfda, drugs, reaction.strip())
                elif reaction.strip() and src.startswith("My local"):
                    faers = fc.from_results(drugs, reaction.strip())
                    if faers is None:
                        st.write("No local result for that exact triplet and reaction; continuing without FAERS context.")
                st.write("Profiling drugs, finding shared enzymes and proteins, searching literature, challenging each hypothesis …")
                rep = explain(drugs, reaction.strip() or None, faers, svc)
                status.update(label="Done", state="complete")
            st.session_state["mech_report"] = rep
        except Exception as e:  # noqa: BLE001 - show the cause to the user instead of a stack trace
            st.error(f"The workflow failed: {type(e).__name__}: {e}")
            return
    rep = st.session_state.get("mech_report")
    if not rep:
        st.info("Enter drugs and press **Explain and challenge**. With no network access the page will say which sources were unreachable.")
        st.subheader("How an explanation is built")
        steps = [("1 · Profile", "Targets, enzyme roles and label warnings for each drug (ChEMBL, UniProt, openFDA labels).", ui.TEAL),
                 ("2 · Overlaps", "Shared enzymes, shared proteins, pathways and safety axes between the drugs.", ui.TEAL),
                 ("3 · Evidence", "Label interaction text, Europe PMC abstracts and FAERS counts, ranked by BM25.", ui.AMBER),
                 ("4 · Score", "Bayesian log-odds per hypothesis, with correlated evidence discounted and a credible interval.", ui.AMBER),
                 ("5 · Challenge", "Contradictions, gaps and rival explanations are shown next to every claim.", ui.RISK)]
        for col, (t, b, c) in zip(st.columns(5), steps):
            col.markdown(ui.card(t, b, accent=c), unsafe_allow_html=True)
        return
    _availability_banner(rep)
    st.markdown(f"### {' + '.join(rep['query']['drugs'])}" + (f" → {rep['query']['reaction'].lower()}" if rep["query"].get("reaction") else ""))
    st.write(rep["narrative"])
    f = rep.get("faers")
    if f:
        st.markdown(f"**FAERS:** {f.get('verdict')} · ROR {f.get('ror', float('nan')):.1f} (95% CI {f.get('ror_lcl', float('nan')):.1f}–{f.get('ror_ucl', float('nan')):.1f}) · "
                    f"{f.get('n_events')} of {f.get('n_reports')} reports · {f.get('source')}")
    st.caption(f"{rep['calibration']}. LLM {'used for wording and extra hypotheses (validated against retrieved evidence)' if rep['llm_used'] else 'not used: every statement comes from rules and cited evidence'}.")
    t1, t2, t3, t4, t5 = st.tabs(["Explain", "Challenge", "Evidence", "Drugs & structure", "Workflow trace"])
    with t1:
        render_hypotheses(rep)
    with t2:
        render_challenge(rep)
    with t3:
        ev = pd.DataFrame(rep["evidence"].values())
        if len(ev):
            st.dataframe(ev[["id", "kind", "source", "section", "text", "ref"]], hide_index=True, width="stretch",
                         column_config={"ref": st.column_config.LinkColumn("link")})
        st.caption("Literature searched: " + (", ".join(f"{k}: {v} abstracts" for k, v in rep["literature_abstracts"].items()) or "none"))
    with t4:
        render_drugs(rep)
    with t5:
        for step in rep["trace"]:
            st.markdown(f"- {step}")
        st.caption("Sources that answered: " + (", ".join(rep["sources_used"]) or "none"))
    c1, c2 = st.columns(2)
    from pharmasense_ai.report import render_markdown
    c1.download_button("Download report (Markdown)", render_markdown(rep), file_name="mechanism_report.md")
    c2.download_button("Download report (JSON)", json.dumps(rep, indent=1, default=str), file_name="mechanism_report.json")
    with st.expander("Standing caveats"):
        for c in rep["caveats"]:
            st.markdown(f"- {c}")


# ------------------------------------------------------------------------------------------------ page 2
LIBRARY_DEFAULT = ["ibuprofen", "naproxen", "ketoprofen", "flurbiprofen", "diclofenac", "celecoxib", "aspirin", "paracetamol", "warfarin", "simvastatin", "atorvastatin",
                   "pravastatin", "rosuvastatin", "fluoxetine", "sertraline", "paroxetine", "citalopram", "amiodarone", "fluconazole"]


def page_structure():
    st.title("Structure & similar drugs")
    st.caption("Molecular structure, look-alike drugs under four fingerprints, what is observed for each, and what is only *inferred* from analogues.")
    with st.form("struct"):
        c1, c2 = st.columns([2, 3])
        drug = c1.text_input("Drug", value="ibuprofen")
        lib_txt = c2.text_input("Compare against (comma-separated; PubChem structures are fetched once, then cached)", value=", ".join(LIBRARY_DEFAULT))
        use_chembl = st.checkbox("Also search all of ChEMBL for similar molecules (slower, needs internet)", value=False)
        go = st.form_submit_button("Find look-alikes", type="primary")
    if not go and "struct_rep" not in st.session_state:
        return
    if go:
        try:
            from pharmasense_ai import similar
            svc = services()
            names = sorted({n.strip().lower() for n in lib_txt.split(",") if n.strip()} | {drug.strip().lower()})
            with st.spinner("Looking up structures …"):
                lib, pending = similar.build_library_timed(names, svc, budget_s=30.0)
            smi = lib.get(drug.strip().lower())
            if not smi:
                why = ("PubChem did not answer (slow or unreachable). Try again in a minute; finished lookups are cached."
                       if drug.strip().lower() in pending else "PubChem does not recognise this name; try the generic name.")
                st.error(f"No structure found for '{drug}'. {why}")
                return
            stats = pd.read_parquet(RES / "drug_reaction_stats.parquet") if (RES / "drug_reaction_stats.parquet").exists() else None
            rep = similar.analog_report(drug.strip(), smi, lib, stats)
            rep["_smiles"] = smi
            rep["_pending"] = pending
            rep["chembl_similar"] = []
            if use_chembl:
                with st.spinner("Searching ChEMBL …"):
                    try:
                        rep["chembl_similar"] = svc.chembl.similar(smi, 70, 10)
                    except Exception:  # noqa: BLE001
                        rep["chembl_similar"] = []
            with st.spinner("Drawing structures …"):
                from pharmasense_ai import chem
                pics = chem.svgs_isolated([(smi, 340, 240)] + [(a["smiles"], 220, 160) for a in rep["analogues"]])
            rep["_svg_main"], rep["_svg_analogues"] = pics[0], pics[1:]
            st.session_state["struct_rep"] = rep
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed: {type(e).__name__}: {e}")
            return
    rep = st.session_state["struct_rep"]
    c1, c2 = st.columns([1, 2])
    with c1:
        _show_svg(rep.get("_svg_main") or None, rep["drug"].title())
    with c2:
        d = rep["descriptors"]
        if d:
            st.dataframe(pd.DataFrame([d]).T.rename(columns={0: "value"}), width="stretch")
        st.caption(f"Murcko scaffold: `{rep['scaffold'] or 'n/a'}`")
    if rep["observed_reactions"]:
        st.markdown("**Observed in your FAERS sample (statistically clear signals):** " + ", ".join(r.lower() for r in rep["observed_reactions"][:12]))
    else:
        st.caption("No local FAERS signals for this drug (or no local results yet).")
    st.markdown("#### Structural look-alikes")
    if not rep["analogues"]:
        st.info("No library molecule passed the similarity cut-off.")
    if rep.get("_pending"):
        st.caption("Not looked up in time (PubChem slow): " + ", ".join(rep["_pending"]) + ". Press the button again to include them.")
    pics = rep.get("_svg_analogues") or [""] * len(rep["analogues"])
    for a, pic in zip(rep["analogues"], pics):
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 2, 2])
            with c1:
                _show_svg(pic or None)
            with c2:
                st.markdown(f"**{a['drug'].title()}** · mean Tanimoto **{a['similarity']:.2f}**")
                st.caption(" · ".join(f"{k} {v:.2f}" for k, v in a["by_fingerprint"].items()))
                if a["scaffold_note"]:
                    st.caption(a["scaffold_note"])
                if a["drug"].title() in [x.title() for x in rep["possible_activity_cliffs"]]:
                    st.warning("Possible activity cliff: structurally close but very different observed reactions.")
            with c3:
                st.markdown("Observed signals: " + (", ".join(r.lower() for r in a["observed_reactions"][:8]) or "none in local sample"))
                if a["reaction_overlap"] == a["reaction_overlap"]:
                    st.caption(f"Reaction overlap with {rep['drug']}: {a['reaction_overlap']:.0%} (Jaccard)")
    if rep["inferred_from_analogues"]:
        st.markdown("#### Inferred from analogues (NOT observed for this drug)")
        st.dataframe(pd.DataFrame(rep["inferred_from_analogues"]), hide_index=True, width="stretch")
    if rep["chembl_similar"]:
        with st.expander("ChEMBL similarity search (≥70%)"):
            st.dataframe(pd.DataFrame(rep["chembl_similar"]), hide_index=True, width="stretch")
    st.caption(rep["caveat"])


# ------------------------------------------------------------------------------------------------ page 3
def page_discovery():
    st.title("Discovery")
    st.caption("Combination signals in your FAERS results, checked against the drug labels and the literature to see which are *not already documented*.")
    f = RES / "triplet_results.parquet"
    if not f.exists():
        st.info("Run `python run_pipeline.py` first: Discovery screens its triplet results.")
        return
    c1, c2 = st.columns(2)
    n = c1.slider("Candidates to explain", 1, 20, 6)
    top = c2.slider("Consider the top N by ROR lower bound", 5, 60, 30)
    if st.button("Screen candidates", type="primary"):
        try:
            from pharmasense_ai import discover
            T = pd.read_parquet(f)
            reports = pd.read_parquet(RES / "reports_clean.parquet") if (RES / "reports_clean.parquet").exists() else None
            with st.spinner("Checking each candidate against labels and literature …"):
                cand = discover.candidates_from_results(T, reports, top=top)
                out = discover.run_discovery(services(), cand, n, log=lambda *_: None)
            st.session_state["disc"] = out
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed: {type(e).__name__}: {e}")
            return
    out = st.session_state.get("disc")
    if out is None:
        return
    if not len(out):
        st.info("No combination-like candidates in the current results.")
        return
    st.caption("'Candidate novel' means: not found in the component drugs' FDA labels or the Europe PMC abstracts retrieved. It does not mean unknown to science.")
    view = out.drop(columns=["report"]).assign(posterior=lambda d: d["posterior"].round(2), ror_lcl=lambda d: d["ror_lcl"].round(1))
    st.dataframe(view, hide_index=True, width="stretch")
    pick = st.selectbox("Open a candidate", range(len(out)), format_func=lambda i: f"{out.iloc[i]['drugs']} → {out.iloc[i]['reaction'].lower()}")
    rep = out.iloc[pick]["report"]
    _availability_banner(rep)
    t1, t2 = st.tabs(["Explain", "Challenge"])
    with t1:
        render_hypotheses(rep)
    with t2:
        render_challenge(rep)


# ------------------------------------------------------------------------------------------------ page 4
def page_validation():
    st.title("Validation")
    st.warning("The benchmark is hand-curated and small. It tests whether the engine **recovers documented mechanisms** and stays quiet where the mechanism is "
               "not established. It does **not** show that Discover finds true unknown interactions.")
    f = ROOT / "data" / "benchmark" / "last_run.json"
    if not f.exists():
        st.info("No benchmark run saved yet. Run `python -m pharmasense_ai validate --out data/benchmark/last_run.json` (needs internet; add `--calibrate` to re-fit the evidence weights).")
    else:
        d = json.loads(f.read_text())
        s = d["summary"]
        c = st.columns(4)
        c[0].metric("Recall (positives)", f"{s['recall']['value']:.0%}", f"95% CI {s['recall']['ci95'][0]:.0%}–{s['recall']['ci95'][1]:.0%}", delta_color="off")
        c[1].metric("False-positive rate", f"{s['false_positive_rate']['value']:.0%}", f"95% CI {s['false_positive_rate']['ci95'][0]:.0%}–{s['false_positive_rate']['ci95'][1]:.0%}", delta_color="off")
        if "auroc" in s:
            ci = s.get("auroc_ci95") or [float("nan")] * 2
            c[2].metric("AUROC", f"{s['auroc']:.2f}", f"95% CI {ci[0]:.2f}–{ci[1]:.2f}", delta_color="off")
        c[3].metric("Rows", f"{s['n_positive']} + / {s['n_negative']} −")
        st.dataframe(pd.DataFrame(d["rows"]), hide_index=True, width="stretch")
        if s["missed"]:
            st.markdown("**Missed positives:** " + "; ".join(f"{m['drug_a']}+{m['drug_b']} ({m['expected']})" for m in s["missed"]))
        if s["false_alarms"]:
            st.markdown("**False alarms:** " + "; ".join(f"{m['drug_a']}+{m['drug_b']} ({m['expected']})" for m in s["false_alarms"]))
    b = ROOT / "data" / "benchmark" / "known_interactions.csv"
    if b.exists():
        with st.expander("The benchmark list (audit it)"):
            st.dataframe(pd.read_csv(b), hide_index=True, width="stretch")
    diag = ROOT / "data" / "results" / "null_calibration.json"
    if diag.exists():
        with st.expander("FAERS statistics: permutation-null check"):
            st.json(json.loads(diag.read_text()))


# ------------------------------------------------------------------------------------------------ glossary
def glossary_tab():
    st.markdown("#### How the mechanism engine turns evidence into numbers")
    st.markdown("**Evidence combination.** Each hypothesis starts from a prior (by mechanism type) and adds log-likelihood-ratios. Several sentences from the same label are *not* independent, so within an evidence group of *n* items:")
    st.latex(r"n_{\mathrm{eff}}=\frac{n}{1+\rho\,(n-1)},\qquad \mathrm{logit}\,P(H\mid E)=\mathrm{logit}\,P(H)+\sum_{g}\bar{\ell}_g\,n_{\mathrm{eff},g}")
    st.caption("ρ = within-group correlation (0.7 by default). The weights ℓ are hand-set defaults unless you ran `validate --calibrate`. The 90% interval comes from Monte-Carlo over uncertainty in those weights.")
    st.markdown("**Independent causes** (noisy-OR): probability that at least one of several routes is real.")
    st.latex(r"P=1-\prod_i (1-p_i)")
    st.markdown("**Whole-database counts without a NOT operator** (Möbius inversion): openFDA returns 'reports containing at least these drugs', from which the exact groups are rebuilt.")
    st.latex(r"\mathrm{exact}(T)=\sum_{S\supseteq T}(-1)^{|S|-|T|}\,\mathrm{atleast}(S)")
    st.markdown("**Additive-scale interaction** (alongside the multiplicative test): excess over the sum of the separate effects.")
    st.latex(r"\mathrm{RERI}=OR_{AB}-OR_A-OR_B+1,\qquad I_3=OR_{ABC}-OR_{AB}-OR_{AC}-OR_{BC}+OR_A+OR_B+OR_C-1")
    st.markdown("**Link prediction on the knowledge graph** (personalised PageRank): how strongly a drug reaches a reaction through many short paths.")
    st.latex(r"r=\alpha\,s+(1-\alpha)\,P\,r")
    st.markdown("**Structural similarity** (Tanimoto on bit fingerprints, averaged over four kinds):")
    st.latex(r"T(a,b)=\frac{|a\wedge b|}{|a\vee b|}")
    st.markdown("**Conformal acceptance.** A hypothesis with posterior *s* gets a p-value from calibration scores of known-true mechanisms:")
    st.latex(r"p=\frac{1+\#\{s_i\le s\}}{n+1}")
    st.caption("Accepted if p > α. With ~20 calibration examples this is coarse: treat it as a sanity check, not a guarantee.")
    st.markdown("**Cluster check.** Many FAERS reports of one combination can come from a single study or litigation batch. Observed concentration of dates and identical drug lists is compared with random draws from the exposed pool (Monte Carlo); a flagged signal may be an artefact of clustering.")


PAGES = {"Mechanism explorer": page_mechanism, "Structure & analogues": page_structure, "Discovery": page_discovery, "Validation": page_validation}
