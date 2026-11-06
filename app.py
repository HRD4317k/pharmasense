"""PharmaSense - run from this folder:  streamlit run app.py
Reads data/results/ written by process_and_mine.py.

Pages: Overview | Drug search (drug -> reactions -> single drug or combination?) |
       Combination signals | Glossary & formulas | ML models
Everyday pages use plain language only; all mathematics lives on the Glossary page.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import exclusions
import pharmasense_core as core
import ui

try:                                   # optional mechanism-engine pages (need langgraph / rdkit / networkx)
    import app_ai
except Exception as _e:                # noqa: BLE001 - the original app must keep working without them
    app_ai, _AI_ERR = None, _e

ROOT = Path(__file__).resolve().parent
RES = ROOT / "data" / "results"
INK, TEAL, RISK, MUTE, AMBER = ui.INK, ui.TEAL, ui.RISK, ui.MUTE, ui.AMBER

VERDICTS = ["Combination effect", "Probable combination", "Pair effect", "Single-drug effect", "Cannot separate"]
VCOLOR = {"Combination effect": RISK, "Probable combination": "#E58AA8", "Pair effect": AMBER,
          "Single-drug effect": TEAL, "Cannot separate": MUTE, "No signal": "#C9D6DA"}
VPLAIN = {
    "Combination effect": "The risk appears only when all three drugs are taken together - higher than for any single drug or pair.",
    "Probable combination": "Looks like a combination effect, but some sub-groups are too small to be completely sure.",
    "Pair effect": "Two of the three drugs together explain the signal; the third looks like a bystander.",
    "Single-drug effect": "One drug on its own explains the signal; the other two look like bystanders.",
    "Cannot separate": "The drugs are almost always reported together, so this data cannot say which one is responsible.",
}

st.set_page_config(page_title="PharmaSense · drug-combination safety signals", page_icon="💊", layout="wide")
ui.inject()                            # theme: ui.py (palette, CSS, hero/card helpers, chart style) + .streamlit/config.toml


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load(stamp):
    D = pd.read_parquet(RES / "drug_reaction_stats.parquet")
    f = RES / "triplet_results.parquet"
    T = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    M = pd.read_parquet(RES / "drug_month.parquet")
    X = pd.read_parquet(RES / "excluded_terms.parquet")
    meta = json.loads((RES / "meta.json").read_text())
    return D, T, M, X, meta


@st.cache_resource(show_spinner="Preparing drug-partner analysis…")
def load_matrices(stamp):
    return core.build_matrices(pd.read_parquet(RES / "reports_clean.parquet"))


def fx(x):
    return f"{x:,.0f}" if x >= 100 else (f"{x:.1f}" if x >= 10 else f"{x:.2f}")


def combo(r):
    return " + ".join(core.disp(r[c]) for c in ("drug_a", "drug_b", "drug_c"))


def badge(v):
    return f'<span class="badge" style="background:{VCOLOR.get(v, MUTE)}">{v}</span>'


def style(fig, h=380):
    return ui.style_fig(fig, h)


def ladder(row):
    return ladder_from(core.evidence_table(row), {"REF": MUTE, "A": TEAL, "B": TEAL, "C": TEAL,
                                                   "AB": AMBER, "AC": AMBER, "BC": AMBER, "ABC": RISK})


def ladder_from(E, col):
    """Evidence ladder: how much higher is the reaction's odds in each exclusive group, with 95% CI."""
    fig = go.Figure()
    for e in E.itertuples():
        hover = f"{e.group}<br>{e.events} of {e.n:,} reports mention the reaction"
        if e.cat != "REF":
            hover += f"<br>{fx(e.ror)}x (95% CI {fx(e.lcl)}-{fx(e.ucl)})"
        fig.add_trace(go.Scatter(
            x=[e.ror], y=[e.group], mode="markers", marker=dict(size=11, color=col[e.cat]),
            error_x=dict(type="data", symmetric=False, array=[max(e.ucl - e.ror, 0)], arrayminus=[max(e.ror - e.lcl, 0)],
                         color=col[e.cat], thickness=2),
            hovertext=hover, hoverinfo="text", showlegend=False))
    fig.add_vline(x=1, line_dash="dot", line_color=MUTE)
    fig.update_xaxes(type="log", title="Odds of the reaction vs reports with none of these drugs "
                                       "(1 = no difference; log scale)")
    fig.update_yaxes(autorange="reversed")
    return style(fig, 300 if len(E) <= 4 else 330)


def show_detail(row):
    """Verdict badge + plain-language justification + evidence ladder for one result row."""
    v = row["verdict"]
    st.markdown(f"### {combo(row)} → {core.disp_rx(row['reaction'])}")
    st.markdown(badge(v), unsafe_allow_html=True)
    st.write(VPLAIN.get(v, ""))
    st.plotly_chart(ladder(row), width="stretch")
    st.markdown("**Why this verdict**")
    st.write(row["justification"])
    if row.get("flags"):
        st.warning("Read with care: " + row["flags"])
    st.caption("Attribution only considers these three drugs. Other co-reported drugs (a fourth medicine, say) "
               "may also contribute. Statistical signals from spontaneous reports - not proof of cause. "
               "See *Glossary & formulas* for how every number is calculated.")


# ---------------------------------------------------------------- pages
def page_overview(D, T, M, X, meta):
    st.markdown(ui.hero(
        "FDA adverse-event reports · openFDA",
        f"{meta['n_reports']:,} reports · {meta['n_drugs']:,} drugs · {meta['n_reactions']:,} reactions",
        f"Reports received {meta.get('date_min','')[:6]} to {meta.get('date_max','')[:6]} · each report lists at least three drugs",
        ("Disproportionality: ROR · PRR · IC", "Exact tests + Benjamini-Hochberg", "Mechanism engine: LangGraph + RAG", "Bayesian evidence scoring")),
        unsafe_allow_html=True)
    sig = T[T["verdict"] != "No signal"] if len(T) else T
    c = st.columns(4)
    c[0].metric("Reports analysed", f"{meta['n_reports']:,}")
    c[1].metric("3-drug combinations tested", f"{meta.get('n_tested', 0):,}",
                help="Every (3-drug combination, reaction) pair with at least 3 reports.")
    c[2].metric("Statistically clear signals", f"{len(sig):,}")
    c[3].metric("True combination effects", f"{int((sig['verdict'] == 'Combination effect').sum()) if len(sig) else 0:,}",
                help="Risk higher than any single drug or pair inside the combination.")
    if not len(T):
        st.info("No 3-drug combination reached the minimum number of reports. Fetch more data or relax the thresholds.")
        return
    left, right = st.columns(2)
    with left:
        st.subheader("Who is responsible?")
        vc = sig["verdict"].value_counts().reindex(VERDICTS).dropna()
        fig = go.Figure(go.Bar(x=vc.values, y=vc.index, orientation="h", marker_color=[VCOLOR[v] for v in vc.index]))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 330), width="stretch")
        st.caption("For every signal we check whether one drug, a pair, or the full combination explains it.")
    with right:
        st.subheader("How big are the ratios?")
        ror = np.clip(T["ror"].to_numpy(), 0.1, 1e4)
        edges = np.logspace(-1, 4, 51)
        counts, _ = np.histogram(ror, edges)
        fig = go.Figure(go.Bar(x=np.sqrt(edges[:-1] * edges[1:]), y=counts, marker_color=TEAL))
        fig.add_vline(x=1, line_dash="dot", line_color=RISK, annotation_text="1 = no association")
        fig.update_xaxes(type="log", title="Reporting odds ratio, every combination tested")
        fig.update_yaxes(title="Combinations")
        st.plotly_chart(style(fig, 330).update_layout(bargap=0.04), width="stretch")
        st.caption(f"Typical combination: {meta['median_ror_all_tested']:.1f}. "
                   f"Only {100 * meta['share_ror_over_10']:.1f}% exceed 10 and "
                   f"{100 * meta['share_ror_over_100']:.2f}% exceed 100 - those are rare, very specific combinations.")
    st.subheader("Strongest combination effects")
    top = sig[sig["verdict"].isin(["Combination effect", "Probable combination"])].sort_values("ror_lcl", ascending=False).head(10)
    if len(top):
        st.dataframe(pd.DataFrame({"Combination": top.apply(combo, axis=1), "Reaction": top["reaction"].map(core.disp_rx),
                                   "Verdict": top["verdict"], "Reports with all three": top["n_reports"],
                                   "Odds ratio": top["ror"], "Lowest plausible ratio": top["ror_lcl"]}),
                     hide_index=True, width="stretch",
                     column_config={"Combination": st.column_config.TextColumn(width="large"),
                                    "Reports with all three": st.column_config.NumberColumn("Reports (all 3)", width="small"),
                                    "Odds ratio": st.column_config.NumberColumn(format="%.1f", width="small"),
                                    "Lowest plausible ratio": st.column_config.NumberColumn("Lower 95%", format="%.1f", width="small")})
    else:
        st.info("No true combination effects were found in this data.")
    if meta.get("excluded_reaction_occurrences"):
        st.caption(f"{meta['excluded_reaction_occurrences']:,} reaction mentions were left out because they are not "
                   f"genuine adverse events (e.g. \"Feeling abnormal\", \"Drug ineffective\") or are the disease the drug "
                   f"treats. See *Glossary & formulas → What is excluded*.")


def overall_reading(drug, rx, srow, adj, tbl, T):
    """One plain-language conclusion combining the adjusted odds ratio, the partner analysis and 3-drug results."""
    D, R = core.disp(drug), core.disp_rx(rx).lower()
    names = lambda kind: " and ".join(core.disp(x) for x in tbl.loc[tbl["main_kind"] == kind, "partner"])
    pairs, partners = names("pair"), names("partner")
    adj_null = adj is not None and adj["lcl"] <= 1
    adj_sig = adj is not None and adj["lcl"] > 1
    alone = bool((tbl["verdict"] == "Drug alone").any())
    if adj_null and partners:
        out = (f"Overall reading: once other drugs are accounted for, {D} itself shows no clear association with {R}. "
               f"The reaction tracks {partners}, not {D}.")
    elif adj_null and srow["signal"]:
        out = (f"Overall reading: the crude association disappears once co-reported drugs are accounted for - it mostly "
               f"reflects what {D} is commonly taken with.")
    elif pairs:
        out = (f"Overall reading: {D} appears to act together with {pairs}. {R.capitalize()} is far more common when both are "
               f"present than with either drug alone.")
    elif partners:
        out = f"Overall reading: the reaction mainly tracks {partners} rather than {D}."
    elif adj_sig or alone:
        out = (f"Overall reading: {D} on its own appears associated with {R}; no partner drug adds to the effect.")
    elif srow["signal"]:
        out = f"Overall reading: {D} shows an association with {R}, but the data cannot say whether a partner drug is involved."
    else:
        out = f"Overall reading: no clear signal for {D} and {R}."
    if len(T):
        m = T[(T["reaction"] == rx) & (T["verdict"] == "Combination effect") &
              ((T["drug_a"] == drug) | (T["drug_b"] == drug) | (T["drug_c"] == drug))]
        if len(m):
            top = m.sort_values("ror_lcl", ascending=False).iloc[0]
            out += (f" At the 3-drug level, {len(m)} combination(s) containing {D} show the effect only when all three "
                    f"are present (strongest: {combo(top)}).")
    return out


def page_search(D, T, M, X, meta):
    st.title("Drug search")
    st.caption("Type a drug name to see its reported reactions, and whether each reaction looks like the drug on its "
               "own or a combination of drugs.")
    counts = D.groupby("drug")["n_drug"].first().sort_values(ascending=False)
    choice = st.selectbox("Drug", list(counts.index), index=None, placeholder="Start typing a drug name…",
                          format_func=lambda s: f"{core.disp(s)}  ({counts[s]:,} reports)")
    if not choice:
        st.subheader("Most reported drugs")
        st.dataframe(pd.DataFrame({"Drug": [core.disp(x) for x in counts.index[:15]], "Reports": counts.values[:15]}),
                     hide_index=True, width="stretch")
        return
    S = D[D["drug"] == choice].copy()
    n_drug = int(S["n_drug"].iloc[0])
    c = st.columns(3)
    c[0].metric("Reports mentioning this drug", f"{n_drug:,}", help=f"{100 * n_drug / meta['n_reports']:.1f}% of all reports analysed")
    c[1].metric("Reactions reported (≥3 reports)", f"{len(S):,}")
    c[2].metric("Statistically clear signals", f"{int(S['signal'].sum()):,}",
                help="Reaction reported more often with this drug than with other drugs (lower 95% bound of the odds ratio above 1).")

    st.subheader(f"Reactions reported with {core.disp(choice)}")
    f1, f2, f3 = st.columns([2, 1, 1])
    min_r = f1.slider("Minimum reports", 3, max(int(S["a"].max()), 4), 3)
    only_sig = f2.toggle("Only clear signals", False)
    more = f3.toggle("Show extra measures", False, help="PRR and IC are alternative disproportionality measures - see the Glossary.")
    V = S[S["a"] >= min_r]
    if only_sig:
        V = V[V["signal"]]
    V = V.sort_values("a", ascending=False)
    cols = {"Reaction": V["reaction"].map(core.disp_rx), "Reports": V["a"], "% of this drug's reports": V["pct_of_drug_reports"],
            "Odds ratio": V["ror"], "Lower 95%": V["ror_lcl"], "Upper 95%": V["ror_ucl"],
            "Clear signal": np.where(V["signal"], "Yes", "-")}
    if more:
        cols.update({"PRR": V["prr"], "IC025": V["ic025"]})
    st.dataframe(pd.DataFrame(cols), hide_index=True, width="stretch", height=320,
                 column_config={k: st.column_config.NumberColumn(format="%.2f") for k in
                                ["% of this drug's reports", "Odds ratio", "Lower 95%", "Upper 95%", "PRR", "IC025"]})
    st.caption("Odds ratio 1 = reported no more often than with other drugs; above 1 = more often. "
               "'Lower 95%' is the most cautious plausible value - a signal needs it above 1.")
    top = V.head(15).iloc[::-1]
    if len(top):
        fig = px.bar(top, x="a", y=top["reaction"].map(core.disp_rx), orientation="h",
                     color=np.where(top["signal"], "Clear signal", "Not a clear signal"),
                     color_discrete_map={"Clear signal": RISK, "Not a clear signal": MUTE})
        fig.update_layout(xaxis_title="Reports", yaxis_title=None, legend_title=None)
        st.plotly_chart(style(fig, 420), width="stretch")

    st.subheader("Single drug or a combination?")
    options = list(S.sort_values("a", ascending=False)["reaction"])
    rx = st.selectbox("Reaction to investigate", options, index=0, format_func=core.disp_rx)
    srow = S[S["reaction"] == rx].iloc[0]
    D_, R_ = core.disp(choice), core.disp_rx(rx).lower()
    st.write(f"On its own, **{D_}** is reported with {R_} in {int(srow['a'])} of {n_drug:,} reports "
             f"({srow['pct_of_drug_reports']:.1f}%); odds ratio {fx(srow['ror'])} "
             f"(95% CI {fx(srow['ror_lcl'])}-{fx(srow['ror_ucl'])}) - "
             + ("a clear signal." if srow["signal"] else "not a clear signal.") + " That crude figure ignores what else "
             "the patients were taking, so we test two more things:")
    stamp = (RES / "meta.json").stat().st_mtime
    Xm, Ym, drugs_m, rxs_m = load_matrices(stamp)
    tbl, main = core.explain_drug_reaction(Xm, Ym, drugs_m, rxs_m, choice, rx)
    adj = core.adjusted_or(Xm, Ym, drugs_m, rxs_m, choice, rx)

    st.markdown("**1. Holding other common drugs fixed.**")
    if adj:
        st.write(f"After accounting for the {adj['n_adjust']} most commonly co-reported drugs, the odds ratio for {D_} is "
                 f"**{fx(adj['or'])}** (95% CI {fx(adj['lcl'])}-{fx(adj['ucl'])}) - "
                 + ("still a clear signal." if adj["lcl"] > 1 else "no longer a clear signal."))
    else:
        st.write("Could not be estimated reliably (too few reports with this reaction, or it almost never occurs without this drug).")

    st.markdown("**2. Is it the drug alone, or the drug together with a partner?**")
    st.caption("For each frequent partner we compare reports with neither drug, with only one of them, and with both. "
               "The strongest partner is set aside and the rest re-tested, so a partner that merely rides along with the "
               "real one does not look responsible.")
    if tbl.empty:
        st.info("Not enough reports to analyse partner drugs.")
    else:
        def plabel(r):
            v, P = r["verdict"], core.disp(r["partner"])
            return {"Pair effect": f"{D_} + {P} together", "Drug alone": f"{D_} on its own",
                    "Partner alone": f"{P} on its own (not {D_})", "Both alone": "Each drug on its own",
                    "Cannot separate": "Cannot separate"}.get(v, "No clear signal")
        tbl["label"] = tbl.apply(plabel, axis=1)
        shown = tbl[~tbl["verdict"].isin(["No signal"])]
        st.info(overall_reading(choice, rx, srow, adj, tbl, T))
        if len(shown):
            st.dataframe(pd.DataFrame({"Partner": shown["partner"].map(core.disp), "Evidence points to": shown["label"],
                                       "Reports with both": shown["n_both"], "Odds ratio (both)": shown["ror_AB"],
                                       f"{D_} without partner": shown["ror_A"], "Partner without drug": shown["ror_B"]}),
                         hide_index=True, width="stretch", height=min(60 + 36 * len(shown), 330),
                         column_config={k: st.column_config.NumberColumn(format="%.1f") for k in
                                        ["Odds ratio (both)", f"{D_} without partner", "Partner without drug"]})
            pick = st.selectbox("See the evidence for a partner", list(shown.index), index=0,
                                format_func=lambda i: f"{core.disp(shown.loc[i, 'partner'])} - {shown.loc[i, 'label']}")
            row = shown.loc[pick]
            st.plotly_chart(ladder_from(row["_E"], {"REF": MUTE, "A": TEAL, "B": TEAL, "AB": RISK}), width="stretch")
            st.write(row["justification"])
            if row.get("note"):
                st.caption("Note: " + row["note"] + ".")
        with st.expander(f"All {len(tbl)} partners analysed"):
            st.dataframe(pd.DataFrame({"Partner": tbl["partner"].map(core.disp), "Evidence points to": tbl["label"],
                                       "Reports with both": tbl["n_both"], "Odds ratio (both)": tbl["ror_AB"],
                                       "Note": tbl["note"]}), hide_index=True, width="stretch",
                         column_config={"Odds ratio (both)": st.column_config.NumberColumn(format="%.1f")})

    if len(T):
        mine = T[(T["reaction"] == rx) & ((T["drug_a"] == choice) | (T["drug_b"] == choice) | (T["drug_c"] == choice))]
        mine = mine[mine["verdict"] != "No signal"].sort_values("ror_lcl", ascending=False)
        with st.expander(f"3-drug combinations that include {D_} ({len(mine)} with a clear signal for {R_})"):
            if mine.empty:
                st.write("None of the tested 3-drug combinations shows a clear signal for this reaction.")
            else:
                st.caption("Each combination is judged on its own three drugs only; a drug outside the three can still "
                           "influence the verdict. The partner analysis above is the better guide to single vs pair effects.")
                partners = mine.apply(lambda r: " + ".join(core.disp(r[c]) for c in ("drug_a", "drug_b", "drug_c") if r[c] != choice), axis=1)
                st.dataframe(pd.DataFrame({"With": partners, "Verdict": mine["verdict"], "Reports with all three": mine["n_reports"],
                                           "Odds ratio": mine["ror"]}), hide_index=True, width="stretch", height=240,
                             column_config={"Odds ratio": st.column_config.NumberColumn(format="%.1f")})
                idx = st.selectbox("See the evidence for a combination", list(mine.head(300).index), index=0,
                                   format_func=lambda i: f"{partners[i]} - {mine.loc[i, 'verdict']}")
                show_detail(mine.loc[idx])

    st.subheader("Reporting over time")
    tr = M[M["drug"] == choice].sort_values("month")
    if len(tr) > 1:
        fig = px.bar(tr, x="month", y="reports", color_discrete_sequence=[TEAL])
        fig.update_layout(xaxis_title=None, yaxis_title="Reports per month")
        st.plotly_chart(style(fig, 280), width="stretch")
    else:
        st.caption("Not enough months of data for a trend.")


def page_combos(D, T, M, X, meta):
    st.title("Combination signals")
    st.caption("Every 3-drug combination with a statistically clear signal, and who the evidence says is responsible.")
    if not len(T):
        st.info("No combination analysis available.")
        return
    sig = T[T["verdict"] != "No signal"]
    a, b, c = st.columns([2, 2, 1])
    vs = a.multiselect("Verdict", VERDICTS, default=["Combination effect", "Probable combination", "Pair effect"],
                       help="Single-drug effects are hidden by default: they are one drug's signal repeated in many triplets.")
    rxs = b.multiselect("Reaction", sorted(sig["reaction"].unique()), format_func=core.disp_rx)
    min_n = c.slider("Min. reports", 3, max(int(sig["n_events"].max()), 4), 3)
    q = st.text_input("Contains drug", placeholder="e.g. warfarin").strip().upper()
    V = sig[sig["verdict"].isin(vs) & (sig["n_events"] >= min_n)]
    if rxs:
        V = V[V["reaction"].isin(rxs)]
    if q:
        V = V[V[["drug_a", "drug_b", "drug_c"]].apply(lambda col: col.str.contains(q, regex=False)).any(axis=1)]
    V = V.sort_values("ror_lcl", ascending=False)
    st.write(f"{len(V):,} combinations")
    st.dataframe(pd.DataFrame({"Combination": V.apply(combo, axis=1), "Reaction": V["reaction"].map(core.disp_rx),
                               "Verdict": V["verdict"], "Reports with all three": V["n_reports"],
                               "With reaction": V["n_events"], "Odds ratio": V["ror"], "Lower 95%": V["ror_lcl"]}),
                 hide_index=True, width="stretch", height=320,
                 column_config={"Odds ratio": st.column_config.NumberColumn(format="%.1f"),
                                "Lower 95%": st.column_config.NumberColumn(format="%.1f")})
    if len(V):
        i = st.selectbox("See the evidence", list(V.head(500).index),
                         format_func=lambda i: f"{combo(V.loc[i])} → {core.disp_rx(V.loc[i, 'reaction'])}")
        show_detail(V.loc[i])


def page_glossary(D, T, M, X, meta):
    st.title("Glossary & formulas")
    st.caption("You do not need any of this to use the app. It is here for anyone who wants to check how a number is made.")
    t1, t2, t3, t4, t5, t6 = st.tabs(["Key definitions", "Formulas", "How the verdict is decided", "What is excluded", "Limitations", "Mechanism engine"])
    with t6:
        if app_ai is not None:
            app_ai.glossary_tab()
        else:
            st.info("The mechanism engine is not installed. `pip install -r requirements.txt`")
    with t1:
        defs = {
            "FAERS / openFDA": "The FDA Adverse Event Reporting System: voluntary reports from patients, doctors and manufacturers. openFDA serves it through a public API.",
            "Report": "One submitted case. Each report lists the drugs taken and the reactions that occurred.",
            "Adverse event / reaction": "A medical problem reported while taking a drug. It is *reported*, not proven to be caused by the drug.",
            "Polypharmacy": "Taking several medicines at once. Here: reports listing at least three different drugs.",
            "Signal": "A pattern that is reported more often than chance would explain and deserves a closer look. Not proof of harm.",
            "Odds ratio (ROR)": "How many times higher the odds of a reaction are in reports with a drug than in reports without it. **1 = no association**; 2 = twice the odds. It is not capped: a rare reaction that almost never appears otherwise can give very large values.",
            "95% confidence interval": "The range of values the true ratio plausibly lies in. 'Lower 95%' is the most cautious end. A signal needs it above 1.",
            "PRR": "Proportional reporting ratio - a similar measure based on proportions instead of odds. Common alongside ROR.",
            "IC / IC025": "Information Component - a measure that shrinks unreliable small-count results toward 0 (0 = no association). IC025 above 0 counts as a signal.",
            "Reference group": "For a 3-drug combination: reports that list none of the three drugs.",
            "Sub-group": "Reports with only some of the three drugs - e.g. 'A only' or 'A + B only' (without the third).",
            "Single / pair / combination effect": "Whether the signal is explained by one drug, two drugs, or only appears when all three are taken together.",
            "Interaction test": "A statistical check whether the combination adds risk beyond what the separate drugs would give (on the odds scale).",
            "q-value (FDR)": "A p-value adjusted for the thousands of combinations tested at once. q below 0.05 means roughly under 5% of such findings are expected to be false alarms.",
            "Confounding by indication": "A reaction that is really the disease the drug treats (e.g. 'Plasma cell myeloma' for a myeloma drug), not a side effect. Such terms are removed.",
            "Support": "How many reports contain a combination. Rare combinations are not analysed.",
        }
        for k, v in defs.items():
            with st.expander(k):
                st.markdown(v)
    with t2:
        st.markdown("**Counts used everywhere.** For a drug (or combination) and a reaction, count reports in a 2×2 table:")
        st.markdown("| | has the reaction | no reaction |\n|---|---|---|\n| **has the drug** | a | b |\n| **does not** | c | d |")
        st.markdown("**Reporting odds ratio (ROR)**")
        st.latex(r"\mathrm{ROR}=\frac{a/b}{c/d}=\frac{a\,d}{b\,c}")
        st.latex(r"\mathrm{SE}(\ln \mathrm{ROR})=\sqrt{\tfrac1a+\tfrac1b+\tfrac1c+\tfrac1d},\qquad "
                 r"95\%\ \mathrm{CI}=\exp\!\big(\ln \mathrm{ROR}\pm 1.96\,\mathrm{SE}\big)")
        st.caption("If any of a, b, c, d is zero, 0.5 is added to all four (Haldane-Anscombe). It is not added otherwise. "
                   "Worked example: a=20, b=80, c=100, d=9800 → ROR = 24.5 (95% CI 14.4-41.5).")
        st.markdown("**Proportional reporting ratio (PRR) and chi-square**")
        st.latex(r"\mathrm{PRR}=\frac{a/(a+b)}{c/(c+d)},\quad \mathrm{SE}(\ln\mathrm{PRR})=\sqrt{\tfrac1a-\tfrac1{a+b}+\tfrac1c-\tfrac1{c+d}}")
        st.latex(r"\chi^2=\frac{N\,\big(\max(|ad-bc|-N/2,\,0)\big)^2}{(a+b)(c+d)(a+c)(b+d)},\qquad N=a+b+c+d")
        st.caption("Evans criteria: PRR ≥ 2, χ² ≥ 4 and at least 3 reports.")
        st.markdown("**Information Component (IC)**")
        st.latex(r"\mathrm{IC}=\log_2\frac{a+0.5}{E+0.5},\quad E=\frac{n_{\text{drug}}\,n_{\text{reaction}}}{N},\quad "
                 r"\mathrm{IC}_{025}\approx \mathrm{IC}-\frac{3.3}{\sqrt{a+0.5}}-\frac{2}{(a+0.5)^{3/2}}")
        st.markdown("**Mutually exclusive groups for a 3-drug combination A, B, C.** Every report falls into exactly one of "
                    "eight groups: none of the three (reference), A only, B only, C only, A+B only, A+C only, B+C only, all three. "
                    "Each group's ROR is taken against the reference group.")
        st.markdown("**Head-to-head comparison** (is all-three higher than a sub-group *k*?)")
        st.latex(r"\mathrm{H}_k=\frac{a_{ABC}/b_{ABC}}{a_k/b_k},\qquad 95\%\ \mathrm{CI\ as\ for\ ROR}")
        st.markdown("**Three-way interaction** (saturated logistic model, computed exactly from the 8 group counts), "
                    "with $O_k$ = events ÷ non-events in group *k*:")
        st.latex(r"\beta_3=\ln\frac{O_{ABC}\;O_A\,O_B\,O_C}{O_{AB}\,O_{AC}\,O_{BC}\;O_{\text{ref}}},\qquad "
                 r"\mathrm{SE}(\beta_3)=\sqrt{\sum_{k}\Big(\tfrac1{a_k}+\tfrac1{b_k}\Big)},\qquad z=\beta_3/\mathrm{SE}")
        st.markdown("**Is the combination above the reference at all?** One-sided exact (Fisher / hypergeometric) test, "
                    "conditioned on the minimum of 3 reports. Exact tests stay valid at small counts, where the normal "
                    "approximation behind the confidence interval is too optimistic.")
        st.markdown("**Drug + partner groups.** For a drug A and a partner B every report falls in one of four groups: "
                    "neither, A without B, B without A, both. ROR of each group is taken against 'neither', and "
                    "'both' is compared head-to-head with 'A without B' and 'B without A' using the same formula.")
        st.markdown("**Adjusted odds ratio (holding other drugs fixed).** Logistic regression of the reaction on drug A "
                    "plus the 40 most commonly co-reported other drugs ($x_j$ = 1 if the report lists drug j):")
        st.latex(r"\ln\frac{P(\text{reaction})}{1-P(\text{reaction})}=\beta_0+\beta_A x_A+\sum_{j=1}^{40}\beta_j x_j,"
                 r"\qquad \mathrm{OR}_{\text{adj}}=e^{\beta_A},\quad 95\%\ \mathrm{CI}=e^{\beta_A\pm1.96\,\mathrm{SE}(\beta_A)}")
        st.markdown("**'All-three beats every sub-group' (intersection-union test).** The six one-sided exact head-to-head "
                    "p-values are computed; the combination only passes if the **largest** of them is small "
                    "(all six must pass). That p-value is then Benjamini-Hochberg-adjusted across all signals.")
        st.markdown("**Multiple testing - Benjamini-Hochberg.** With p-values sorted $p_{(1)}\\le\\dots\\le p_{(m)}$:")
        st.latex(r"q_{(i)}=\min_{j\ge i}\ \frac{m}{j}\,p_{(j)}")
    with t3:
        st.markdown("A combination is only judged if it is a **clear signal**: at least 3 reports with all three drugs and the "
                    "reaction, lower 95% bound of its ROR above 1, **and** q < 0.05 (exact test, corrected for every "
                    "combination tested). Then the all-three group is compared head-to-head with each of the six sub-groups:")
        for v in VERDICTS:
            st.markdown(f"- {badge(v)} {VPLAIN[v]}", unsafe_allow_html=True)
        st.markdown("""
**Rules, in order**
1. *Combination effect* - the all-three group is significantly higher than **all six** sub-groups, and this survives correction for the thousands of combinations compared (q < 0.05).
2. *Single-drug effect* - a single drug on its own has a clear signal and the all-three group is **not** significantly higher than it.
3. *Pair effect* - same, for a pair (when no single drug explains it).
4. *Probable combination* - the all-three group beats the sub-groups in each comparison but the result is not conclusive after correction, or some sub-groups are too small (<10 reports) to be ruled out.
5. *Cannot separate* - no sub-group has a signal of its own and the sub-groups are too small to compare.

The three-way interaction test is shown as supporting evidence. It is not required, because it has low power when a sub-group has few cases.""")
        st.markdown("#### The Drug search page: drug alone, or drug + partner?")
        st.markdown("""
For the drug and reaction you pick, each frequent partner drug is analysed (groups: neither, drug without partner, partner without drug, both):
- **Pair effect** - both together are significantly higher than the drug alone *and* than the partner alone.
- **Drug alone** - the drug already shows the signal without the partner, and adding the partner is not significantly higher.
- **Partner alone** - the partner shows the signal without the drug; the reaction belongs to the partner.
- **Each alone / Cannot separate / No clear signal** - as the names say.

All tests are exact and corrected for the number of partners tested. The strongest partner is then **set aside** (reports containing it are removed) and the rest re-tested, so a drug that merely travels with the real cause is not blamed. The *overall reading* combines this with the adjusted odds ratio: if the drug's adjusted odds ratio is no longer a clear signal, it is not the cause.""")
    with t4:
        st.markdown("A reaction is dropped **before any statistics** if it is not a specific medical event. These terms describe "
                    "circumstances, disease course or paperwork, are very common and highly clustered, and create "
                    "misleadingly large odds ratios. The lists below can be edited in `exclusions.py`.")
        for cat, terms in exclusions.CATEGORIES.items():
            with st.expander(f"{cat} ({len(terms)} terms)"):
                st.write(", ".join(sorted(t.capitalize() for t in terms)))
        st.markdown("Also removed automatically: **any reaction that equals the indication of a drug in the same report** "
                    "(the disease being treated is not a side effect).")
        if len(X):
            st.markdown("**Most frequently excluded in this data**")
            st.dataframe(X.head(15).rename(columns={"reaction": "Term", "category": "Why", "occurrences": "Mentions"})
                         .assign(Term=lambda d: d["Term"].map(core.disp_rx)), hide_index=True, width="stretch")
        st.markdown("Drug names are also merged to ingredient level (brand → generic when openFDA knows it; strengths, "
                    "tablet/capsule words and salts removed), and a 'triplet' made of the same ingredient listed under "
                    "several names is not counted as three drugs.")
    with t5:
        st.markdown("""
- Spontaneous reports show *what was reported*, not how often something happens, and reporting is uneven (news, litigation, new drugs).
- A statistical signal is **not** proof that a drug caused a reaction. Never use these results for medical decisions.
- Attribution considers **only the three drugs in each combination**. A fourth drug can make a harmless partner look responsible, or a 3-drug combination look like a 4-drug one.
- Drugs listed as concomitant (not suspect) are included by default; re-run with `--drug-scope suspect` to use only suspect/interacting drugs.
- openFDA returns the first records of each month, not a random sample of FAERS, and the same case can be reported more than once (only different versions of the same manufacturer case number are merged).
- Term lists for non-adverse-events are a curated heuristic, because mapping to MedDRA classes needs a licensed dictionary.""")


def page_ml():
    st.title("ML models")
    mp, fp = ROOT / "models" / "metrics.csv", ROOT / "models" / "feature_importance.csv"
    jp = ROOT / "data" / "processed" / "model_metrics.json"
    if not mp.exists():
        st.info("No model results yet. Run `python run_pipeline.py --skip-fetch --full`, which trains the models and writes "
                "models/metrics.csv.")
        return
    met = pd.read_csv(mp)
    info = json.loads(jp.read_text()) if jp.exists() else {}
    target = str(info.get("Target", "the target reaction")).lower()
    st.caption("A sanity probe, not a diagnostic tool: do the drugs on a report carry any information about its reaction?")

    st.markdown(ui.callout(
        f"<b>The question.</b> Take the 50 most-reported drugs. For each report, a model sees only <i>which of them are listed</i> "
        f"and must guess whether the report mentions <b>{target}</b>. If drug combinations carry real information, the models beat "
        f"chance; if not, they hover at an AUC of 0.5."), unsafe_allow_html=True)

    if info.get("n_reports"):
        c = st.columns(4)
        c[0].metric("Reports used", f"{info['n_reports']:,}")
        c[1].metric(f"Mention {target}", f"{info['n_positive']:,}", help="Reports where the target reaction appears.")
        c[2].metric("Base rate", f"{100 * info['prevalence']:.1f}%", help="A model that guesses at random would 'find' this share. Compare AUPRC against it.")
        c[3].metric("Distinct receive dates", f"{info.get('n_receive_dates', '?')}", help="Used to keep each reporting batch together when testing.")

    st.subheader("What each model is doing")
    k = st.columns(3)
    k[0].markdown(ui.card("Lasso", "Gives every drug a plus or minus score and adds them up. A penalty switches off drugs that add nothing, "
                          "so a short, readable list is left, with a direction: does this drug raise or lower the odds?",
                          "Linear · readable · picks drugs", TEAL), unsafe_allow_html=True)
    k[1].markdown(ui.card("Random forest", "100 decision trees each vote on the report. Trees can learn 'drug A matters only when drug B "
                          "is also there', which a straight sum cannot.",
                          "Trees · finds combinations", AMBER), unsafe_allow_html=True)
    k[2].markdown(ui.card("XGBoost", "Trees built one after another, each correcting the mistakes of the ones before. Usually the strongest on "
                          "large data; it also supplies the 'which drugs mattered' chart below.",
                          "Boosting · strongest on big data", RISK), unsafe_allow_html=True)

    st.subheader("How well do they do?")
    has_rand = "auc_random" in met.columns
    fig = go.Figure()
    if has_rand:
        fig.add_bar(name="Random split (optimistic)", x=met["model"], y=met["auc_random"], marker_color="#9FC9CE")
    fig.add_bar(name="Whole report dates held out" if has_rand else "AUC", x=met["model"], y=met["auc"], marker_color=TEAL)
    fig.add_hline(y=0.5, line_dash="dot", line_color=RISK)
    fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=RISK, dash="dot"), name="0.5 = guessing")
    fig.update_layout(barmode="group", yaxis_title="ROC AUC (higher is better)", yaxis_range=[0, 1.0])
    left, right = st.columns([3, 2])
    left.plotly_chart(style(fig, 360), width="stretch")
    prev = info.get("prevalence")
    fig2 = go.Figure(go.Bar(x=met["model"], y=met["auprc"], marker_color=[TEAL, AMBER, RISK][:len(met)]))
    if prev:
        fig2.add_hline(y=prev, line_dash="dot", line_color=RISK)
    fig2.update_layout(yaxis_title=f"AUPRC (dotted line = base rate {prev:.1%})" if prev else "AUPRC", yaxis_rangemode="tozero",
                       title="Finding the reports that mention it")
    right.plotly_chart(style(fig2, 360), width="stretch")

    best = float(met["auc"].max())
    gap = float((met["auc_random"] - met["auc"]).mean()) if has_rand else 0.0
    if best < 0.62:
        msg, kind = (f"<b>Close to chance.</b> The best model reaches an AUC of {best:.2f}. Knowing which common drugs are on a report "
                     f"tells us very little about whether it mentions {target}. That is the expected, honest result for a common, "
                     "non-specific symptom, and it is why PharmaSense leans on disproportionality statistics and mechanism "
                     "evidence instead of a black-box predictor.", "warn")
    elif best < 0.8:
        msg, kind = (f"<b>A modest signal.</b> Best AUC {best:.2f}: the drug list helps a little, but far from decisively.", "")
    else:
        msg, kind = (f"<b>A strong score ({best:.2f}) - check it before believing it.</b> Strong scores in FAERS often come from one "
                     "reporting batch, not from pharmacology.", "risk")
    if gap > 0.03:
        msg += (f" The random split scores about {gap:.2f} higher than the batch-aware test: part of an ordinary split's score is "
                "near-duplicate reports from the same batch appearing on both sides.")
    st.markdown(ui.callout(msg, kind), unsafe_allow_html=True)
    with st.expander("Why two scores? (leakage)"):
        st.write("FAERS reports arrive in batches - one study, one litigation, one manufacturer upload - and reports in a batch look "
                 "alike. An ordinary random split puts some of them in training and some in testing, so the model is partly tested on "
                 "copies of what it has already seen. The batch-aware test keeps every report received on the same date together, so "
                 "the model is always tested on batches it has never seen. The target is also never a term PharmaSense itself discards "
                 "as non-specific, and is skipped if most of its reports sit on one date.")

    if fp.exists():
        fi = pd.read_csv(fp).head(15).iloc[::-1]
        if "lasso_coef" in fi.columns:
            fi["Lasso says"] = np.where(fi["lasso_coef"] > 1e-9, "raises the odds",
                                np.where(fi["lasso_coef"] < -1e-9, "lowers the odds", "dropped (no effect)"))
            fig3 = px.bar(fi, x="importance", y="feature", orientation="h", color="Lasso says",
                          color_discrete_map={"raises the odds": RISK, "lowers the odds": TEAL, "dropped (no effect)": "#C9D6DA"})
        else:
            fig3 = px.bar(fi, x="importance", y="feature", orientation="h", color_discrete_sequence=[TEAL])
        fig3.update_layout(xaxis_title="Importance (XGBoost)", yaxis_title=None, legend=dict(y=-0.16, yanchor="top"))
        st.subheader("Which drugs the models leaned on")
        st.plotly_chart(style(fig3, 500).update_layout(legend=dict(orientation="h", y=-0.16, yanchor="top", x=0, title_text="")), width="stretch")
        st.caption("Importance shows which drugs the model used, not that they cause the reaction. Colours come from the Lasso: the "
                   "direction of each drug's effect once the other drugs are held fixed.")
    st.caption("Try another reaction: `python src/analysis/advanced_models.py --target NAUSEA` (needs `python load_to_duckdb.py` first).")


# ---------------------------------------------------------------- main
st.sidebar.markdown(ui.brand(), unsafe_allow_html=True)
AI_PAGES = list(app_ai.PAGES) if app_ai is not None else []
page = st.sidebar.radio("Page", ["Overview", "Drug search", "Combination signals", "Glossary & formulas", "ML models"] + AI_PAGES,
                        label_visibility="collapsed")
st.sidebar.markdown('<div class="ps-note"><b>New to these terms?</b> Open <i>Glossary &amp; formulas</i>.<br><br>'
                    'Research signals only. Association is not causation; not medical advice.</div>', unsafe_allow_html=True)

meta_file = RES / "meta.json"
if page in AI_PAGES:
    app_ai.PAGES[page]()
elif page == "ML models":
    page_ml()
elif not meta_file.exists():
    st.title("No results yet")
    st.info("Run `python run_pipeline.py` (or fetch_fda_api.py then process_and_mine.py), then refresh this page.")
else:
    D, T, M, X, meta = load(meta_file.stat().st_mtime)
    {"Overview": page_overview, "Drug search": page_search, "Combination signals": page_combos,
     "Glossary & formulas": page_glossary}[page](D, T, M, X, meta)
