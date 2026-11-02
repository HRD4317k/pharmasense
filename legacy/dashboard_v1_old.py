"""PharmaSense dashboard. Run from the repo root:  streamlit run dashboard_v1/app.py

Reads table `triplet_signals` from data/faers.duckdb when it exists (create it with src/analysis/export_for_dashboard.py), otherwise shows demo data.
Expected columns: drug_a, drug_b, drug_c, reaction, n_reports, ror, ror_lcl, max_pair_ror,
max_single_ror, interaction_coef, p_value  (q_value is computed with Benjamini-Hochberg if missing).
Optional: models/metrics.csv and models/feature_importance.csv (written by PharmaSense_Colab.ipynb).
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
INK, TEAL, RISK, MUTE = "#12303A", "#0E7C86", "#C2185B", "#6B8791"
KEYS = ["drug_a", "drug_b", "drug_c", "reaction"]
DRUGS = ["Warfarin", "Aspirin", "Clopidogrel", "Metformin", "Lisinopril", "Atorvastatin", "Omeprazole",
         "Amlodipine", "Furosemide", "Digoxin", "Amiodarone", "Simvastatin", "Ibuprofen", "Spironolactone"]
REACTIONS = ["Acute kidney injury", "GI haemorrhage", "Rhabdomyolysis", "Hyperkalaemia", "Bradycardia",
             "Hypoglycaemia", "QT prolongation", "Hepatotoxicity"]

st.set_page_config(page_title="PharmaSense", page_icon="💊", layout="wide")
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html,body,[class*="css"],.stMarkdown{{font-family:'IBM Plex Sans',sans-serif}}
h1,h2,h3{{font-family:'Bricolage Grotesque',sans-serif!important;color:{INK};letter-spacing:-.02em}}
.block-container{{max-width:1240px;padding-top:1.8rem}}
[data-testid=stSidebar]{{background:{INK}}}
[data-testid=stSidebar] *{{color:#DCEBEE!important}}
.hero{{background:{INK};color:#fff;border-radius:18px;padding:2.4rem 2.6rem;margin-bottom:1.4rem}}
.hero small{{color:#8FB8C0;font-size:.95rem}}
.combo{{font-family:'Bricolage Grotesque',sans-serif;font-size:2.4rem;font-weight:700;line-height:1.15;margin:.6rem 0}}
.combo i{{font-style:normal;color:#5FD0DB}} .combo em{{font-style:normal;color:#FF7AA8}}
.kpi{{background:#fff;border:1px solid #DCE6E9;border-radius:12px;padding:1rem 1.2rem}}
.kpi b{{display:block;font-family:'Bricolage Grotesque',sans-serif;font-size:2rem;color:{INK}}}
.kpi span{{color:{MUTE};font-size:.88rem}}
</style>""", unsafe_allow_html=True)


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum(np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1], 1)
    return q


@st.cache_data(show_spinner=False)
def load():
    db = ROOT / "data" / "faers.duckdb"
    if db.exists():
        try:
            import duckdb
            con = duckdb.connect(str(db), read_only=True)
            df = con.sql("SELECT * FROM triplet_signals").df()
            con.close()
            if "q_value" not in df:
                df["q_value"] = bh(df["p_value"])
            return df, False
        except Exception as e:
            st.sidebar.warning(f"Could not read triplet_signals: {e}")
    rng = np.random.default_rng(7); rows = []
    for _ in range(600):
        a, b, c = sorted(rng.choice(DRUGS, 3, replace=False)); n = int(rng.integers(5, 400))
        ror = rng.lognormal(.9, .8); pair = ror * rng.uniform(.4, 1.3); single = pair * rng.uniform(.5, 1.2)
        coef = math.log(ror / pair) + rng.normal(0, .2); z = abs(coef) / (1.5 / math.sqrt(n))
        rows.append(dict(drug_a=a, drug_b=b, drug_c=c, reaction=rng.choice(REACTIONS), n_reports=n, ror=ror,
                         ror_lcl=ror * math.exp(-3 / math.sqrt(n)), max_pair_ror=pair, max_single_ror=single,
                         interaction_coef=coef, p_value=math.erfc(z / math.sqrt(2))))
    df = pd.DataFrame(rows).drop_duplicates(KEYS).reset_index(drop=True)
    df["q_value"] = bh(df["p_value"])
    return df, True


def style(fig, h=380):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="IBM Plex Sans", color=INK), margin=dict(l=0, r=0, t=30, b=0))
    return fig


def kpi(col, value, label):
    col.markdown(f'<div class="kpi"><b>{value}</b><span>{label}</span></div>', unsafe_allow_html=True)


df, demo = load()
df["triplet"] = df.drug_a + " + " + df.drug_b + " + " + df.drug_c
g3 = df.ror_lcl > 1
g4 = g3 & (df.ror > df.max_pair_ror) & (df.ror > df.max_single_ror)
g5 = g4 & (df.q_value < .05) & (df.interaction_coef > 0)
df["passed"] = g5

st.sidebar.markdown("## 💊 PharmaSense")
st.sidebar.caption("3-drug interaction signals from FAERS")
page = st.sidebar.radio("View", ["Overview", "Signal explorer", "Heatmap", "ML models", "Glossary"],
                        label_visibility="collapsed")
if demo:
    st.sidebar.info("Showing demo data. Put your results in data/faers.duckdb (table triplet_signals) to see real signals.")

if page == "Overview":
    sig = df[g5].sort_values("ror", ascending=False)
    if len(sig):
        t = sig.iloc[0]
        st.markdown(f"""<div class="hero"><small>Strongest signal that passed all five gates</small>
        <div class="combo"><i>{t.drug_a}</i> + <i>{t.drug_b}</i> + <i>{t.drug_c}</i><br>→ <em>{t.reaction}</em></div>
        <small>ROR {t.ror:.1f} (95% CI lower bound {t.ror_lcl:.1f}) · {int(t.n_reports)} reports · q = {t.q_value:.3g}</small></div>""",
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="hero"><div class="combo">No triplet has passed all five gates yet.</div></div>',
                    unsafe_allow_html=True)
    c = st.columns(4)
    kpi(c[0], f"{len(df):,}", "Candidate triplet-reaction pairs")
    kpi(c[1], f"{int(g5.sum()):,}", "Validated synergistic signals")
    kpi(c[2], f"{df.n_reports.sum():,.0f}", "Supporting reports")
    kpi(c[3], f"{df[g5].ror.median():.1f}" if g5.any() else "–", "Median ROR of validated signals")
    left, right = st.columns([2, 3])
    with left:
        st.subheader("Gate funnel")
        f = go.Figure(go.Funnel(y=["Gates 1–2 · frequent", "Gate 3 · ROR lower CI > 1", "Gate 4 · beats pairs and singles",
                                   "Gate 5 · FDR-adjusted interaction"], x=[len(df), g3.sum(), g4.sum(), g5.sum()],
                                marker=dict(color=[MUTE, "#4E9AA3", TEAL, RISK]), textinfo="value"))
        st.plotly_chart(style(f, 360), width="stretch")
    with right:
        st.subheader("Top validated signals by ROR")
        top = sig.head(10).iloc[::-1]
        fig = px.bar(top, x="ror", y=top.triplet + " → " + top.reaction, orientation="h", color_discrete_sequence=[TEAL])
        fig.update_layout(yaxis_title=None, xaxis_title="Reporting odds ratio")
        st.plotly_chart(style(fig, 360), width="stretch")

elif page == "Signal explorer":
    st.title("Signal explorer")
    a, b, c = st.columns(3)
    rx = a.multiselect("Reaction", sorted(df.reaction.unique()))
    max_n = max(1, int(df.n_reports.max()))
    min_n = b.slider("Minimum reports", 0, max_n, min(10, max_n))
    only = c.toggle("Only validated signals", True)
    v = df[(df.n_reports >= min_n) & (g5 if only else True)]
    if rx:
        v = v[v.reaction.isin(rx)]
    v = v.sort_values("ror", ascending=False)
    cols = ["triplet", "reaction", "n_reports", "ror", "ror_lcl", "max_pair_ror", "interaction_coef", "q_value"]
    st.dataframe(v[cols], width="stretch", hide_index=True, height=300,
                 column_config={"ror": st.column_config.NumberColumn("ROR", format="%.2f"),
                                "ror_lcl": st.column_config.NumberColumn("CI lower", format="%.2f"),
                                "max_pair_ror": st.column_config.NumberColumn("Best pair ROR", format="%.2f"),
                                "interaction_coef": st.column_config.NumberColumn("Interaction β", format="%.2f"),
                                "q_value": st.column_config.NumberColumn("q (BH)", format="%.4f")})
    if len(v):
        st.subheader("Signal detail")
        pick = st.selectbox("Choose a signal", v.index, format_func=lambda i: f"{v.triplet[i]} → {v.reaction[i]}")
        r = df.loc[pick]
        l, rt = st.columns(2)
        fig = go.Figure(go.Bar(x=["Strongest single drug", "Strongest pair", "Triplet"],
                               y=[r.max_single_ror, r.max_pair_ror, r.ror], marker_color=[MUTE, "#4E9AA3", RISK]))
        fig.update_layout(title="Does the triplet add risk beyond its parts?", yaxis_title="ROR")
        l.plotly_chart(style(fig), width="stretch")
        if demo:
            yrs = np.arange(2014, 2025); rng = np.random.default_rng(int(r.n_reports))
            n = np.maximum(1, rng.poisson(r.n_reports / 11 * np.linspace(.4, 1.8, len(yrs))))
            fig = px.area(x=yrs, y=n, color_discrete_sequence=[TEAL], title="Reports per year (demo trend)")
            fig.update_layout(xaxis_title=None, yaxis_title="Reports")
            rt.plotly_chart(style(fig), width="stretch")
        else:
            rt.info("Add a `signal_trends` table to show reports per year here.")

elif page == "Heatmap":
    st.title("Drug × reaction heatmap")
    st.caption("Highest ROR among validated signals that include each drug.")
    long = df[g5].melt(id_vars=["reaction", "ror"], value_vars=["drug_a", "drug_b", "drug_c"], value_name="drug")
    if long.empty:
        st.info("No validated signals to plot.")
    else:
        m = long.pivot_table(index="drug", columns="reaction", values="ror", aggfunc="max")
        fig = px.imshow(m, color_continuous_scale=["#EAF3F4", TEAL, RISK], aspect="auto", labels=dict(color="ROR"))
        st.plotly_chart(style(fig, 520), width="stretch")

elif page == "ML models":
    st.title("Predicting severe ADRs")
    mp = ROOT / "models" / "metrics.csv"
    if mp.exists():
        met = pd.read_csv(mp)
    else:
        st.info("Showing placeholder numbers. Run PharmaSense_Colab.ipynb and save its metrics.csv to models/.")
        met = pd.DataFrame({"model": ["XGBoost", "Random Forest", "Lasso"], "auc": [.91, .88, .79],
                            "auprc": [.74, .69, .52], "f1": [.68, .64, .47]})
    fig = px.bar(met.melt("model", var_name="metric"), x="model", y="value", color="metric", barmode="group",
                 color_discrete_sequence=[TEAL, "#4E9AA3", RISK])
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    st.plotly_chart(style(fig), width="stretch")
    st.dataframe(met.round(3), hide_index=True, width="stretch")
    fp = ROOT / "models" / "feature_importance.csv"
    if fp.exists():
        fi = pd.read_csv(fp).head(15).iloc[::-1]
        st.plotly_chart(style(px.bar(fi, x="importance", y="feature", orientation="h",
                                     color_discrete_sequence=[TEAL], title="Top features (XGBoost)")), width="stretch")

else:
    st.title("Glossary")
    terms = {
        "FAERS": "FDA Adverse Event Reporting System: spontaneous reports of side effects from patients, doctors and manufacturers.",
        "ROR": "Reporting odds ratio. Odds of a reaction among reports with the drug(s) divided by the odds among reports without. Above 1 means the reaction is reported more often.",
        "FP-Growth": "Algorithm that finds drug sets that appear together often enough (minimum support) without testing every combination.",
        "Gate 3 and 4": "The triplet's ROR must be clearly above 1 and higher than every single drug and every pair inside it.",
        "Interaction coefficient": "The 3-way term in the logistic regression. A positive value means the three drugs together raise risk beyond their separate effects.",
        "Benjamini-Hochberg FDR": "Adjusts p-values (q-values) so that, among signals you accept, the expected share of false positives stays low.",
        "Disclaimer": "These are statistical signals from spontaneous reports. They do not prove causation and are not medical advice.",
    }
    for k, d in terms.items():
        with st.expander(k):
            st.write(d)
