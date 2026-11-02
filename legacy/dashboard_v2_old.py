import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INK, TEAL, RISK, MUTE = "#12303A", "#0E7C86", "#C2185B", "#6B8791"

st.set_page_config(page_title="PharmaSense Advanced", page_icon="🧬", layout="wide")

# Custom CSS for a cleaner UI/UX
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
.kpi{{background:#fff;border:1px solid #DCE6E9;border-radius:12px;padding:1rem 1.2rem;box-shadow: 0 4px 6px rgba(0,0,0,0.05)}}
.kpi b{{display:block;font-family:'Bricolage Grotesque',sans-serif;font-size:2rem;color:{INK}}}
.kpi span{{color:{MUTE};font-size:.88rem;text-transform:uppercase;letter-spacing:0.05em;font-weight:600}}
</style>""", unsafe_allow_html=True)

def kpi(col, value, label):
    col.markdown(f'<div class="kpi"><b>{value}</b><span>{label}</span></div>', unsafe_allow_html=True)

def style_fig(fig, h=380):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="IBM Plex Sans", color=INK), margin=dict(l=0, r=0, t=30, b=0))
    return fig

@st.cache_data
def load_data():
    path = ROOT / "data" / "significant_triplets.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

st.sidebar.markdown("## 🧬 PharmaSense *v2*")
st.sidebar.caption("Advanced Real-World Polypharmacy Mining")

df = load_data()

if df.empty:
    st.warning("No significant signals found yet. Please run `python fetch_fda_api.py` followed by `python process_and_mine.py`.")
else:
    df["triplet"] = df["drug_a"].str.title() + " + " + df["drug_b"].str.title() + " + " + df["drug_c"].str.title()
    df["reaction"] = df["reaction"].str.title()
    
    # Hero Section
    top_signal = df.sort_values("ror_lcl", ascending=False).iloc[0]
    st.markdown(f"""<div class="hero"><small>Most Critical Synergistic Interaction Detected</small>
    <div class="combo"><i>{top_signal.drug_a.title()}</i> + <i>{top_signal.drug_b.title()}</i> + <i>{top_signal.drug_c.title()}</i><br>→ <em>{top_signal.reaction.title()}</em></div>
    <small>Reporting Odds Ratio: {top_signal.ror:.1f} (Lower 95% CI: {top_signal.ror_lcl:.1f}) · {int(top_signal.n_reports)} FDA Reports</small></div>""",
                unsafe_allow_html=True)
                
    # KPI metrics
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, f"{len(df)}", "Validated Signals")
    kpi(c2, f"{df['reaction'].nunique()}", "Adverse Reactions")
    kpi(c3, f"{df['n_reports'].sum():,}", "Supporting Reports")
    kpi(c4, f"{df['ror'].max():.1f}", "Max Detected ROR")
    
    st.markdown("---")
    
    # Visualizations
    left, right = st.columns([2, 3])
    with left:
        st.subheader("Signal Distribution by Reaction")
        reaction_counts = df['reaction'].value_counts().head(8).reset_index()
        reaction_counts.columns = ['reaction', 'count']
        fig_pie = px.pie(reaction_counts, values='count', names='reaction', 
                         color_discrete_sequence=px.colors.sequential.Teal)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label', showlegend=False)
        st.plotly_chart(style_fig(fig_pie, 360), width="stretch")
        
    with right:
        st.subheader("Top Synergistic Triplets (by Lower Confidence Bound)")
        top_signals = df.sort_values("ror_lcl", ascending=False).head(10).iloc[::-1]
        fig_bar = px.bar(top_signals, x="ror_lcl", y="triplet", color="reaction",
                         orientation="h", color_discrete_sequence=[TEAL, RISK, "#4E9AA3", MUTE])
        fig_bar.update_layout(yaxis_title=None, xaxis_title="Reporting Odds Ratio (Lower 95% CI)",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(style_fig(fig_bar, 360), width="stretch")
    
    st.subheader("Explore the Data")
    
    # Filters
    col_a, col_b = st.columns([1, 2])
    lo, hi = int(df.n_reports.min()), int(df.n_reports.max())
    min_reports = col_a.slider("Minimum Supporting Reports", min_value=lo, max_value=max(hi, lo + 1), value=lo)
    selected_reactions = col_b.multiselect("Filter by Reaction", options=sorted(df.reaction.unique()))
    
    filtered_df = df[df.n_reports >= min_reports]
    if selected_reactions:
        filtered_df = filtered_df[filtered_df.reaction.isin(selected_reactions)]
        
    st.dataframe(
        filtered_df[["triplet", "reaction", "n_reports", "ror", "ror_lcl"]].sort_values("ror", ascending=False),
        width="stretch", hide_index=True,
        column_config={
            "triplet": "Drug Combination",
            "reaction": "Adverse Reaction",
            "n_reports": st.column_config.NumberColumn("Reports", format="%d"),
            "ror": st.column_config.NumberColumn("ROR", format="%.2f"),
            "ror_lcl": st.column_config.NumberColumn("ROR Lower 95% CI", format="%.2f"),
        }
    )
