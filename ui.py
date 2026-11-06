"""PharmaSense visual identity: one palette, one stylesheet, a few HTML building blocks, one Plotly style.

Theme idea: a clinical, pharmacy-lab look. Deep teal "ink" for structure, a capsule (two halves = two drugs) as the logo,
a benzene-ring hexagon grid as texture, and magenta reserved for risk so that colour always means something:
    teal = drug / neutral evidence   magenta = risk or combination effect   amber = pair / caution   green = supported

Used by app.py (owns set_page_config) and, through the shared CSS, by app_ai.py. No data logic lives here.
"""
from __future__ import annotations

from urllib.parse import quote

import streamlit as st

INK, TEAL, RISK, MUTE, AMBER, GREEN = "#12303A", "#0E7C86", "#C2185B", "#6B8791", "#E0A030", "#2E8B57"
AQUA, LINE, MINT, BG = "#5FD0C8", "#DCE8EA", "#E3F3F2", "#F4F9FA"
COLORWAY = [TEAL, RISK, AMBER, "#4E9AA3", GREEN, "#8E6FB5"]

# ------------------------------------------------------------------------------------------------ artwork
# Capsule: left half teal, right half magenta (two drugs, one report). Two plain paths, no ids, so it can repeat on a page.
def capsule(size: int = 40, rotate: int = -40) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
            f'<g transform="rotate({rotate} 32 32)">'
            f'<path d="M32 20H17a12 12 0 0 0 0 24H32Z" fill="{TEAL}"/><path d="M32 20H47a12 12 0 0 1 0 24H32Z" fill="{RISK}"/>'
            f'<rect x="9" y="24" width="38" height="4" rx="2" fill="#fff" opacity=".35"/></g></svg>')


def _hex_tile() -> str:
    """Benzene-ring style hexagon grid, tiled as the hero background."""
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="56" height="100" viewBox="0 0 56 100">'
           '<path d="M28 66L0 50L0 16L28 0L56 16L56 50L28 66L28 100" fill="none" stroke="#ffffff" stroke-opacity=".07" stroke-width="1.4"/>'
           '<path d="M28 0L28 34L0 50L0 84L28 100L56 84L56 50L28 34" fill="none" stroke="#ffffff" stroke-opacity=".07" stroke-width="1.4"/></svg>')
    return "url(\"data:image/svg+xml;utf8," + quote(svg) + "\")"


def _molecule(w: int = 300, h: int = 190) -> str:
    """Decorative drug-association network for the hero: three drug nodes joined to a reaction node."""
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 300 190" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
            f'<g stroke="#ffffff" stroke-opacity=".35" stroke-width="2" fill="none">'
            f'<path d="M55 45L150 95"/><path d="M55 145L150 95"/><path d="M150 20L150 95"/><path d="M150 95L245 95"/>'
            f'<path d="M245 95L275 45" stroke-dasharray="4 5"/></g>'
            f'<circle cx="55" cy="45" r="17" fill="{TEAL}" stroke="#fff" stroke-width="3"/>'
            f'<circle cx="55" cy="145" r="17" fill="{TEAL}" stroke="#fff" stroke-width="3"/>'
            f'<circle cx="150" cy="20" r="17" fill="{TEAL}" stroke="#fff" stroke-width="3"/>'
            f'<circle cx="150" cy="95" r="11" fill="#ffffff" fill-opacity=".92"/>'
            f'<circle cx="245" cy="95" r="22" fill="{RISK}" stroke="#fff" stroke-width="3"/>'
            f'<text x="245" y="101" text-anchor="middle" font-family="Segoe UI,sans-serif" font-weight="700" font-size="17" fill="#fff">AE</text>'
            f'<circle cx="275" cy="45" r="7" fill="{AMBER}"/></svg>')


# ------------------------------------------------------------------------------------------------ stylesheet
def _css() -> str:
    return f"""
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{{--ink:{INK};--teal:{TEAL};--risk:{RISK};--mute:{MUTE};--amber:{AMBER};--green:{GREEN};--aqua:{AQUA};--line:{LINE};--mint:{MINT};--bg:{BG};
  --shadow:0 1px 2px rgba(18,48,58,.05),0 10px 28px -14px rgba(18,48,58,.22)}}
html,body,[class*="css"],.stApp,.stMarkdown{{font-family:'IBM Plex Sans','Segoe UI',system-ui,sans-serif;color-scheme:light}}
.stApp{{background:radial-gradient(900px 420px at 92% -8%,#DDF1EF 0%,transparent 62%),radial-gradient(700px 380px at -5% 105%,#FBE7EF 0%,transparent 60%),{BG}}}
header[data-testid="stHeader"]{{background:transparent}}
footer{{visibility:hidden}}
.block-container{{max-width:1240px;padding-top:2rem;padding-bottom:4rem}}

/* ---- type */
h1,h2,h3,h4,h5{{font-family:'Bricolage Grotesque','Segoe UI',sans-serif!important;color:var(--ink)!important;letter-spacing:-.02em}}
h1{{font-weight:800;font-size:2.35rem!important}}
h1::before{{content:"";display:inline-block;width:38px;height:16px;border-radius:999px;margin-right:.7rem;vertical-align:middle;
  background:linear-gradient(90deg,var(--teal) 50%,var(--risk) 50%);transform:rotate(-32deg) translateY(-3px);box-shadow:0 4px 10px -3px rgba(194,24,91,.45)}}
h2{{font-weight:700;font-size:1.5rem!important;margin-top:1.4rem!important}}
h3{{font-weight:700}}
code{{font-family:'IBM Plex Mono',ui-monospace,monospace;background:var(--mint);color:var(--ink);border-radius:6px;padding:.08rem .35rem}}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{{color:#4B6770!important;opacity:1}}
a{{color:var(--teal)}}

/* ---- sidebar */
[data-testid="stSidebar"]{{background:linear-gradient(185deg,#0A222C 0%,#12303A 52%,#0E4F58 100%);border-right:0}}
[data-testid="stSidebar"] *{{color:#DCEBEE}}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{{color:#8FB8C0}}
.ps-brand{{display:flex;align-items:center;gap:.7rem;padding:.4rem .2rem 1.1rem}}
.ps-brand .name{{font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:1.45rem;color:#fff;line-height:1}}
.ps-brand .tag{{font-size:.72rem;color:#8FB8C0;letter-spacing:.02em;margin-top:.25rem;line-height:1.25}}
[data-testid="stSidebar"] [data-testid="stRadioGroup"]{{gap:.1rem}}
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div{{position:relative;width:100%}}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]{{display:flex;width:100%;padding:.5rem .85rem;border-radius:12px;transition:background .15s,transform .15s;cursor:pointer}}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover{{background:rgba(255,255,255,.08);transform:translateX(2px)}}
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"]{{background:linear-gradient(90deg,rgba(95,208,200,.30),rgba(95,208,200,.04));box-shadow:inset 3px 0 0 {AQUA}}}
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] p{{color:#fff;font-weight:600}}
[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child{{display:none}}
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(1){{margin-top:1.6rem}}
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(6){{margin-top:2.7rem}}
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(1)::before,
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(6)::before{{position:absolute;top:-1.3rem;left:.85rem;font-size:.66rem;letter-spacing:.16em;font-weight:600;color:{AQUA}}}
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(1)::before{{content:"SIGNAL STATISTICS"}}
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(6)::before{{content:"MECHANISM ENGINE"}}
.ps-note{{margin-top:1rem;padding:.8rem .9rem;border-radius:12px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.10);font-size:.78rem;line-height:1.4;color:#B9D3D8}}

/* ---- hero */
.hero{{position:relative;overflow:hidden;border-radius:22px;padding:2.2rem 2.6rem;margin-bottom:1.4rem;color:#fff;
  background:{_hex_tile()},linear-gradient(120deg,#0A222C 0%,#12303A 45%,#0E6A74 100%);box-shadow:0 18px 40px -20px rgba(10,34,44,.7)}}
.hero::after{{content:"";position:absolute;right:-60px;top:-60px;width:260px;height:260px;border-radius:50%;background:radial-gradient(circle,rgba(194,24,91,.35),transparent 65%)}}
.hero .eyebrow{{font-size:.74rem;letter-spacing:.18em;font-weight:600;color:{AQUA};text-transform:uppercase}}
.hero .big{{font-family:'Bricolage Grotesque',sans-serif;font-size:2.35rem;font-weight:800;line-height:1.12;margin:.55rem 0 .5rem;max-width:62%;color:#fff}}
.hero small{{color:#A9CBD1;font-size:.98rem}}
.hero .art{{position:absolute;right:1.6rem;top:50%;transform:translateY(-50%);opacity:.95;z-index:1}}
.hero .chips{{margin-top:1.1rem;display:flex;gap:.5rem;flex-wrap:wrap;position:relative;z-index:2;max-width:62%}}
.chip{{display:inline-block;padding:.22rem .75rem;border-radius:999px;font-size:.78rem;font-weight:500;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.22);color:#E6F4F5}}
@media (max-width:1000px){{.hero .art{{display:none}}.hero .big,.hero .chips{{max-width:100%}}.hero .big{{font-size:1.8rem}}}}

/* ---- cards, metrics, containers */
[data-testid="stMetric"]{{position:relative;overflow:hidden;background:#fff;border:1px solid var(--line);border-radius:16px;padding:1rem 1.15rem 1rem 1.35rem;box-shadow:var(--shadow)}}
[data-testid="stMetric"]::before{{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:linear-gradient(var(--teal),{AQUA})}}
[data-testid="stMetricLabel"] p{{color:var(--mute)!important;font-weight:500;font-size:.88rem}}
[data-testid="stMetricValue"]{{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;color:var(--ink)}}
[data-testid="stPlotlyChart"]{{background:#fff;border:1px solid var(--line);border-radius:16px;padding:.5rem .7rem;box-shadow:var(--shadow)}}
[data-testid="stDataFrame"]{{border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:var(--shadow)}}
[data-testid="stExpander"]{{background:#fff;border:1px solid var(--line)!important;border-radius:14px;box-shadow:0 1px 2px rgba(18,48,58,.04)}}
[data-testid="stExpander"] summary p{{font-weight:600}}
[data-testid="stAlert"]{{border-radius:14px;border:1px solid var(--line)}}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]){{border-radius:16px}}
div[class*="stVerticalBlock"][style*="border"]{{border-radius:16px!important;background:#fff;box-shadow:var(--shadow)}}

/* ---- tabs as pills */
[role="tablist"]{{gap:.45rem;padding-bottom:.5rem}}
[data-testid="stTabs"] > div:first-of-type{{border-bottom:0!important;box-shadow:none!important}}
[data-testid="stTab"]{{border-radius:999px!important;padding:.4rem 1.05rem!important;background:#fff;border:1px solid var(--line)!important;height:auto!important;transition:background .15s}}
[data-testid="stTab"]:hover{{background:var(--mint)}}
[data-testid="stTab"][data-selected="true"]{{background:var(--teal)!important;border-color:var(--teal)!important}}
[data-testid="stTab"][data-selected="true"] p{{color:#fff!important;font-weight:600}}
[data-testid="stTab"] .react-aria-SelectionIndicator{{display:none!important}}

/* ---- inputs */
[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-baseweb="textarea"]>div{{border-radius:12px!important;border-color:var(--line)!important;background:#fff!important}}
.stButton>button,.stDownloadButton>button{{border-radius:999px;font-weight:600;padding:.45rem 1.3rem;border:1px solid var(--teal);transition:transform .12s,box-shadow .12s}}
.stButton>button:hover{{transform:translateY(-1px);box-shadow:0 8px 18px -10px rgba(14,124,134,.8)}}
.stButton>button[kind="primary"]{{background:linear-gradient(120deg,var(--teal),#0A5F69);color:#fff}}

/* ---- building blocks */
.badge{{display:inline-flex;align-items:center;gap:.4rem;padding:.18rem .8rem;border-radius:999px;color:#fff;font-weight:600;font-size:.84rem;box-shadow:0 4px 10px -5px rgba(18,48,58,.55)}}
.badge::before{{content:"";width:.5rem;height:.5rem;border-radius:50%;background:rgba(255,255,255,.85)}}
.ps-card{{min-height:12.5rem;background:#fff;border:1px solid var(--line);border-radius:16px;padding:1.1rem 1.2rem;box-shadow:var(--shadow);height:100%;position:relative;overflow:hidden}}
.ps-card::before{{content:"";position:absolute;left:0;right:0;top:0;height:4px;background:var(--accent,var(--teal))}}
.ps-card .ico{{display:inline-flex;align-items:center;justify-content:center;width:2.1rem;height:2.1rem;border-radius:10px;background:var(--mint);margin-bottom:.5rem}}
.ps-card h4{{margin:.1rem 0 .35rem;font-size:1.08rem}}
.ps-card p{{margin:0;color:#35525B;font-size:.93rem;line-height:1.5}}
.ps-card .tagline{{display:inline-block;margin-top:.7rem;padding:.12rem .6rem;border-radius:999px;background:var(--mint);color:var(--teal);font-size:.76rem;font-weight:600}}
.ps-callout{{border-radius:14px;padding:.9rem 1.1rem;margin:.6rem 0 1rem;background:var(--mint);border:1px solid #BFE3E0;border-left:6px solid var(--teal);color:var(--ink);line-height:1.5}}
.ps-callout.warn{{background:#FFF6E4;border-color:#F2DAA5;border-left-color:var(--amber)}}
.ps-callout.risk{{background:#FDEBF2;border-color:#F4C4D6;border-left-color:var(--risk)}}
.ps-callout b{{font-family:'Bricolage Grotesque',sans-serif}}
"""


def inject() -> None:
    """Call once, right after st.set_page_config."""
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------ html blocks
def brand() -> str:
    return (f'<div class="ps-brand">{capsule(46)}<div><div class="name">PharmaSense</div>'
            f'<div class="tag">Drug-combination safety signals<br>and the mechanisms behind them</div></div></div>')


def hero(eyebrow: str, big: str, sub: str = "", chips: tuple[str, ...] = ()) -> str:
    c = "".join(f'<span class="chip">{x}</span>' for x in chips)
    return (f'<div class="hero"><div class="eyebrow">{eyebrow}</div><div class="big">{big}</div>'
            f'<small>{sub}</small>' + (f'<div class="chips">{c}</div>' if c else "") + f'<div class="art">{_molecule()}</div></div>')


def card(title: str, body: str, tagline: str = "", accent: str = TEAL, icon: str = "") -> str:
    ico = f'<div class="ico">{icon}</div>' if icon else ""
    tag = f'<span class="tagline">{tagline}</span>' if tagline else ""
    return f'<div class="ps-card" style="--accent:{accent}">{ico}<h4>{title}</h4><p>{body}</p>{tag}</div>'


def callout(html: str, kind: str = "") -> str:
    return f'<div class="ps-callout {kind}">{html}</div>'


# ------------------------------------------------------------------------------------------------ plotly
def style_fig(fig, h: int = 380):
    """One chart look for the whole app: transparent paper (the card behind it is white), soft grid, rounded bars."""
    fig.update_layout(
        height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=COLORWAY,
        font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color=INK, size=13), margin=dict(l=8, r=14, t=40, b=36),
        hoverlabel=dict(bgcolor=INK, font=dict(color="#fff", family="IBM Plex Sans, Segoe UI, sans-serif")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        title=dict(font=dict(family="Bricolage Grotesque, Segoe UI, sans-serif", size=16, color=INK), x=0.0, xanchor="left"),
        bargap=0.28)
    fig.update_xaxes(gridcolor="#E8F0F2", zerolinecolor="#C9D9DD", linecolor=LINE, tickcolor=LINE, automargin=True)
    fig.update_yaxes(gridcolor="#E8F0F2", zerolinecolor="#C9D9DD", linecolor=LINE, tickcolor=LINE, automargin=True)
    try:                                    # rounded bar ends need plotly >= 5.24; harmless to skip on older versions
        fig.update_traces(marker_cornerradius=6, selector=dict(type="bar"))
    except Exception:  # noqa: BLE001
        pass
    return fig
