"""PharmaSense statistical engine (shared by process_and_mine.py and app.py).

Contents
  1. Formulas      ror(), prr(), ic()          - textbook disproportionality measures
  2. Cleaning      normalize_drug(), prepare_reports()
  3. Attribution   evaluate_triplets(), classify_triplets()  - single drug vs pair vs combination
  4. Orchestration run_mining(), save_results()

Every formula is documented in plain language on the app's "Glossary & formulas" page.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from string import capwords

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom, norm
from sklearn.preprocessing import MultiLabelBinarizer
from statsmodels.stats.multitest import multipletests

import exclusions

Z95 = 1.959963984540054          # 97.5th percentile of the standard normal
CATS = ["REF", "A", "B", "AB", "C", "AC", "BC", "ABC"]   # index == bitmask (A=1, B=2, C=4)
SUBS = ["A", "B", "C", "AB", "AC", "BC"]                 # the six strict sub-groups of a triplet
SPARSE_N = 10                    # a sub-group with fewer reports than this is "too small to compare"


# ----------------------------------------------------------------------------------------
# 1. FORMULAS
# ----------------------------------------------------------------------------------------
def _f(*xs):
    return [np.asarray(x, dtype=float) for x in xs]


def correct_zero_cells(a, b, c, d):
    """Haldane-Anscombe: add 0.5 to ALL four cells, but ONLY when some cell is zero."""
    a, b, c, d = _f(a, b, c, d)
    add = np.where((a == 0) | (b == 0) | (c == 0) | (d == 0), 0.5, 0.0)
    return a + add, b + add, c + add, d + add


def ror(a, b, c, d):
    """Reporting odds ratio with 95% CI (Woolf).  2x2:  a=exposed+event  b=exposed+no event
    c=unexposed+event  d=unexposed+no event.  Returns (ROR, lower, upper)."""
    a, b, c, d = correct_zero_cells(a, b, c, d)
    ln = np.log(a) + np.log(d) - np.log(b) - np.log(c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return np.exp(ln), np.exp(ln - Z95 * se), np.exp(ln + Z95 * se)


def prr(a, b, c, d):
    """Proportional reporting ratio, 95% CI, and Yates-corrected chi-square.
    Returns (PRR, lower, upper, chi2).  Zero correction only if a or c is zero."""
    a0, b0, c0, d0 = _f(a, b, c, d)
    add = np.where((a0 == 0) | (c0 == 0), 0.5, 0.0)
    a_, b_, c_, d_ = a0 + add, b0 + add, c0 + add, d0 + add
    est = (a_ / (a_ + b_)) / (c_ / (c_ + d_))
    se = np.sqrt(1 / a_ - 1 / (a_ + b_) + 1 / c_ - 1 / (c_ + d_))
    n = a0 + b0 + c0 + d0
    den = (a0 + b0) * (c0 + d0) * (a0 + c0) * (b0 + d0)
    num = n * np.maximum(np.abs(a0 * d0 - b0 * c0) - n / 2, 0) ** 2
    chi2 = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return est, np.exp(np.log(est) - Z95 * se), np.exp(np.log(est) + Z95 * se), chi2


def ic(a, n_drug, n_event, n_total):
    """Information Component (Bayesian shrinkage; 0 = no association) with the standard
    approximate 95% bounds.  Returns (IC, IC025, IC975)."""
    a, n_drug, n_event, n_total = _f(a, n_drug, n_event, n_total)
    expected = n_drug * n_event / n_total
    val = np.log2((a + 0.5) / (expected + 0.5))
    lo = val - 3.3 * (a + 0.5) ** -0.5 - 2.0 * (a + 0.5) ** -1.5
    hi = val + 2.4 * (a + 0.5) ** -0.5 - 0.5 * (a + 0.5) ** -1.5
    return val, lo, hi


# ----------------------------------------------------------------------------------------
# 2. CLEANING
# ----------------------------------------------------------------------------------------
FORM_WORDS = {"TABLET", "TABLETS", "CAPSULE", "CAPSULES", "INJECTION", "INJECTABLE", "SOLUTION", "ORAL",
              "CREAM", "OINTMENT", "GEL", "PATCH", "SPRAY", "SUSPENSION", "SYRUP", "POWDER", "FILM",
              "COATED", "EXTENDED", "RELEASE", "DELAYED", "ER", "XR", "SR", "XL", "DR", "USP", "FOR",
              "INTRAVENOUS", "SUBCUTANEOUS", "TOPICAL", "INJ", "KIT", "VIAL"}
SALT_WORDS = {"HYDROCHLORIDE", "HCL", "HYDROBROMIDE", "SODIUM", "POTASSIUM", "CALCIUM", "MAGNESIUM",
              "SULFATE", "SULPHATE", "PHOSPHATE", "MALEATE", "FUMARATE", "SUCCINATE", "TARTRATE",
              "MESYLATE", "BESYLATE", "CITRATE", "ACETATE", "BROMIDE", "NITRATE", "MONOHYDRATE",
              "DIHYDRATE", "HYDRATE", "ANHYDROUS", "DISODIUM", "TROMETHAMINE"}
JUNK_DRUGS = {"UNKNOWN", "UNSPECIFIED", "UNK", "NOT SPECIFIED", "DRUG NOT SPECIFIED", "NONE", "N/A",
              "NA", "UNSPECIFIED DRUG", "PRODUCT", "DRUG", "UNKNOWN DRUG", "CONCOMITANT DRUG"}
GENERIC_STOP = {"HUMAN PRESCRIPTION DRUG", "HUMAN OTC DRUG", "UNKNOWN", "UNSPECIFIED"}
NO_INDICATION = {"PRODUCT USED FOR UNKNOWN INDICATION", "UNKNOWN", "UNSPECIFIED", ""}

_DOSE_UNITS = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:MG|MCG|UG|G|KG|ML|L|IU|UNITS?|U|MEQ|MMOL)\b")
_DOSE_PCT = re.compile(r"\d+(?:[.,]\d+)?\s*%")


def normalize_drug(raw):
    """Collapse brand/strength/dosage-form/salt variants to one ingredient-level name.
    'METFORMIN HCL 500 MG TABLET' and 'METFORMIN HYDROCHLORIDE' -> 'METFORMIN'."""
    if not raw:
        return ""
    s = str(raw).upper()
    s = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", s)
    s = _DOSE_PCT.sub(" ", _DOSE_UNITS.sub(" ", s))
    tokens = [t.strip(".-") for t in re.split(r"[\s,;/&+]+", s)]
    tokens = [t for t in tokens if t and t not in FORM_WORDS and not t.isdigit()]
    while len(tokens) > 1 and tokens[-1] in SALT_WORDS:
        tokens.pop()
    name = " ".join(tokens)
    return "" if (len(name) < 3 or name in JUNK_DRUGS) else name


_STOP_TOK = {"AND", "WITH", "PLUS", "OF"} | SALT_WORDS | FORM_WORDS


def ingredient_tokens(name):
    return {t for t in re.split(r"[\s,;/&+]+", name) if len(t) >= 4 and t not in _STOP_TOK}


def overlapping(n1, n2):
    """True if one name's ingredients are contained in the other's (LISINOPRIL vs
    LISINOPRIL AND HYDROCHLOROTHIAZIDE): they are the same drug, not two drugs."""
    t1, t2 = ingredient_tokens(n1), ingredient_tokens(n2)
    return bool(t1) and bool(t2) and (t1 <= t2 or t2 <= t1)


def disp(name):
    """Display form of a drug name."""
    return capwords(str(name).lower())


def disp_rx(term):
    """Display form of a reaction term."""
    return str(term).capitalize()


def _ls(x):
    if x is None:
        return []
    try:
        return list(x)
    except TypeError:
        return []


def prepare_reports(raw, drug_scope="all"):
    """raw: one row per report (see fetch_fda_api.py).  Returns (reports, excluded_terms, info).
    Per report: ingredient-level drug set (>= 3 distinct drugs required) and the set of
    genuine reactions (non-AE terms and the report's own indications removed)."""
    need = {"report_id", "date", "drug_names", "drug_generic", "drug_roles", "drug_indications", "reactions"}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"Raw file is missing columns {sorted(missing)} - re-run fetch_fda_api.py.")
    excluded = Counter()
    rows, few_drugs, no_rx = [], 0, 0
    for rec in raw.to_dict("records"):
        drugs = set()
        for nm, gn, ro in zip(_ls(rec["drug_names"]), _ls(rec["drug_generic"]), _ls(rec["drug_roles"])):
            if drug_scope == "suspect" and str(ro) not in ("1", "3"):   # 1 = suspect, 3 = interacting
                continue
            gn = str(gn or "").strip().upper()
            d = normalize_drug(gn) if gn and gn not in GENERIC_STOP else ""
            d = d or normalize_drug(nm)
            if d:
                drugs.add(d)
        if len(drugs) < 3:
            few_drugs += 1
            continue
        inds = {str(i).strip().upper() for i in _ls(rec["drug_indications"])} - NO_INDICATION
        rx = set()
        for r in _ls(rec["reactions"]):
            r = str(r).strip().upper()
            if not r:
                continue
            cat = exclusions.classify(r)
            if cat:
                excluded[(r, cat)] += 1
            elif r in inds:
                excluded[(r, "Indication of a drug in the same report")] += 1
            else:
                rx.add(r)
        if not rx:
            no_rx += 1
            continue
        date = str(rec.get("date") or "")
        rows.append({"report_id": rec["report_id"], "date": date,
                     "month": f"{date[:4]}-{date[4:6]}" if len(date) >= 6 else "",
                     "patient_sex": rec.get("patient_sex"), "patient_age": rec.get("patient_age"),
                     "drugs": sorted(drugs), "reactions": sorted(rx)})
    reports = pd.DataFrame(rows)
    exc = (pd.DataFrame([{"reaction": k[0], "category": k[1], "occurrences": v} for k, v in excluded.items()])
           .sort_values("occurrences", ascending=False) if excluded else
           pd.DataFrame(columns=["reaction", "category", "occurrences"]))
    info = {"raw_reports": int(len(raw)), "kept_reports": int(len(reports)),
            "dropped_fewer_than_3_distinct_drugs": few_drugs, "dropped_no_genuine_reaction": no_rx,
            "excluded_reaction_occurrences": int(exc["occurrences"].sum()) if len(exc) else 0,
            "drug_scope": drug_scope}
    return reports, exc, info


# ----------------------------------------------------------------------------------------
# 3. SINGLE-DRUG STATISTICS
# ----------------------------------------------------------------------------------------
def drug_reaction_stats(X, drugs, Y, reactions, min_events=3, min_drug_reports=10):
    """ROR / PRR / IC for every (drug, reaction) pair with >= min_events reports."""
    N = X.shape[0]
    n_drug = np.asarray(X.sum(0)).ravel()
    n_rx = np.asarray(Y.sum(0)).ravel()
    C = (X.T.astype(np.int32) @ Y.astype(np.int32)).tocoo()
    m = (C.data >= min_events) & (n_drug[C.row] >= min_drug_reports)
    di, ri, a = C.row[m], C.col[m], C.data[m].astype(float)
    nd, nr = n_drug[di].astype(float), n_rx[ri].astype(float)
    b, c, d = nd - a, nr - a, N - nd - nr + a
    r, rl, ru = ror(a, b, c, d)
    p, pl, pu, chi2 = prr(a, b, c, d)
    i, il, iu = ic(a, nd, nr, N)
    out = pd.DataFrame({
        "drug": np.asarray(drugs)[di], "reaction": np.asarray(reactions)[ri],
        "a": a.astype(int), "n_drug": nd.astype(int), "n_reaction": nr.astype(int), "N": N,
        "pct_of_drug_reports": 100 * a / nd,
        "ror": r, "ror_lcl": rl, "ror_ucl": ru, "prr": p, "prr_lcl": pl, "prr_ucl": pu, "chi2": chi2,
        "ic": i, "ic025": il, "ic975": iu})
    out["signal"] = (out["ror_lcl"] > 1) & (out["a"] >= min_events)
    return out.sort_values(["drug", "a"], ascending=[True, False]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# 4. ATTRIBUTION: single drug, pair, or combination?
# ----------------------------------------------------------------------------------------
def evaluate_triplets(X, Y, drug_names, rx_names, triplets, min_events=3, log=print):
    """For each triplet (i, j, k) split ALL reports into 8 mutually exclusive groups by which of
    the three drugs they list (none / A only / B only / A+B only / C only / ... / all three), and
    count reports with each reaction in each group.  Returns one row per (triplet, reaction) that
    has >= min_events reports in the all-three group."""
    N = X.shape[0]
    Ycsr = Y.tocsr()
    Ycsc = Y.tocsc()
    n_rx = np.asarray(Y.sum(0)).ravel()
    weights = np.array([1, 2, 4])
    blocks = []
    for t, (i, j, k) in enumerate(triplets):
        M = X[:, [i, j, k]].toarray().astype(np.int8)
        code = (M * weights).sum(1)
        rows7 = np.flatnonzero(code == 7)
        if rows7.size == 0:
            continue
        cnt7 = np.asarray(Ycsr[rows7].sum(0)).ravel()
        cand = np.flatnonzero(cnt7 >= min_events)
        if cand.size == 0:
            continue
        G = sparse.csr_matrix((np.ones(N, dtype=np.float32), (np.arange(N), code)), shape=(N, 8))
        n_k = np.asarray(G.sum(0)).ravel()
        EV = np.asarray((G.T @ Ycsc[:, cand]).todense())            # 8 x cand
        blocks.append(_triplet_block([drug_names[i], drug_names[j], drug_names[k]],
                                     np.asarray(rx_names)[cand], n_rx[cand], n_k, EV))
        if (t + 1) % 500 == 0:
            log(f"  evaluated {t + 1}/{len(triplets)} triplets")
    if not blocks:
        return pd.DataFrame()
    return add_exact_tests(pd.concat(blocks, ignore_index=True), min_events)


def add_exact_tests(res, min_events=3):
    """One-sided exact test of the all-three group against the reference group, then Benjamini-Hochberg over every
    row. Factored out of evaluate_triplets so a single combination counted from the whole openFDA database
    (faers_live.py) goes through exactly the same test."""
    # Is the all-three group really above the reference group?  One-sided EXACT (Fisher / hypergeometric)
    # test, not a Wald test: Wald is badly anti-conservative at small counts.  Because only combinations
    # with >= min_events cases are tested, the p-value is conditioned on that filter (truncated null),
    # then Benjamini-Hochberg-adjusted over EVERY combination tested.
    M = (res["n_ABC"] + res["n_REF"]).to_numpy()
    K = (res["a_ABC"] + res["a_REF"]).to_numpy()
    n_draw, a_obs = res["n_ABC"].to_numpy(), res["a_ABC"].to_numpy()
    p_obs = hypergeom.sf(a_obs - 1, M, K, n_draw)
    p_min = hypergeom.sf(min_events - 1, M, K, n_draw)
    res["p_ror"] = np.where(p_min > 0, np.minimum(p_obs / np.where(p_min > 0, p_min, 1), 1.0), 1.0)
    res["q_ror"] = multipletests(res["p_ror"].to_numpy(), method="fdr_bh")[1]
    return res




def _triplet_block(names, rx, n_rx_total, n_k, EV):
    c = EV.shape[1]
    a = EV.astype(float)
    b = n_k[:, None].astype(float) - a
    d = {"drug_a": [names[0]] * c, "drug_b": [names[1]] * c, "drug_c": [names[2]] * c,
         "reaction": rx, "n_reaction_total": n_rx_total.astype(int)}
    for code, cat in enumerate(CATS):
        d[f"n_{cat}"] = np.full(c, int(n_k[code]))
        d[f"a_{cat}"] = a[code].astype(int)
    for code in range(1, 8):                       # each group vs the reference (none of the 3 drugs)
        r, lo, hi = ror(a[code], b[code], a[0], b[0])
        d[f"ror_{CATS[code]}"], d[f"lcl_{CATS[code]}"], d[f"ucl_{CATS[code]}"] = r, lo, hi
    for code in range(1, 7):                       # all-three group vs each strict sub-group
        h, lo, hi = ror(a[7], b[7], a[code], b[code])
        d[f"h2h_{CATS[code]}"], d[f"h2h_lcl_{CATS[code]}"], d[f"h2h_ucl_{CATS[code]}"] = h, lo, hi
    # three-way interaction in the saturated logistic model (exact, no fitting needed)
    zero_any = ((a == 0) | (b == 0)).any(axis=0)
    add = np.where(zero_any, 0.5, 0.0)
    ac, bc = a + add, b + add
    L = np.log(ac) - np.log(bc)
    beta = L[7] - L[3] - L[5] - L[6] + L[1] + L[2] + L[4] - L[0]
    se = np.sqrt((1 / ac + 1 / bc).sum(axis=0))
    d["beta3"], d["se_beta3"] = beta, se
    d["p_beta3"] = 2 * norm.sf(np.abs(beta / se))
    return pd.DataFrame(d)


def group_label(cat, A, B, C, short=True):
    """Human label of an exclusive group, e.g. 'Amiodarone + Warfarin only'."""
    m = {"A": [A], "B": [B], "C": [C], "AB": [A, B], "AC": [A, C], "BC": [B, C], "ABC": [A, B, C], "REF": []}
    if cat == "REF":
        return "None of the three"
    if cat == "ABC":
        return "All three together"
    return " + ".join(disp(x) for x in m[cat]) + " only"


def _fx(x):
    return f"{x:,.0f}" if x >= 100 else (f"{x:.1f}" if x >= 10 else f"{x:.2f}")


def classify_triplets(res, min_events=3):
    """Add verdict / driver / justification / quality flags to the evaluate_triplets() output."""
    n = len(res)
    res["ror"], res["ror_lcl"], res["ror_ucl"] = res["ror_ABC"], res["lcl_ABC"], res["ucl_ABC"]
    res["n_reports"], res["n_events"] = res["n_ABC"], res["a_ABC"]
    verdict, codes, drivers, texts, flags = ["No signal"] * n, [""] * n, [""] * n, [""] * n, [""] * n
    sig = ((res["a_ABC"] >= min_events) & (res["lcl_ABC"] > 1) & (res["q_ror"] < 0.05)).to_numpy()
    # 3-way interaction test: FDR over the signals only (gatekeeping - it is only interpreted for signals)
    q3 = np.full(n, np.nan)
    if sig.any():
        q3[sig] = multipletests(res.loc[sig, "p_beta3"].to_numpy(), method="fdr_bh")[1]
    res["q_beta3"] = q3
    # Does the all-three group beat EVERY sub-group?  Intersection-union test: p = the WORST of the six one-sided
    # exact head-to-head p-values; then BH over the signals, so ~2% of triplets cannot pass by chance.
    a7, n7 = res["a_ABC"].to_numpy(), res["n_ABC"].to_numpy()
    p_each = []
    for sub in SUBS:
        ak, nk = res[f"a_{sub}"].to_numpy(), res[f"n_{sub}"].to_numpy()
        p_each.append(hypergeom.sf(a7 - 1, n7 + nk, a7 + ak, n7))
    p_iut = np.max(p_each, axis=0)
    qh = np.full(n, np.nan)
    if sig.any():
        qh[sig] = multipletests(p_iut[sig], method="fdr_bh")[1]
    res["q_h2h"] = qh
    for idx in np.flatnonzero(sig):
        r = res.iloc[idx]
        A, B, C = r["drug_a"], r["drug_b"], r["drug_c"]
        nm = {"A": A, "B": B, "C": C}
        beats = {s: r[f"h2h_lcl_{s}"] > 1 for s in SUBS}
        sub_sig = {s: (r[f"a_{s}"] >= min_events) and (r[f"lcl_{s}"] > 1) for s in SUBS}
        confirmed = (r["beta3"] > 0) and (r["q_beta3"] < 0.05)
        unbeaten = [s for s in SUBS if sub_sig[s] and not beats[s]]
        singles = [s for s in unbeaten if len(s) == 1]
        pairs = [s for s in unbeaten if len(s) == 2]
        sparse_groups = [s for s in SUBS if r[f"n_{s}"] < SPARSE_N]
        if all(beats.values()):
            v, code = ("Combination effect" if r["q_h2h"] < 0.05 else "Probable combination"), "ABC"
        elif singles:
            v, code = "Single-drug effect", ",".join(sorted(singles))
        elif pairs:
            best = max(pairs, key=lambda s: r[f"ror_{s}"])
            v, code = "Pair effect", best
        elif any(sub_sig.values()):
            v, code = "Probable combination", "ABC"   # beats every sub-group that has a signal; others too small
        else:
            v, code = "Cannot separate", ""
        verdict[idx], codes[idx] = v, code
        names = [nm[ch] for part in code.split(",") if part for ch in part] if code and code != "ABC" else \
                ([A, B, C] if code == "ABC" else [])
        drivers[idx] = " | ".join(dict.fromkeys(names))
        texts[idx] = _explain(r, nm, v, code, beats, sub_sig, confirmed, sparse_groups)
        f = []
        if r["a_ABC"] < 5:
            f.append("fewer than 5 cases")
        if sparse_groups:
            f.append("some sub-groups have <10 reports")
        if r["a_ABC"] / max(r["n_reaction_total"], 1) >= 0.5:
            f.append("over half of all reports of this reaction involve this triplet")
        if r["a_REF"] < 5:
            f.append("reaction almost never reported without these drugs, so the ratio is unstable")
        if r["ror_ABC"] >= 50:
            f.append("very large ROR - check for clustering/duplicates")
        flags[idx] = "; ".join(f)
    texts = [t.replace("{N_TESTS}", f"{n:,}") for t in texts]
    res["verdict"], res["driver_codes"], res["driver_drugs"] = verdict, codes, drivers
    res["justification"], res["flags"] = texts, flags
    return res


def _explain(r, nm, verdict, code, beats, sub_sig, confirmed, sparse_groups):
    A, B, C = nm["A"], nm["B"], nm["C"]
    rx = disp_rx(r["reaction"]).lower()
    lab = lambda s: group_label(s, A, B, C)
    s = (f"{int(r['n_ABC'])} reports list all three drugs and {int(r['a_ABC'])} of them mention {rx}. "
         f"Compared with reports that list none of the three, the odds are {_fx(r['ror_ABC'])}x higher "
         f"(95% CI {_fx(r['lcl_ABC'])}-{_fx(r['ucl_ABC'])}; exact test, still significant after correcting for the "
         f"{{N_TESTS}} combinations tested, q = {r['q_ror']:.2g}). "
         f"On their own: " + "; ".join(
             f"{lab(x)} {_fx(r[f'ror_{x}'])}x ({_fx(r[f'lcl_{x}'])}-{_fx(r[f'ucl_{x}'])}, {int(r[f'n_{x}'])} reports)"
             for x in SUBS) + ". ")
    min_h = min(r[f"h2h_lcl_{x}"] for x in SUBS)
    interaction = (f"The 3-way interaction test also supports this (q = {r['q_beta3']:.2g})." if confirmed else
                   f"The 3-way interaction test is not significant (q = {r['q_beta3']:.2g}); that test has low "
                   f"power when a sub-group has few cases, so it does not overturn the finding.")
    if verdict == "Combination effect":
        s += (f"The all-three group is significantly higher than every single drug and every pair "
              f"(weakest head-to-head lower bound {_fx(min_h)} > 1; still significant after correcting for the many "
              f"comparisons, q = {r['q_h2h']:.2g}). The excess risk appears only when all three are taken together. " + interaction)
    elif verdict == "Probable combination" and all(beats.values()):
        s += (f"The all-three group is higher than every single drug and pair in each comparison, but after correcting "
              f"for the many comparisons made the difference is not conclusive (q = {r['q_h2h']:.2g}). " + interaction)
    elif verdict == "Probable combination":
        s += ("The all-three group is significantly higher than every sub-group that shows its own signal, "
              "but some sub-groups are too small to compare ("
              + ", ".join(lab(x) for x in sparse_groups) + "), so a smaller cause cannot be fully ruled out.")
    elif verdict == "Single-drug effect":
        who = " and ".join(disp(nm[ch]) for ch in code.split(","))
        s += (f"{who} on its own already shows the signal and the all-three group is not significantly higher "
              f"(head-to-head lower bound {_fx(min(r[f'h2h_lcl_{x}'] for x in code.split(',')))} <= 1). "
              f"The other drugs look like bystanders.")
    elif verdict == "Pair effect":
        s += (f"{' + '.join(disp(nm[ch]) for ch in code)} without the third drug already shows the signal and the "
              f"all-three group is not significantly higher (head-to-head lower bound "
              f"{_fx(r[f'h2h_lcl_{code}'])} <= 1). The third drug looks like a bystander.")
    else:
        s += ("No single drug or pair has enough reports of its own to compare against"
              + (f" ({', '.join(lab(x) for x in sparse_groups)})" if sparse_groups else "")
              + ". The three drugs are almost always reported together, so this data cannot say whether one "
                "drug or the combination is responsible.")
    return s


def evidence_table(row):
    """Per-group evidence for the 'evidence ladder' chart. row: a result row (Series/dict)."""
    A, B, C = row["drug_a"], row["drug_b"], row["drug_c"]
    out = [{"group": group_label("REF", A, B, C), "cat": "REF", "n": int(row["n_REF"]), "events": int(row["a_REF"]),
            "ror": 1.0, "lcl": 1.0, "ucl": 1.0}]
    for cat in ["A", "B", "C", "AB", "AC", "BC", "ABC"]:
        out.append({"group": group_label(cat, A, B, C), "cat": cat, "n": int(row[f"n_{cat}"]),
                    "events": int(row[f"a_{cat}"]), "ror": row[f"ror_{cat}"],
                    "lcl": row[f"lcl_{cat}"], "ucl": row[f"ucl_{cat}"]})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------
# 4b. DRUG + PARTNER (pair) ATTRIBUTION and co-medication-adjusted odds ratio  (used by the Drug search page)
# ----------------------------------------------------------------------------------------
PAIR_VERDICTS = ["Pair effect", "Drug alone", "Both alone", "Partner alone", "Cannot separate", "No signal"]


def build_matrices(reports):
    """Sparse report x drug and report x reaction matrices (csc) from the cleaned reports."""
    mlb_d, mlb_r = MultiLabelBinarizer(sparse_output=True), MultiLabelBinarizer(sparse_output=True)
    X = mlb_d.fit_transform(reports["drugs"]).astype(np.int32).tocsc()
    Y = mlb_r.fit_transform(reports["reactions"]).astype(np.int32).tocsc()
    return X, Y, mlb_d.classes_, mlb_r.classes_


def pair_evidence(drug, partner, g_n, g_a):
    """Evidence table (same shape as evidence_table) for the 4 exclusive groups of a drug + partner."""
    lab = {"REF": "Neither drug", "A": f"{disp(drug)} only", "B": f"{disp(partner)} only", "AB": "Both drugs"}
    rows = [{"group": lab["REF"], "cat": "REF", "n": g_n["REF"], "events": g_a["REF"], "ror": 1.0, "lcl": 1.0, "ucl": 1.0}]
    for k in ("A", "B", "AB"):
        r, lo, hi = ror(g_a[k], g_n[k] - g_a[k], g_a["REF"], g_n["REF"] - g_a["REF"])
        rows.append({"group": lab[k], "cat": k, "n": g_n[k], "events": g_a[k], "ror": float(r), "lcl": float(lo), "ucl": float(hi)})
    return pd.DataFrame(rows)


def pair_attribution(X, Y, drugs, rxs, drug, reaction, top_k=15, min_pair=10, min_events=3, exclude=()):
    """For one drug and one reaction: for each frequent partner P, split ALL reports into four exclusive
    groups (neither / drug without P / P without drug / both) and decide whether the reaction goes with
    the drug alone, the partner alone, or the pair together.  Returns one row per partner."""
    drugs = list(drugs)
    di, ri = drugs.index(drug), list(rxs).index(reaction)
    keep = np.ones(X.shape[0], dtype=bool)                 # optionally set aside reports with chosen partners
    for e in exclude:
        keep &= ~(X[:, e].toarray().ravel() > 0)
    xd = (X[:, di].toarray().ravel() > 0) & keep
    y = Y[:, ri].toarray().ravel() > 0
    co = np.asarray(X[xd].sum(0)).ravel()
    co[di] = 0
    for e in exclude:
        co[e] = 0
    rows = []
    for pi in np.argsort(-co)[: top_k * 2]:
        if co[pi] < min_pair or len(rows) >= top_k:
            continue
        if overlapping(drug, drugs[pi]):
            continue
        xp = X[:, pi].toarray().ravel() > 0
        masks = {"REF": ~xd & ~xp & keep, "A": xd & ~xp, "B": ~xd & xp & keep, "AB": xd & xp}
        g_n = {k: int(m.sum()) for k, m in masks.items()}
        g_a = {k: int((m & y).sum()) for k, m in masks.items()}
        E = pair_evidence(drug, drugs[pi], g_n, g_a)
        e = E.set_index("cat")
        h_a = ror(g_a["AB"], g_n["AB"] - g_a["AB"], g_a["A"], g_n["A"] - g_a["A"])
        h_b = ror(g_a["AB"], g_n["AB"] - g_a["AB"], g_a["B"], g_n["B"] - g_a["B"])
        def exact_p(k):                      # one-sided exact test of group k vs reference, conditioned on >= min_events
            M_, K_, n_ = g_n[k] + g_n["REF"], g_a[k] + g_a["REF"], g_n[k]
            p_obs, p_min = hypergeom.sf(g_a[k] - 1, M_, K_, n_), hypergeom.sf(min_events - 1, M_, K_, n_)
            return float(min(p_obs / p_min, 1.0)) if p_min > 0 else 1.0
        p, p_a, p_b = exact_p("AB"), exact_p("A"), exact_p("B")
        rows.append({"partner": drugs[pi], "n_both": g_n["AB"], "a_both": g_a["AB"],
                     "n_A": g_n["A"], "a_A": g_a["A"], "n_B": g_n["B"], "a_B": g_a["B"],
                     "ror_AB": e.loc["AB", "ror"], "lcl_AB": e.loc["AB", "lcl"], "ucl_AB": e.loc["AB", "ucl"],
                     "ror_A": e.loc["A", "ror"], "lcl_A": e.loc["A", "lcl"], "ucl_A": e.loc["A", "ucl"],
                     "ror_B": e.loc["B", "ror"], "lcl_B": e.loc["B", "lcl"], "ucl_B": e.loc["B", "ucl"],
                     "h2h_lcl_A": float(h_a[1]), "h2h_lcl_B": float(h_b[1]), "p": p, "p_A": p_a, "p_B": p_b, "_E": E})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    allp = np.concatenate([out["p"], out["p_A"], out["p_B"]])           # one BH family for all partner tests
    allq = multipletests(allp, method="fdr_bh")[1]
    m = len(out)
    out["q"], out["q_A"], out["q_B"] = allq[:m], allq[m:2 * m], allq[2 * m:]
    v, t = [], []
    for r in out.to_dict("records"):
        ver = _pair_verdict(r, min_events)
        v.append(ver)
        t.append(_pair_text(r, drug, reaction, ver))
    out["verdict"], out["justification"] = v, t
    out["_o"] = out["verdict"].map({k: i for i, k in enumerate(PAIR_VERDICTS)})
    return out.sort_values(["_o", "ror_AB"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)


def explain_drug_reaction(X, Y, drugs, rxs, drug, reaction, top_k=15, min_pair=10, min_events=3, max_steps=3):
    """Greedy, explainable partner analysis.
    1) analyse every frequent partner;  2) take the partner with the strongest evidence - either a pair effect
    (drug + partner much higher than either alone) or a partner that is a strong cause on its own;
    3) set aside reports containing it and re-test the rest, so apparent effects that were only riding on
    that partner disappear;  repeat (max_steps).   Returns (table, main_partners).
    The table's `main_kind` column is 'pair' / 'partner' for the chosen partners, '' otherwise."""
    drugs = list(drugs)
    cur = pair_attribution(X, Y, drugs, rxs, drug, reaction, top_k, min_pair, min_events)
    if cur.empty:
        return cur, []
    base_flagged = set(cur.loc[cur["verdict"].isin(["Pair effect", "Partner alone"]), "partner"])
    final = {r["partner"]: dict(r, note="") for r in cur.to_dict("records")}
    kinds, main, excl = {}, [], []

    def score(r):
        if r["verdict"] == "Pair effect":
            return min(r["h2h_lcl_A"], r["h2h_lcl_B"])
        return r["lcl_B"] if r["verdict"] == "Partner alone" else -np.inf

    for _ in range(max_steps):
        cand = [r for r in cur.to_dict("records") if r["partner"] not in main and score(r) > -np.inf]
        if not cand:
            break
        best = max(cand, key=score)
        main.append(best["partner"])
        kinds[best["partner"]] = "pair" if best["verdict"] == "Pair effect" else "partner"
        excl.append(drugs.index(best["partner"]))
        cur = pair_attribution(X, Y, drugs, rxs, drug, reaction, top_k, min_pair, min_events, exclude=tuple(excl))
        seen = set(cur["partner"]) if len(cur) else set()
        why = f"re-tested without reports containing {', '.join(disp(m) for m in main)}"
        for r in (cur.to_dict("records") if len(cur) else []):
            if r["partner"] in main:
                continue
            n = why + (" - the earlier signal is explained by that partner"
                       if r["partner"] in base_flagged and r["verdict"] not in ("Pair effect", "Partner alone") else "")
            final[r["partner"]] = dict(r, note=n)
        for part, r in list(final.items()):          # partners with too few reports left after setting aside
            if part not in main and part not in seen:
                final[part] = dict(r, verdict="No signal", note=why + " - too few reports left; "
                                   + ("the earlier signal is explained by that partner" if part in base_flagged else "no signal"))
        if cur.empty:
            break
    out = pd.DataFrame(list(final.values()))
    out["main_kind"] = out["partner"].map(kinds).fillna("")
    out["_o"] = out["verdict"].map({k: i for i, k in enumerate(PAIR_VERDICTS)})
    out = out.sort_values(["_o", "ror_AB"], ascending=[True, False]).drop(columns="_o")
    return out.reset_index(drop=True), main


def _pair_verdict(r, min_events):
    sig_ab = r["a_both"] >= min_events and r["lcl_AB"] > 1 and r["q"] < 0.05
    sig_a = r["a_A"] >= min_events and r["lcl_A"] > 1 and r["q_A"] < 0.05
    sig_b = r["a_B"] >= min_events and r["lcl_B"] > 1 and r["q_B"] < 0.05
    beats_a, beats_b = r["h2h_lcl_A"] > 1, r["h2h_lcl_B"] > 1
    if sig_ab:
        if beats_a and beats_b:
            return "Pair effect"
        a_expl, b_expl = sig_a and not beats_a, sig_b and not beats_b
        if a_expl and b_expl:
            return "Both alone"
        if a_expl:
            return "Drug alone"
        if b_expl:
            return "Partner alone"
        return "Cannot separate"
    if sig_a and sig_b:
        return "Both alone"
    if sig_a:
        return "Drug alone"
    if sig_b:
        return "Partner alone"
    return "No signal"


def _pair_text(r, drug, reaction, ver):
    A, B, rx = disp(drug), disp(r["partner"]), disp_rx(reaction).lower()
    g = lambda k: f"{_fx(r['ror_' + k])}x ({_fx(r['lcl_' + k])}-{_fx(r['ucl_' + k])}, {int(r['n_' + ('both' if k == 'AB' else k)]):,} reports)"
    s = (f"{int(r['n_both']):,} reports list both {A} and {B}; {int(r['a_both'])} mention {rx}. Compared with reports "
         f"listing neither: both together {g('AB')}; {A} without {B} {g('A')}; {B} without {A} {g('B')}. ")
    if ver == "Pair effect":
        s += (f"Both together are significantly higher than {A} alone and than {B} alone (head-to-head lower bounds "
              f"{_fx(r['h2h_lcl_A'])} and {_fx(r['h2h_lcl_B'])}, both > 1): the reaction goes with the combination.")
    elif ver == "Drug alone":
        s += f"{A} already shows the signal without {B}, and adding {B} is not significantly higher - {B} adds nothing detectable."
    elif ver == "Partner alone":
        s += f"{B} already shows the signal without {A}, and adding {A} is not significantly higher - the reaction belongs to {B}, not {A}."
    elif ver == "Both alone":
        s += f"Each drug shows the signal on its own and the pair is not significantly higher than either."
    elif ver == "Cannot separate":
        s += "The pair has a signal, but there are too few reports of either drug without the other to say which is responsible."
    else:
        s += "No clear signal for this pairing."
    return s


def adjusted_or(X, Y, drugs, rxs, drug, reaction, top_k=40, min_events=30):
    """Odds ratio for `drug` after holding the top_k most common OTHER drugs fixed (multivariable
    logistic regression).  Returns dict or None when it cannot be fitted reliably."""
    import statsmodels.api as sm
    drugs = list(drugs)
    di, ri = drugs.index(drug), list(rxs).index(reaction)
    y = (Y[:, ri].toarray().ravel() > 0).astype(float)
    if y.sum() < min_events:
        return None
    freq = np.asarray(X.sum(0)).ravel()
    order = [i for i in np.argsort(-freq) if i != di and not overlapping(drug, drugs[i])][:top_k]
    xd = (X[:, di].toarray().ravel() > 0).astype(float)
    design = np.column_stack([np.ones(len(y)), xd, (X[:, order].toarray() > 0).astype(float)])
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.Logit(y, design).fit(disp=0, maxiter=200)
        b, se = float(fit.params[1]), float(fit.bse[1])
        if not fit.mle_retvals.get("converged", True) or not np.isfinite(se) or se > 5:
            return None
        return {"or": float(np.exp(b)), "lcl": float(np.exp(b - Z95 * se)), "ucl": float(np.exp(b + Z95 * se)),
                "n_adjust": len(order), "events": int(y.sum())}
    except Exception:
        return None


# ----------------------------------------------------------------------------------------
# 5. ORCHESTRATION
# ----------------------------------------------------------------------------------------
def run_mining(reports, min_triplet=None, min_events=3, max_drugs=300, max_triplets=20000,
               min_drug_reports=10, log=print):
    from mlxtend.frequent_patterns import fpgrowth

    N = len(reports)
    mlb_d, mlb_r = MultiLabelBinarizer(sparse_output=True), MultiLabelBinarizer(sparse_output=True)
    X = mlb_d.fit_transform(reports["drugs"]).astype(np.int32).tocsc()
    Y = mlb_r.fit_transform(reports["reactions"]).astype(np.int32).tocsc()
    drugs, rxs = mlb_d.classes_, mlb_r.classes_
    log(f"{N:,} reports | {len(drugs):,} distinct drugs | {len(rxs):,} distinct reactions")

    stats = drug_reaction_stats(X, drugs, Y, rxs, min_events, min_drug_reports)
    log(f"Single-drug statistics: {len(stats):,} drug-reaction pairs "
        f"({int(stats['signal'].sum()):,} with ROR lower CI > 1)")

    if min_triplet is None:
        min_triplet = max(10, int(0.001 * N))
    n_drug = np.asarray(X.sum(0)).ravel()
    keep = np.flatnonzero(n_drug >= min_triplet)
    keep = keep[np.argsort(-n_drug[keep])][:max_drugs]
    triplets = []
    if len(keep) >= 3:
        sub = X[:, keep].astype(bool)
        dfb = pd.DataFrame.sparse.from_spmatrix(sub, columns=list(range(len(keep))))
        fi = fpgrowth(dfb, min_support=min_triplet / N, use_colnames=True, max_len=3)
        fi = fi[fi["itemsets"].apply(len) == 3].sort_values("support", ascending=False)
        for items in fi["itemsets"]:
            idx = sorted(int(keep[x]) for x in items)
            if any(overlapping(drugs[p], drugs[q]) for p in idx for q in idx if p < q):
                continue                       # same ingredient listed twice, not three drugs
            triplets.append(tuple(idx))
            if len(triplets) >= max_triplets:
                break
    log(f"FP-Growth: {len(triplets):,} candidate triplets (>= {min_triplet} reports each, "
        f"{len(keep)} most frequent drugs)")

    res = evaluate_triplets(X, Y, drugs, rxs, triplets, min_events, log) if triplets else pd.DataFrame()
    if len(res):
        res = classify_triplets(res, min_events)

    mon = reports[["month", "drugs"]].explode("drugs").rename(columns={"drugs": "drug"})
    drug_month = mon[mon["month"] != ""].groupby(["drug", "month"]).size().reset_index(name="reports")
    dates = reports["date"][reports["date"].str.len() >= 8]
    meta = {"n_reports": N, "n_drugs": int(len(drugs)), "n_reactions": int(len(rxs)),
            "date_min": dates.min() if len(dates) else "", "date_max": dates.max() if len(dates) else "",
            "min_triplet_reports": int(min_triplet), "min_events": int(min_events),
            "n_candidate_triplets": len(triplets), "n_tested": int(len(res))}
    if len(res):
        meta["verdict_counts"] = res["verdict"].value_counts().to_dict()
        meta["median_ror_all_tested"] = float(res["ror"].median())
        meta["share_ror_over_10"] = float((res["ror"] > 10).mean())
        meta["share_ror_over_100"] = float((res["ror"] > 100).mean())
    return {"triplets": res, "drug_reaction": stats, "drug_month": drug_month, "meta": meta}


def save_results(result, reports, excluded, info, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports.to_parquet(out / "reports_clean.parquet", index=False)
    result["drug_reaction"].to_parquet(out / "drug_reaction_stats.parquet", index=False)
    result["drug_month"].to_parquet(out / "drug_month.parquet", index=False)
    excluded.to_parquet(out / "excluded_terms.parquet", index=False)
    res = result["triplets"]
    if len(res):
        res.to_parquet(out / "triplet_results.parquet", index=False)
        cols = ["drug_a", "drug_b", "drug_c", "reaction", "verdict", "driver_drugs", "n_reports", "n_events",
                "ror", "ror_lcl", "ror_ucl", "q_ror", "beta3", "q_beta3", "flags", "justification"]
        (res[res["verdict"] != "No signal"].sort_values("ror_lcl", ascending=False)[cols]
         .to_csv(out / "triplet_signals.csv", index=False))
    (out / "meta.json").write_text(json.dumps({**result["meta"], **info}, indent=2, default=str))
