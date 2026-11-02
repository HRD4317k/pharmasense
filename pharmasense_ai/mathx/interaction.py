"""Interaction on the ADDITIVE scale, to complement the repo's multiplicative (logistic beta_3) test.

Why both: "does the combination add risk beyond what the separate drugs add?" has two defensible answers.
 * multiplicative (beta_3): odds of A+B equal OR_A x OR_B                    - already in pharmasense_core
 * additive (RERI):         excess odds of A+B equal (OR_A-1) + (OR_B-1)     - public-health relevant: it
   measures how many extra reports the combination produces over the two drugs' separate contributions.
An effect can be super-multiplicative and sub-additive, so reporting only one can mislead.

Definitions (reference group = none of the drugs, OR measured against it):
    RERI = OR_AB - OR_A - OR_B + 1          > 0 : more than additive      (0 : additive)
    S    = (OR_AB - 1) / ((OR_A - 1) + (OR_B - 1))   synergy index        (> 1 : more than additive)
Third-order contrast for A,B,C (inclusion-exclusion; the part due to the *triple*, not to any pair):
    I3 = OR_ABC - OR_AB - OR_AC - OR_BC + OR_A + OR_B + OR_C - 1
Assumptions - stated, not hidden: reporting ORs stand in for risks (rare-outcome approximation, and FAERS is not a
cohort); the groups are independent samples; each group's log-odds is approximately normal (poor when a group has
fewer than ~5 events - intervals are then too narrow, which the caller should flag).
"""
from __future__ import annotations

import numpy as np

PAIR = ["REF", "A", "B", "AB"]
TRIPLE = ["REF", "A", "B", "C", "AB", "AC", "BC", "ABC"]


def _logodds(groups: dict, keys):
    """log-odds and SE per group; add 0.5 to every cell of every group only if some cell is zero (Haldane-Anscombe)."""
    ev = np.array([groups[k][0] for k in keys], float)
    n = np.array([groups[k][1] for k in keys], float)
    non = n - ev
    if (ev == 0).any() or (non == 0).any():
        ev, non = ev + 0.5, non + 0.5
    return np.log(ev) - np.log(non), np.sqrt(1 / ev + 1 / non)


def odds_ratios(groups: dict, keys) -> dict:
    L, se = _logodds(groups, keys)
    return {k: float(np.exp(L[i] - L[0])) for i, k in enumerate(keys)}


def _boot_or(groups, keys, n_boot, rng):
    L, se = _logodds(groups, keys)
    draws = rng.normal(L, se, size=(n_boot, len(keys)))
    return np.exp(draws - draws[:, [0]])                    # columns align with keys; column 0 is the reference


def _summ(point, boot):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"estimate": float(point), "lcl": float(lo), "ucl": float(hi)}


def reri_pair(groups: dict, n_boot: int = 5000, seed: int = 0) -> dict:
    """groups: {"REF":(events,n), "A":..., "B":..., "AB":...}. Returns RERI, synergy index and the multiplicative ratio."""
    o = odds_ratios(groups, PAIR)
    B = _boot_or(groups, PAIR, n_boot, np.random.default_rng(seed))
    A_, B_, AB = B[:, 1], B[:, 2], B[:, 3]
    reri_b = AB - A_ - B_ + 1
    den = (A_ - 1) + (B_ - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_b = np.where(den > 0, (AB - 1) / den, np.nan)
    mult_b = AB / (A_ * B_)
    reri = o["AB"] - o["A"] - o["B"] + 1
    den0 = (o["A"] - 1) + (o["B"] - 1)
    out = {"reri": _summ(reri, reri_b),
           "synergy_index": _summ((o["AB"] - 1) / den0 if den0 > 0 else np.nan, s_b[~np.isnan(s_b)]) if (~np.isnan(s_b)).sum() > 50 else
                            {"estimate": float("nan"), "lcl": float("nan"), "ucl": float("nan")},
           "multiplicative_ratio": _summ(o["AB"] / (o["A"] * o["B"]), mult_b),
           "p_not_super_additive": float(np.mean(reri_b <= 0)),       # one-sided: share of draws with RERI <= 0
           "odds_ratios": o}
    out["more_than_additive"] = out["reri"]["lcl"] > 0
    return out


def triple_contrast(groups: dict, n_boot: int = 5000, seed: int = 0) -> dict:
    """Excess of the all-three group over the singles, and the pure three-way component I3."""
    o = odds_ratios(groups, TRIPLE)
    B = _boot_or(groups, TRIPLE, n_boot, np.random.default_rng(seed))
    col = {k: B[:, i] for i, k in enumerate(TRIPLE)}

    def i3(g):
        return g["ABC"] - g["AB"] - g["AC"] - g["BC"] + g["A"] + g["B"] + g["C"] - 1

    def excess(g):
        return g["ABC"] - 1 - ((g["A"] - 1) + (g["B"] - 1) + (g["C"] - 1))

    e_b, t_b = excess(col), i3(col)
    out = {"excess_over_singles": _summ(excess(o), e_b), "three_way_component": _summ(i3(o), t_b),
           "p_not_super_additive": float(np.mean(e_b <= 0)), "odds_ratios": o}
    out["more_than_additive"] = out["excess_over_singles"]["lcl"] > 0
    return out


def groups_from_row(row, cats) -> dict:
    """Pull {cat: (events, n)} out of a pharmasense_core.evaluate_triplets result row (columns a_<cat>, n_<cat>)."""
    return {c: (int(row[f"a_{c}"]), int(row[f"n_{c}"])) for c in cats}
