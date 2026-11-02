"""Combine pieces of evidence into one posterior probability, without double counting.

Model. For a hypothesis H with prior odds o0:   posterior log-odds = logit(prior) + sum_g L_g
where each *evidence group* g (one label, one paper, one database) contributes L_g. Naive Bayes would add every
sentence separately; ten sentences from one label are NOT ten independent confirmations. Within a group of n items
of the same sign with mean log-likelihood-ratio m we therefore use an effective count

        n_eff = n / (1 + rho (n - 1))          rho = intra-group correlation (0: independent, 1: one item)
        L_g   = m * n_eff

Supporting and contradicting items are bucketed separately, so a paper that both supports and qualifies a claim
counts both ways.

The log-likelihood-ratios in DEFAULT_LOG_LR are *judgement priors*. They are deliberately modest, and
`validate.calibrate()` re-estimates them from the hand-checked benchmark by regularised logistic regression;
until that has been run, results are labelled "uncalibrated". The credible interval comes from Monte-Carlo over
the uncertainty in those weights (a shared log-normal error per evidence kind, plus prior uncertainty), so it
widens when a hypothesis rests on a kind of evidence we trust little.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_LOG_LR = {
    "label_class": 1.4,       # structured FDA class field ("... CYP3A4 Inhibitors [MoA]")         LR ~ 4
    "label_role": 0.9,        # explicit sentence in the drug's own label                         LR ~ 2.5
    "label_context": 0.4,     # indirect wording only                                               LR ~ 1.5
    "label_ddi": 1.8,         # the partner is named in this label's drug-interaction text          LR ~ 6
    "database": 0.7,          # curated target / mechanism record                                   LR ~ 2
    "literature": 1.0,        # abstract sentence naming both drugs and the mechanism                LR ~ 2.7
    "faers": 1.1,             # FAERS: effect only with the combination                             LR ~ 3
    "network": 0.4,           # protein-network proximity only                                      LR ~ 1.5
    "contradiction": -1.4,    # explicit negation naming both drugs                                 LR ~ 0.25
    "faers_against": -0.9,    # FAERS: one drug alone explains the signal                           LR ~ 0.4
}
PRIORS = {
    "PK-enzyme": 0.25, "PK-transporter": 0.15, "PD-shared-target": 0.15, "PD-effect-axis": 0.20,
    "PD-pathway": 0.05, "indirect-network": 0.03, "documented": 0.50, "llm-proposed": 0.05,
}


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class Item:
    kind: str
    group: str
    weight: float = 1.0            # scales the table value, e.g. 0.6 for a pair-level (non-specific) contradiction


def _contributions(items, table, rho, kind_scale=None):
    buckets: dict[tuple, list[float]] = {}
    for it in items:
        base = table[it.kind] * it.weight
        if kind_scale:
            base *= kind_scale.get(it.kind, 1.0)
        buckets.setdefault((it.group, base >= 0), []).append(base)
    total_pos = total_neg = 0.0
    for (_, positive), vals in buckets.items():
        n = len(vals)
        n_eff = n / (1 + rho * (n - 1))
        v = float(np.mean(vals)) * n_eff
        if positive:
            total_pos += v
        else:
            total_neg += v
    return total_pos, total_neg, len({g for g, _ in buckets})


def combine(items, prior: float, rho: float = 0.7, table: dict | None = None) -> dict:
    pos, neg, n_groups = _contributions(items, table or DEFAULT_LOG_LR, rho)
    lo = float(logit(prior) + pos + neg)
    return {"log_odds": lo, "posterior": float(expit(lo)), "support": pos, "against": neg, "n_groups": n_groups}


def posterior_interval(items, prior: float, rho: float = 0.7, table: dict | None = None, sigma: float = 0.5,
                       n: int = 2000, seed: int = 0, q=(5, 95)) -> tuple[float, float]:
    """Monte-Carlo interval: each evidence kind's weight is multiplied by exp(N(0, sigma^2)) (shared across the
    items of that kind), and the prior is perturbed on the logit scale by N(0, sigma^2)."""
    table = table or DEFAULT_LOG_LR
    rng = np.random.default_rng(seed)
    kinds = sorted({i.kind for i in items})
    out = np.empty(n)
    for s in range(n):
        scale = {k: float(np.exp(rng.normal(0, sigma))) for k in kinds}
        pos, neg, _ = _contributions(items, table, rho, scale)
        out[s] = expit(logit(prior) + rng.normal(0, sigma) + pos + neg)
    lo, hi = np.percentile(out, q)
    return float(lo), float(hi)


def noisy_or(probs, groups=None, rho: float = 0.0) -> float:
    """P(at least one cause is real). Independent causes: 1 - prod(1 - p). If `groups` is given, causes that share
    a group are partly redundant: their joint 'miss' probability is raised to n_eff/n (rho as above)."""
    probs = np.asarray(probs, float)
    if probs.size == 0:
        return 0.0
    if groups is None:
        return float(1 - np.prod(1 - probs))
    miss = 1.0
    for g in set(groups):
        idx = [i for i, x in enumerate(groups) if x == g]
        n = len(idx)
        n_eff = n / (1 + rho * (n - 1))
        miss *= float(np.prod(1 - probs[idx])) ** (n_eff / n)
    return float(1 - miss)


def fit_log_lr(X: np.ndarray, y: np.ndarray, kinds: list[str], C: float = 1.0) -> dict:
    """Re-estimate the evidence weights from labelled hypotheses.
    X[i, j] = number of (group-discounted) items of kind j for hypothesis i; y[i] = 1 if the hypothesis is true.
    Returns {kind: weight, '_intercept': logit-prior-shift}. L2-regularised, so a kind seen rarely stays near 0."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, max_iter=1000).fit(X, y)
    table = {k: float(c) for k, c in zip(kinds, clf.coef_[0])}
    table["_intercept"] = float(clf.intercept_[0])
    return table
