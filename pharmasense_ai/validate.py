"""VALIDATE: does the engine recover mechanisms we already know, and does it stay quiet where the mechanism is not established?

Benchmark: data/benchmark/known_interactions.csv - HAND-CURATED from standard pharmacology (about 20 positives, 9
mechanism-specific negatives). Read these limits before quoting any number:
  * It tests RECOVERY OF DOCUMENTED MECHANISMS. The positives are in drug labels, and the engine reads drug labels, so a
    good score is a sanity check, not evidence that Discover works on unknown interactions.
  * Negatives mean "this mechanism is not established for this pair", not "the pair is safe".
  * With ~30 rows every interval is wide. Calibration fits ~10 parameters on ~30 labelled hypotheses; it is shipped with a
    cross-validated check and an explicit overfitting warning, and the default weights remain the fallback.
  * The list was written by one person (and an AI assistant) and has not been independently audited. Audit it.

Metrics: recall on positives (matching hypothesis reaches verdict 'supported'), false-positive rate on negatives (a matching
hypothesis reaches 'supported'), AUROC of the posterior (absent hypothesis scores 0), each with an interval.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .agents.workflow import CALIBRATION_FILE, build_graph
from .config import ROOT
from .mathx import evidence as em
from .mathx import uncertainty as un
from .models import EvidenceStore

BENCH = ROOT / "data" / "benchmark" / "known_interactions.csv"
GOLD_ROLES = ROOT / "data" / "benchmark" / "gold_roles.csv"
RHO = 0.7


def load_benchmark(path=BENCH) -> pd.DataFrame:
    d = pd.read_csv(path).fillna("")
    for c in ("drug_a", "drug_b", "expected_type", "expected_term", "perpetrator"):
        d[c] = d[c].astype(str).str.strip()
    return d


def match(report: dict, row) -> dict | None:
    """The hypothesis that matches the benchmark row's mechanism (same type + term; for PK also the same perpetrator)."""
    for h in report["hypotheses"]:
        if h["mechanism_type"] != row.expected_type or h.get("term", "").lower() != row.expected_term.lower():
            continue
        if row.expected_type.startswith("PK") and row.perpetrator and h["perpetrator"].lower() != row.perpetrator.lower():
            continue
        return h
    return None


def run_benchmark(svc, rows: pd.DataFrame, table=None, log=print) -> tuple[pd.DataFrame, list[dict]]:
    """Explain every row WITHOUT FAERS context (the benchmark is about mechanism knowledge, not about signals)."""
    store, cache, out, reports = EvidenceStore(), {}, [], []
    for r in rows.itertuples():
        app, _ = build_graph(svc, store=store, table=table, cache=cache)
        res = app.invoke({"query": {"drugs": [r.drug_a, r.drug_b], "reaction": r.reaction or None, "faers": None}, "loops": 0, "warnings": [], "trace": []})
        rep = res["report"]
        reports.append(rep)
        h = match(rep, r)
        out.append({"drug_a": r.drug_a, "drug_b": r.drug_b, "expected": f"{r.expected_type}:{r.expected_term}", "label": int(r.label),
                    "proposed": h is not None, "verdict": h["verdict"] if h else "not proposed", "posterior": h["posterior"] if h else 0.0,
                    "supported": bool(h and h["verdict"] == "supported"), "n_other_supported": sum(1 for x in rep["hypotheses"] if x is not h and x["verdict"] == "supported")})
        log(f"  {r.drug_a}+{r.drug_b} [{out[-1]['expected']}] label={r.label} -> {out[-1]['verdict']} ({out[-1]['posterior']:.2f})")
    return pd.DataFrame(out), reports


def summarise(res: pd.DataFrame, seed: int = 0) -> dict:
    pos, neg = res[res.label == 1], res[res.label == 0]
    rec = un.beta_interval(int(pos.supported.sum()), len(pos))
    fpr = un.beta_interval(int(neg.supported.sum()), len(neg)) if len(neg) else (float("nan"),) * 3
    out = {"n_positive": len(pos), "n_negative": len(neg),
           "recall": {"value": int(pos.supported.sum()) / max(len(pos), 1), "ci95": [rec[1], rec[2]], "k": int(pos.supported.sum())},
           "proposed_rate_positives": float(pos.proposed.mean()) if len(pos) else float("nan"),
           "false_positive_rate": {"value": int(neg.supported.sum()) / max(len(neg), 1), "ci95": [fpr[1], fpr[2]], "k": int(neg.supported.sum())},
           "missed": pos[~pos.supported][["drug_a", "drug_b", "expected", "verdict"]].to_dict("records"),
           "false_alarms": neg[neg.supported][["drug_a", "drug_b", "expected", "verdict"]].to_dict("records")}
    if len(pos) and len(neg):
        y, s = res.label.values, res.posterior.values
        out["auroc"] = float(roc_auc_score(y, s))
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(1000):
            i = rng.integers(0, len(res), len(res))
            if 0 < y[i].sum() < len(i):
                boots.append(roc_auc_score(y[i], s[i]))
        out["auroc_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None
        out["brier"] = un.brier(s, y)
    return out


# --------------------------------------------------------------------------- label-parser check
def parser_recall(svc, gold_path=GOLD_ROLES, log=print) -> dict:
    """Recall of the enzyme-role miner against a small hand-written gold list (drug, enzyme, role). Recall only: the gold
    list is not exhaustive, so an extra extracted role is NOT counted as an error and precision is not estimated here."""
    from .profile import build_profile
    gold = pd.read_csv(gold_path)
    store, got, miss = EvidenceStore(), 0, []
    for d, g in gold.groupby("drug"):
        p = build_profile(d, svc, store)
        have = {(r.enzyme, r.role) for r in p.roles}
        for r in g.itertuples():
            if (r.enzyme, r.role) in have:
                got += 1
            else:
                miss.append(f"{d}: {r.role} of {r.enzyme}")
    n = len(gold)
    b = un.beta_interval(got, n)
    return {"n_gold": n, "recall": got / max(n, 1), "ci95": [b[1], b[2]], "missed": miss}


# --------------------------------------------------------------------------- calibration
def features(report: dict, h: dict, kinds: list[str]) -> np.ndarray:
    """Per-kind effective item counts, discounted exactly as the scorer does (n_eff = n / (1 + rho (n-1)) within a group)."""
    ev, x = report["evidence"], dict.fromkeys(kinds, 0.0)
    groups: dict[tuple, int] = {}
    for e in h["supporting"] + h["contradicting"]:
        if e in ev:
            k = {"label_text": "label_context"}.get(ev[e]["kind"], ev[e]["kind"])
            if k in x:
                groups[(k, ev[e].get("group") or ev[e].get("source"))] = groups.get((k, ev[e].get("group") or ev[e].get("source")), 0) + 1
    for (k, _), n in groups.items():
        x[k] += n / (1 + RHO * (n - 1))
    return np.array([x[k] for k in kinds])


def calibrate(rows: pd.DataFrame, reports: list[dict], out_path=CALIBRATION_FILE, alpha: float = 0.1, C: float = 0.5) -> dict:
    """Fit evidence weights by L2-regularised logistic regression, check by leave-one-out against the default weights, and
    save only the numbers the engine needs. Only hypotheses with a benchmark label are used (others are unlabelled)."""
    kinds = [k for k in em.DEFAULT_LOG_LR if k != "faers" and k != "faers_against"]          # the benchmark has no FAERS context
    X, y, prior = [], [], []
    for r, rep in zip(rows.itertuples(), reports):
        h = match(rep, r)
        if h is None:
            continue
        X.append(features(rep, h, kinds))
        y.append(int(r.label))
        prior.append(h["prior"])
    X, y, prior = np.array(X), np.array(y), np.array(prior)
    if len(y) < 12 or y.sum() < 4 or (len(y) - y.sum()) < 4:
        return {"status": "skipped", "reason": f"need >=12 labelled hypotheses with >=4 of each class; have {len(y)} ({int(y.sum())} positive)"}
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut

    lp = em.logit(prior)

    def engine_p(Xs, w, lps):               # exactly what the engine computes: prior log-odds + weighted evidence (no intercept)
        return em.expit(lps + Xs @ w)

    p_loo = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        if len(set(y[tr])) < 2:
            p_loo[te] = y[tr].mean()
            continue
        w = LogisticRegression(C=C, max_iter=1000).fit(X[tr], y[tr]).coef_[0]
        p_loo[te] = engine_p(X[te], w, lp[te])
    w_default = np.array([em.DEFAULT_LOG_LR[k] for k in kinds])
    p_default = engine_p(X, w_default, lp)
    table = em.fit_log_lr(X, y, kinds, C=C)
    # Sign constraint: the scorer decides support vs. contradiction from the SIGN of a weight, so a fit that flips a kind's
    # direction (possible with ~30 rows and correlated kinds) would silently turn evidence for into evidence against.
    for k in kinds:
        table[k] = min(table[k], 0.0) if k == "contradiction" else max(table[k], 0.0)
    w_fit = np.array([table[k] for k in kinds])
    p_fit = engine_p(X, w_fit, lp)
    pos_scores = p_fit[y == 1]
    result = {"status": "fitted", "n": int(len(y)), "n_positive": int(y.sum()), "C": C, "kinds": kinds,
              "log_lr": {k: round(table[k], 3) for k in kinds}, "intercept_unused": round(table["_intercept"], 3),
              "leave_one_out": {"auroc": float(roc_auc_score(y, p_loo)), "brier": un.brier(p_loo, y)},
              "in_sample": {"auroc": float(roc_auc_score(y, p_fit)), "brier": un.brier(p_fit, y)},
              "default_weights": {"auroc": float(roc_auc_score(y, p_default)), "brier": un.brier(p_default, y)},
              "conformal": un.SplitConformal(alpha).fit(pos_scores).to_dict(),
              "warning": (f"Fitted on only {len(y)} labelled hypotheses with {len(kinds)} parameters. The label evidence that defines "
                          "the benchmark positives is also the engine's main input, so weights for those kinds are inflated relative to "
                          "unseen interactions. Use the leave-one-out figures, not the in-sample fit; the fit is adopted only if its "
                          "leave-one-out Brier score beats the hand-set defaults.")}
    # Only adopt the fit if it generalises better than the hand-set defaults under leave-one-out; otherwise record the check only.
    result["adopted"] = bool(result["leave_one_out"]["brier"] < result["default_weights"]["brier"])
    if result["adopted"]:
        # stored in the same shape load_calibration() expects: weights per kind (the intercept absorbs the prior shift and is not used)
        out = {"log_lr": {**em.DEFAULT_LOG_LR, **result["log_lr"]}, "conformal": result["conformal"], "meta": {k: v for k, v in result.items() if k not in ("conformal",)}}
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=1))
    return result
