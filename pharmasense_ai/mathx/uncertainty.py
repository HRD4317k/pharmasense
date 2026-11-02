"""Calibration and coverage guarantees.

 * `beta_interval` / `bootstrap_ci`  honest intervals for proportions and statistics
 * `brier`, `ece`, `reliability`      does a stated 70% come true ~70% of the time?
 * `SplitConformal`                   accept a hypothesis only if it is not *unusually weak* compared with
   calibration hypotheses known to be true. Guarantee (exchangeability): a new TRUE hypothesis is rejected with
   probability <= alpha. It guards against missing real mechanisms, not against accepting false ones - so it is
   reported next to the posterior, not instead of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


def beta_interval(k: int, n: int, alpha: float = 0.05, prior=(0.5, 0.5)) -> tuple[float, float, float]:
    """Posterior mean and equal-tailed credible interval for a proportion (Jeffreys prior by default)."""
    a, b = prior[0] + k, prior[1] + n - k
    return float(a / (a + b)), float(beta_dist.ppf(alpha / 2, a, b)), float(beta_dist.ppf(1 - alpha / 2, a, b))


def bootstrap_ci(x, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    vals = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    return float(stat(x)), float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def reliability(p, y, bins: int = 10) -> pd.DataFrame:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()),
                         "mean_predicted": float(p[m].mean()), "fraction_true": float(y[m].mean())})
    return pd.DataFrame(rows)


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error: bin-size-weighted gap between predicted and observed frequency."""
    r = reliability(p, y, bins)
    if r.empty:
        return float("nan")
    return float((r["n"] * (r["mean_predicted"] - r["fraction_true"]).abs()).sum() / r["n"].sum())


class SplitConformal:
    """Fit on the scores of hypotheses known to be TRUE (calibration split). A candidate with score s gets the
    conformal p-value  p = (1 + #{calibration scores <= s}) / (n + 1);  it is accepted if p > alpha."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.scores = np.array([])

    def fit(self, true_scores):
        self.scores = np.sort(np.asarray(true_scores, float))
        return self

    @property
    def n(self) -> int:
        return len(self.scores)

    def pvalue(self, s: float) -> float:
        return float((1 + np.searchsorted(self.scores, s, side="right")) / (self.n + 1))

    def accepts(self, s: float) -> bool:
        return self.pvalue(s) > self.alpha

    def threshold(self) -> float:
        """Smallest score that is accepted (nan if the calibration set is too small for this alpha)."""
        k = int(np.floor(self.alpha * (self.n + 1)))          # scores at rank <= k are rejected
        if k < 1:
            return float(-np.inf)
        return float(self.scores[min(k, self.n) - 1])

    def to_dict(self):
        return {"alpha": self.alpha, "scores": self.scores.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["alpha"]).fit(d["scores"])
