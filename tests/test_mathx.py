"""Tests for the mathematics. Run:  python -m pytest tests -q"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pharmasense_ai.mathx import evidence as ev
from pharmasense_ai.mathx import graphmath as gm
from pharmasense_ai.mathx import interaction as ia
from pharmasense_ai.mathx import uncertainty as un

REF_ODDS = 100 / 9900


def make_groups(ors: dict, n=400_000):
    """Counts whose odds ratios vs the reference are exactly `ors` (large n -> sampling noise negligible)."""
    g = {"REF": (round(n * 0.01), n)}
    for k, o in ors.items():
        odds = REF_ODDS * o
        p = odds / (1 + odds)
        g[k] = (round(n * p), n)
    return g


# ------------------------------------------------------------------ interaction
def test_reri_additive_effect_is_zero():
    out = ia.reri_pair(make_groups({"A": 2, "B": 3, "AB": 4}))          # (2-1)+(3-1)=3 excess == 4-1 -> additive
    assert abs(out["reri"]["estimate"]) < 0.1
    assert out["reri"]["lcl"] < 0 < out["reri"]["ucl"]
    assert not out["more_than_additive"]


def test_reri_super_additive():
    out = ia.reri_pair(make_groups({"A": 2, "B": 3, "AB": 10}))
    assert out["reri"]["estimate"] == pytest.approx(6.0, rel=0.03)
    assert out["more_than_additive"] and out["p_not_super_additive"] < 0.01
    assert out["synergy_index"]["estimate"] == pytest.approx(9 / 3, rel=0.03)


def test_super_multiplicative_can_still_be_sub_additive():
    # OR_A=OR_B=1.5, OR_AB=3: product 2.25 < 3 (super-multiplicative) but 3-1=2 > 0.5+0.5=1 -> also super-additive.
    # OR_A=OR_B=5, OR_AB=26: product 25 < 26 (barely super-multiplicative) and 25 > 8 -> super-additive. Choose a real counter-case:
    # OR_A=OR_B=0.5 (protective singles), OR_AB=0.3: product 0.25 < 0.3 -> sub-multiplicative; RERI=0.3-0.5-0.5+1=0.3>0.
    out = ia.reri_pair(make_groups({"A": 0.5, "B": 0.5, "AB": 0.3}))
    assert out["multiplicative_ratio"]["estimate"] == pytest.approx(1.2, rel=0.05)
    assert out["reri"]["estimate"] == pytest.approx(0.3, abs=0.03)


def test_triple_contrast_values():
    g = make_groups({"A": 2, "B": 2, "C": 2, "AB": 5, "AC": 5, "BC": 5, "ABC": 8})
    out = ia.triple_contrast(g)
    assert out["three_way_component"]["estimate"] == pytest.approx(8 - 15 + 6 - 1, abs=0.15)
    assert out["excess_over_singles"]["estimate"] == pytest.approx(8 - 1 - 3, abs=0.15)


def test_zero_cell_is_handled():
    out = ia.reri_pair({"REF": (10, 1000), "A": (0, 50), "B": (5, 60), "AB": (3, 40)}, n_boot=500)
    assert np.isfinite(out["reri"]["estimate"])


# ------------------------------------------------------------------ evidence
def test_prior_only_returns_prior():
    assert ev.combine([], 0.25)["posterior"] == pytest.approx(0.25)


def test_support_raises_and_contradiction_lowers():
    base = ev.combine([], 0.25)["posterior"]
    up = ev.combine([ev.Item("label_class", "label:a")], 0.25)["posterior"]
    down = ev.combine([ev.Item("contradiction", "pmid:1")], 0.25)["posterior"]
    assert down < base < up


def test_same_source_counts_less_than_independent_sources():
    same = ev.combine([ev.Item("label_role", "label:a")] * 5, 0.25)["log_odds"]
    indep = ev.combine([ev.Item("label_role", f"label:{i}") for i in range(5)], 0.25)["log_odds"]
    one = ev.combine([ev.Item("label_role", "label:a")], 0.25)["log_odds"]
    assert one < same < indep
    # rho=1 means "all the same document": no gain at all
    assert ev.combine([ev.Item("label_role", "g")] * 5, 0.25, rho=1.0)["log_odds"] == pytest.approx(one)


def test_interval_contains_point_estimate_and_widens_with_sigma():
    items = [ev.Item("literature", "pmid:1"), ev.Item("label_ddi", "label:a")]
    p = ev.combine(items, 0.2)["posterior"]
    lo, hi = ev.posterior_interval(items, 0.2, sigma=0.3, n=1500)
    lo2, hi2 = ev.posterior_interval(items, 0.2, sigma=0.9, n=1500)
    assert lo <= p <= hi
    assert (hi2 - lo2) > (hi - lo)


def test_noisy_or():
    assert ev.noisy_or([0.5, 0.5]) == pytest.approx(0.75)
    assert ev.noisy_or([0.5, 0.5], groups=["a", "a"], rho=1.0) == pytest.approx(0.5)      # fully redundant
    assert ev.noisy_or([0.5, 0.5], groups=["a", "b"], rho=1.0) == pytest.approx(0.75)     # independent groups
    assert ev.noisy_or([]) == 0.0


def test_fit_log_lr_recovers_signs():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 3, size=(600, 2)).astype(float)         # kind 0 helps, kind 1 hurts
    p = ev.expit(-0.5 + 1.2 * X[:, 0] - 1.0 * X[:, 1])
    y = (rng.random(600) < p).astype(int)
    t = ev.fit_log_lr(X, y, ["good", "bad"])
    assert t["good"] > 0.6 and t["bad"] < -0.5


# ------------------------------------------------------------------ uncertainty
def test_beta_interval():
    m, lo, hi = un.beta_interval(0, 10)
    assert lo < 0.01 and hi < 0.3
    m, lo, hi = un.beta_interval(50, 100)
    assert lo < 0.5 < hi and (hi - lo) < 0.25


def test_calibration_metrics():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)                     # perfectly calibrated
    assert un.ece(p, y) < 0.02
    over = np.clip(p * 0.5 + 0.5, 0, 1)                         # overconfident
    assert un.ece(over, y) > 0.15
    assert un.brier(p, y) < un.brier(over, y)


def test_conformal_coverage_of_true_hypotheses():
    rng = np.random.default_rng(7)
    cal = rng.beta(6, 2, 400)                                   # scores of hypotheses known to be true
    test = rng.beta(6, 2, 20000)
    for alpha in (0.05, 0.1, 0.2):
        cf = un.SplitConformal(alpha).fit(cal)
        covered = np.mean([cf.accepts(s) for s in test])
        assert covered >= 1 - alpha - 0.02
    cf = un.SplitConformal(0.1).fit(cal)
    assert cf.accepts(0.99) and not cf.accepts(0.01)
    assert un.SplitConformal.from_dict(cf.to_dict()).pvalue(0.5) == cf.pvalue(0.5)


# ------------------------------------------------------------------ graph math
def _planted(seed=0, n=120, k=4, p_in=0.25, p_out=0.01):
    return nx.planted_partition_graph(k, n // k, p_in, p_out, seed=seed)


def test_ppr_matches_networkx():
    G = _planted()
    A, nodes = gm.adjacency(G)
    r = gm.personalized_pagerank(A, {3: 1.0}, alpha=0.15)
    ref = nx.pagerank(G, alpha=0.85, personalization={3: 1.0}, tol=1e-12, max_iter=1000)
    assert r.sum() == pytest.approx(1.0)
    assert np.allclose(r, [ref[n] for n in nodes], atol=1e-6)


@pytest.mark.parametrize("method", ["ppr", "aa", "spectral"])
def test_link_prediction_beats_chance_on_structured_graph(method):
    # "clearly above chance" (0.5). Adamic-Adar only scores pairs that share a neighbour, so on this sparse graph
    # (mean degree ~7, 20% of edges hidden). Measured here: PPR 0.82, Adamic-Adar 0.74, spectral 0.69.
    assert gm.holdout_auc(_planted(), method, frac=0.2, seed=1) > 0.65


@pytest.mark.parametrize("method", ["ppr", "aa", "spectral"])
def test_link_prediction_is_near_chance_on_random_graph(method):
    G = nx.gnp_random_graph(120, 0.06, seed=3)
    assert abs(gm.holdout_auc(G, method, frac=0.2, seed=1) - 0.5) < 0.15
