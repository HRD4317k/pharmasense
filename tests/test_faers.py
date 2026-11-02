"""Tests for the whole-database counting path, using a synthetic FAERS with a PLANTED interaction (known ground truth)."""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pharmasense_ai import faers_context as fc  # noqa: E402
from pharmasense_ai import faers_live as fl  # noqa: E402


class SyntheticOpenFDA:
    """Answers event_total(search) from an in-memory table of reports, parsing the same clauses the real client sends."""

    def __init__(self, drugs, rx, names):
        self.D, self.R, self.names = drugs, rx, names             # D: (N, k) bool, R: (N,) bool
        self.calls = 0

    @staticmethod
    def drug_clause(name):
        return f'patient.drug.openfda.generic_name:"{name}"'

    def event_total(self, search=None):
        self.calls += 1
        m = np.ones(len(self.R), bool)
        for part in (search or "").split("+AND+"):
            if not part:
                continue
            if part.startswith("patient.reaction"):
                m &= self.R
            else:
                name = re.search(r'"(.+)"', part).group(1)
                m &= self.D[:, self.names.index(name)]
        return int(m.sum())


def make_world(n=300_000, seed=0, kind="combination"):
    rng = np.random.default_rng(seed)
    D = rng.random((n, 3)) < [0.10, 0.08, 0.06]                    # three drugs, independent exposure
    base = 0.002
    p = np.full(n, base)
    if kind == "combination":                                       # reaction ONLY when all three are present
        p[D.all(1)] = 0.30
    elif kind == "single":                                          # drug A alone causes it
        p[D[:, 0]] = 0.20
    elif kind == "pair":                                            # A+B cause it, C is a bystander
        p[D[:, 0] & D[:, 1]] = 0.30
    R = rng.random(n) < p
    return SyntheticOpenFDA(D, R, ["AAA", "BBB", "CCC"])


def brute_exact(D, R):
    code = (D * (2 ** np.arange(D.shape[1]))).sum(1)
    return {m: (int(R[code == m].sum()), int((code == m).sum())) for m in range(2 ** D.shape[1])}


def test_mobius_inversion_matches_brute_force_counts():
    w = make_world(60_000, seed=3, kind="pair")
    groups = fl.live_group_counts(w, ["AAA", "BBB", "CCC"], "SOME REACTION")
    truth = brute_exact(w.D, w.R)
    for m, cat in enumerate(fl.CATS3):
        assert groups[cat] == truth[m], cat
    assert sum(n for _, n in groups.values()) == len(w.R)
    assert w.calls == 16                                              # N, reaction total, then 7 drug sets x (with / without the reaction) = 16 calls


def test_two_drug_inversion():
    w = make_world(40_000, seed=5)
    g = fl.live_group_counts(w, ["AAA", "BBB"], "X")
    D2 = w.D[:, :2]
    truth = brute_exact(D2, w.R)
    assert [g[c] for c in fl.CATS2] == [truth[m] for m in range(4)]


@pytest.mark.parametrize("kind,expected", [("combination", {"Combination effect", "Probable combination"}),
                                           ("single", {"Single-drug effect"}), ("pair", {"Pair effect"})])
def test_live_verdict_recovers_planted_truth(kind, expected):
    w = make_world(300_000, seed=1, kind=kind)
    out = fc.from_live(w, ["AAA", "BBB", "CCC"], "SOME REACTION")
    assert out["verdict"] in expected, (kind, out["verdict"], out["justification"][:200])
    assert out["source"].startswith("openFDA whole database")


def test_additive_scale_agrees_with_planted_super_additivity():
    w = make_world(300_000, seed=1, kind="combination")
    out = fc.from_live(w, ["AAA", "BBB", "CCC"], "SOME REACTION")
    assert out["additive"]["more_than_additive"] is True


def test_live_pair_path():
    rng = np.random.default_rng(2)
    n = 200_000
    D = rng.random((n, 2)) < [0.15, 0.12]
    p = np.where(D.all(1), 0.25, 0.002)
    w = SyntheticOpenFDA(D, rng.random(n) < p, ["AAA", "BBB"])
    out = fc.from_live(w, ["AAA", "BBB"], "SOME REACTION")
    assert out["verdict"] == "Pair effect" and out["pair_verdict"] == "Pair effect"
    assert out["additive"]["reri"]["lcl"] > 0
