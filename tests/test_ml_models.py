"""src/analysis/advanced_models.py: target choice (no vague terms, no single-batch terms) and batch-aware scoring."""
import importlib.util
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("advanced_models", ROOT / "src" / "analysis" / "advanced_models.py")
am = importlib.util.module_from_spec(spec)
spec.loader.exec_module(am)


@pytest.fixture()
def con():
    """300 cases over 10 receive dates. D1 genuinely raises 'REAL'; 'VAGUE' is the most frequent but excluded by exclusions.py;
    'BATCHY' is frequent but sits on one date."""
    rng = np.random.default_rng(0)
    n = 300
    dates = np.array([f"2023{m:02d}01" for m in range(1, 11)])[np.arange(n) % 10]
    d1 = rng.random(n) < 0.3
    others = rng.random((n, 5)) < 0.3
    real = (d1 & (rng.random(n) < 0.8)) | (~d1 & (rng.random(n) < 0.05))
    vague = rng.random(n) < 0.6
    batchy = (dates == dates[0]) & (rng.random(n) < 0.9)
    ids = [f"c{i}" for i in range(n)]
    c = duckdb.connect(":memory:")
    c.register("demo_df", pd.DataFrame({"caseid": ids, "receive_date": dates}))
    c.execute("CREATE TABLE demo AS SELECT * FROM demo_df")
    rows_d = [(ids[i], "D1") for i in range(n) if d1[i]] + [(ids[i], f"O{j}") for i in range(n) for j in range(5) if others[i, j]]
    c.register("drug_df", pd.DataFrame(rows_d, columns=["caseid", "drugname"]))
    c.execute("CREATE TABLE clean_drug AS SELECT * FROM drug_df")
    rows_r = ([(ids[i], "REAL") for i in range(n) if real[i]] + [(ids[i], "ILL-DEFINED DISORDER") for i in range(n) if vague[i]]
              + [(ids[i], "BATCHY") for i in range(n) if batchy[i]])
    c.register("reac_df", pd.DataFrame(rows_r, columns=["caseid", "reaction"]))
    c.execute("CREATE TABLE clean_reac AS SELECT * FROM reac_df")
    return c


def test_default_target_skips_vague_and_single_batch_terms(con):
    assert am.pick_target(con) == "REAL"


def test_date_concentration_flags_the_batch(con):
    conc = am.date_concentration(con)
    assert conc.loc["BATCHY", "top_share"] > 0.9
    assert conc.loc["REAL", "top_share"] < 0.3


def test_batch_aware_cv_keeps_dates_together_and_finds_the_real_signal(con):
    X, y, groups, names = am.prepare_data(con, "REAL", num_drugs=6)
    assert len(set(groups)) == 10
    P, skipped = am.out_of_fold(X, y, GroupKFold(5), groups)
    assert skipped == 0
    for name, p in P.items():
        assert not np.isnan(p).any(), name
        assert am.score(y, p)["ROC AUC"] > 0.7, name        # D1 truly drives REAL, so every model should beat chance clearly
    # no date appears in both the training and the testing part of any fold
    for tr, te in GroupKFold(5).split(X, y, groups):
        assert not set(groups[tr]) & set(groups[te])


def test_lasso_is_sparse_and_is_really_l1():
    m = am._lasso()
    assert m.get_params()["C"] == 0.1
    rng = np.random.default_rng(1)
    X = rng.integers(0, 2, (400, 30))
    y = (X[:, 0] & (rng.random(400) < 0.9)).astype(int)
    m.fit(X, y)
    assert (np.abs(m.coef_[0]) < 1e-9).sum() >= 10          # an L2 model would have no exact zeros
