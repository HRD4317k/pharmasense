import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fakes import PAPERS_NULL, PAPERS_SUPPORT, FakeServices  # noqa: E402

from pharmasense_ai import validate as v  # noqa: E402


def small_bench():
    return pd.DataFrame([
        dict(drug_a="warfarin", drug_b="fluconazole", reaction="", expected_type="PK-enzyme", expected_term="CYP2C9", perpetrator="fluconazole", label=1),
        dict(drug_a="pravastatin", drug_b="ketoconazole", reaction="", expected_type="PK-enzyme", expected_term="CYP3A4", perpetrator="ketoconazole", label=0),
    ])


def test_benchmark_csv_is_well_formed():
    d = v.load_benchmark()
    assert set(d.label) == {0, 1} and (d.label == 1).sum() >= 15 and (d.label == 0).sum() >= 8
    assert d.expected_type.isin(["PK-enzyme", "PK-transporter", "PD-effect-axis"]).all()
    assert not d.duplicated(["drug_a", "drug_b", "expected_term"]).any()
    # every negative must name a mechanism, otherwise it is not "mechanism-specific"
    assert (d[d.label == 0].expected_term != "").all()


def test_positive_recovered_negative_not_supported():
    svc = FakeServices(papers=[{**PAPERS_SUPPORT, "_needs": ["warfarin", "fluconazole"]}])
    res, reports = v.run_benchmark(svc, small_bench(), log=lambda *_: None)
    pos, neg = res.iloc[0], res.iloc[1]
    assert pos.supported and pos.posterior > 0.6
    assert not neg.supported
    s = v.summarise(res)
    assert s["recall"]["value"] == 1.0 and s["false_positive_rate"]["value"] == 0.0 and s["auroc"] == 1.0


def test_summarise_intervals_and_missed_listed():
    res = pd.DataFrame({"drug_a": list("abcd"), "drug_b": list("wxyz"), "expected": ["t"] * 4, "label": [1, 1, 0, 0], "proposed": [True, False, True, False],
                        "verdict": ["supported", "not proposed", "supported", "untested"], "posterior": [0.9, 0.0, 0.7, 0.1],
                        "supported": [True, False, True, False], "n_other_supported": [0] * 4})
    s = v.summarise(res)
    assert s["recall"]["value"] == 0.5 and s["recall"]["ci95"][0] < 0.5 < s["recall"]["ci95"][1]
    assert s["missed"][0]["drug_a"] == "b" and s["false_alarms"][0]["drug_a"] == "c"


def test_calibration_refuses_tiny_benchmarks(tmp_path):
    svc = FakeServices(papers=[{**PAPERS_SUPPORT, "_needs": ["warfarin", "fluconazole"]}])
    res, reports = v.run_benchmark(svc, small_bench(), log=lambda *_: None)
    out = v.calibrate(small_bench(), reports, out_path=tmp_path / "c.json")
    assert out["status"] == "skipped" and not (tmp_path / "c.json").exists()


def test_calibration_keeps_signs_and_writes_loadable_file(tmp_path, monkeypatch):
    """Synthetic reports where label_ddi / label_class perfectly separate the classes: fitted weights must stay positive,
    the contradiction weight non-positive, and the output must load through the workflow's loader."""
    rng = np.random.default_rng(0)
    rows, reports = [], []
    for i in range(30):
        pos = i % 2 == 0
        ev = {}
        sup = []
        if pos:
            ev = {"E1": {"kind": "label_class", "group": "g1"}, "E2": {"kind": "label_ddi", "group": "g2"}}
            sup = ["E1", "E2"]
        elif rng.random() < 0.3:
            ev = {"E1": {"kind": "label_context", "group": "g1"}}
            sup = ["E1"]
        reports.append({"evidence": ev, "hypotheses": [{"mechanism_type": "PK-enzyme", "term": "CYP3A4", "perpetrator": "x", "prior": 0.25, "supporting": sup, "contradicting": [],
                                                       "verdict": "x", "posterior": 0.5}]})
        rows.append(dict(drug_a="a", drug_b="b", reaction="", expected_type="PK-enzyme", expected_term="CYP3A4", perpetrator="x", label=int(pos)))
    out_path = tmp_path / "cal.json"
    out = v.calibrate(pd.DataFrame(rows), reports, out_path=out_path)
    assert out["status"] == "fitted" and out["n"] == 30
    assert all(out["log_lr"][k] >= 0 for k in out["log_lr"] if k != "contradiction") and out["log_lr"]["contradiction"] <= 0
    assert out["leave_one_out"]["brier"] <= out["default_weights"]["brier"] + 1e-9 or not out["adopted"]
    if out["adopted"]:
        from pharmasense_ai.agents import workflow
        monkeypatch.setattr(workflow, "CALIBRATION_FILE", out_path)
        table, cf = workflow.load_calibration()
        assert table["_calibrated"] and cf.n == 15
