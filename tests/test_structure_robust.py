"""The Structure page must never hang or take the app down: bundled default structures, a time budget on PubChem,
and drawing in a separate process."""
import ast
import time
from pathlib import Path
from types import SimpleNamespace

from pharmasense_ai import chem, similar

ROOT = Path(__file__).resolve().parents[1]


def _default_library():
    tree = ast.parse((ROOT / "app_ai.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "LIBRARY_DEFAULT":
            return ast.literal_eval(node.value)
    raise AssertionError("LIBRARY_DEFAULT not found")


def test_every_default_drug_is_bundled_and_parses():
    lib = similar.bundled_library()
    for name in _default_library():
        assert name in lib, name
        assert chem.mol(lib[name]) is not None, name


class SlowPubChem:
    def __init__(self, delay, answers):
        self.delay, self.answers = delay, answers

    def compound(self, name):
        time.sleep(self.delay)
        a = self.answers.get(name, "unknown")
        if a == "error":
            raise RuntimeError("network down")
        return None if a == "unknown" else {"connectivity_smiles": a}


def _svc(tmp_path, pubchem):
    return SimpleNamespace(settings=SimpleNamespace(cache_dir=tmp_path), pubchem=pubchem)


def test_slow_pubchem_cannot_block_longer_than_the_budget(tmp_path):
    svc = _svc(tmp_path, SlowPubChem(5.0, {"metoprolol": "CC(C)NCC(O)COc1ccc(CCOC)cc1"}))
    t = time.time()
    lib, pending = similar.build_library_timed(["ibuprofen", "metoprolol"], svc, budget_s=0.5)
    assert time.time() - t < 2.0
    assert "ibuprofen" in lib                       # bundled: no network needed
    assert pending == ["metoprolol"]


def test_unreachable_is_pending_not_cached_as_unknown(tmp_path):
    svc = _svc(tmp_path, SlowPubChem(0.0, {"drugx": "error"}))
    lib, pending = similar.build_library_timed(["drugx", "notadrug"], svc, budget_s=5)
    assert pending == ["drugx"]                     # failed: try again later
    assert "drugx" not in lib and "notadrug" not in lib
    lib2, pending2 = similar.build_library_timed(["notadrug"], svc, budget_s=5)
    assert pending2 == []                           # "unknown" answer is remembered, not retried


def test_isolated_drawing_never_raises():
    ok = chem.svgs_isolated([("CC(=O)Oc1ccccc1C(=O)O", 200, 150), ("not-a-smiles", 100, 100)])
    assert ok[0].startswith("<?xml") or "<svg" in ok[0]
    assert ok[1] == ""
    assert chem.svgs_isolated([("CCO", 100, 100)], timeout=0.001) == [""]   # killed child -> no picture, no exception
