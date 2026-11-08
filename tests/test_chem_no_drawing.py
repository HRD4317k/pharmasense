"""If RDKit's drawing module cannot be imported (missing libXrender on a bare server), everything except the picture must still work."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CODE = """
import sys
sys.modules['rdkit.Chem.Draw'] = None          # makes 'from rdkit.Chem.Draw import ...' raise ImportError
from pharmasense_ai import chem, similar          # must import cleanly
asp, ibu = 'CC(=O)Oc1ccccc1C(=O)O', 'CC(C)Cc1ccc(cc1)C(C)C(=O)O'
print('mean', round(chem.similarity_profile(asp, ibu)['mean'], 3))
print('desc', 'mw' in {k.lower() for k in chem.descriptors(asp)} or bool(chem.descriptors(asp)))
print('svg', repr(chem.svg(asp)))
"""


def test_everything_but_the_picture_works_without_the_drawing_module():
    r = subprocess.run([sys.executable, "-c", CODE], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    assert "mean " in r.stdout and "desc True" in r.stdout
    assert "svg ''" in r.stdout            # empty string, so the page shows "structure image not available"


def test_packages_txt_lists_the_system_libraries():
    pk = (ROOT / "packages.txt").read_text().split()
    assert "libxrender1" in pk
