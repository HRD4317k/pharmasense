"""Fast self-check of the statistics:  python tests/check_formulas.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pharmasense_core as c

r, lo, hi = c.ror(20, 80, 100, 9800)                    # textbook: ROR = a*d/(b*c) = 24.5
assert abs(r - 24.5) < 1e-9 and lo < r < hi
assert abs(c.ror(10, 90, 100, 900)[0] - 1.0) < 1e-9     # no association -> exactly 1
assert abs(c.prr(20, 80, 100, 9800)[0] - 19.8) < 1e-9    # (20/100)/(100/9900)... = 19.8
assert c.ror(0, 10, 5, 100)[0] > 0                       # zero cell handled (Haldane only when needed)
assert c.normalize_drug("METFORMIN HCL 500 MG TABLET") == "METFORMIN"
assert c.normalize_drug("SODIUM CHLORIDE") == "SODIUM CHLORIDE"
assert c.overlapping("LISINOPRIL", "LISINOPRIL AND HYDROCHLOROTHIAZIDE")
import exclusions
assert exclusions.classify("Feeling abnormal") and exclusions.classify("RHABDOMYOLYSIS") is None
print("all formula checks passed")
