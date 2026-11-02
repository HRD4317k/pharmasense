# What changed, and where

## Round 3 - mechanism engine (branch `feature/discovery-engine`)

| Area | Files | Notes |
|---|---|---|
| New package | `pharmasense_ai/` | Explain / Challenge / Discover workflow (LangGraph), BM25 RAG over labels + abstracts, evidence maths, KG, analogues, validation, CLI (`python -m pharmasense_ai`). See `docs/ARCHITECTURE.md` |
| App | `app.py` (small edit), `app_ai.py` (new) | four new pages + a "Mechanism engine" glossary tab; the original pages are untouched and the app still starts if the new dependencies are missing |
| Sampling bug fixed | `fetch_fda_api.py` | the old fetcher returned reports from only 15 distinct receive dates (first records of each month), which made the combination signals look more clustered than they are. Now samples day by day (`--sample-days`). **Re-fetch to benefit** - the committed `data/` still comes from the old fetcher |
| Refactor | `pharmasense_core.py` | `add_exact_tests()` extracted from `evaluate_triplets()` so whole-database counts reuse the same test. Output verified identical (max diff 1.2e-10) |
| Exclusions | `exclusions.py` | "ILL-DEFINED DISORDER", "DISORDER", "SYMPTOM", "FEELING UNWELL" are now excluded as vague. The committed results still contain "ILL-DEFINED DISORDER" (23 triplet rows) until you re-run `process_and_mine.py` |
| Diagnostics | `run_pipeline.py` | data-quality check after mining; `--calibrate` runs the permutation null |
| Config | `.env.example` (rewritten), `requirements.txt`, `.gitignore` | where to get every free key; new dependencies; `.cache/` ignored |
| Benchmark | `data/benchmark/` | hand-curated known interactions + gold enzyme roles (audit them) |
| Tests | `tests/` | 54 offline tests (`python -m pytest tests -q`) |
| Not done | `src/analysis/advanced_models.py` | a time-based split and cluster-robust check for the ML models were planned but not implemented |

---

## Round 2 - signal layer rewrite

| Your file | Status | What / where |
|---|---|---|
| `process_and_mine.py` | **rewritten** | `calculate_ror()` (always +0.5, Wald only) → `pharmasense_core.ror()`; mining no longer done *inside* each reaction (selection bias) but globally; adds exact test + FDR, exclusive groups, verdicts |
| `fetch_fda_api.py` | **changed** | `parse_report()` now keeps drug role, indication, generic name; per-case dedupe at the end of `fetch_data()`; output is one row per report (no `explode`) |
| `dashboard.py`, `dashboard_v1/app.py` | **replaced** by `app.py` (old copies in `legacy/`) | one app: Overview, Drug search, Combination signals, Glossary, ML |
| `src/analysis/polypharmacy_miner.py` | **superseded** (in `legacy/`) | its gates now live in `pharmasense_core.evaluate_triplets()` / `classify_triplets()` |
| `src/analysis/signal_detection.py` | **patched** | `calculate_ror()` / `calculate_prr()`: +0.5 only when a cell is zero; chi-square on raw counts |
| `src/analysis/advanced_models.py` | **patched** | default `--target` = most frequent reaction ("Overdose" is now excluded; the old default could never match) |
| `src/analysis/eda.py`, `association_rules.py`, `logistic_regression.py` | unchanged from the previous round | |
| `load_to_duckdb.py`, `run_pipeline.py` | **updated** | read `data/results/reports_clean.parquet`; new steps |
| `exclusions.py`, `pharmasense_core.py`, `app.py`, `tests/` | **new** | |
