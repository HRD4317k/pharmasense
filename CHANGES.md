# What changed, and where

## Round 4 - "Improved UI" + ML models corrected

| Area | Files | Notes |
|---|---|---|
| Visual identity | `ui.py` (new), `.streamlit/config.toml` (new), `app.py`, `app_ai.py` | One palette and stylesheet for every page: capsule logo (two halves = two drugs), hexagon-grid hero, card-style metrics and charts, pill tabs, grouped sidebar navigation, one Plotly style. Colour always means something: teal = drug / neutral, magenta = risk or combination, amber = pair / caution, green = supported. The theme also lives in `.streamlit/config.toml`, so it applies on Streamlit Community Cloud |
| ML models page | `app.py` | Rewritten: plain-language "what each model is doing", honest results, a leakage explainer, importance chart coloured by the Lasso's direction. Explains the numbers instead of only plotting them |
| **ML bug fixed** | `src/analysis/advanced_models.py` | The shipped ML result was an artefact. Its target, "ILL-DEFINED DISORDER", is a term the project's own `exclusions.py` discards, and 94% of its reports were received on one date (one batch), so a random split leaked near-duplicates across train and test and gave AUC 0.97. The script now (1) never targets an excluded term, (2) skips any term with more than half of its reports on one date, (3) reports a random split **and** a batch-aware split (GroupKFold on receive date) side by side, (4) uses absolute paths, (5) fixes a scikit-learn 1.8 deprecation of `penalty="l1"` (chosen by version, so older versions still get a real L1 model) and (6) uses a stronger L1 penalty so the Lasso actually selects drugs (17 of 50 kept) |
| New honest result | `models/`, `data/processed/model_metrics.json` | Target FATIGUE (6.4% of reports): batch-aware AUC 0.61 Lasso / 0.54 Random Forest / 0.51 XGBoost, where 0.5 is guessing. Which common drugs are on a report says very little about a non-specific symptom. This is the expected result, and the reason PharmaSense uses disproportionality statistics and mechanism evidence instead of a black-box predictor |
| Tests | `tests/test_ml_models.py` | 4 new tests (58 in total) |
| Still true | `data/` | The committed data still comes from the old sampler (15 receive dates). Re-run `python run_pipeline.py --full` on your machine for a better sample; the ML page then updates itself |

---

## Round 3 - mechanism engine (now on `main`)

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
| Tests | `tests/` | 54 offline tests at the time (`python -m pytest tests -q`) |
| Done in round 4 | `src/analysis/advanced_models.py` | batch-aware (grouped by receive date) evaluation; see round 4 |

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
