# What changed, and where

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
