# PharmaSense – drug and 3-drug-combination safety signals (openFDA)

## 1. One-time setup
```bash
python -m venv .venv
# Windows:   .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```
Open `.env` and put your openFDA key on the line `OPENFDA_API_KEY=...` (no quotes, no spaces).
Keep `.env` out of git and out of any zip you share.

## 2. Run
```bash
python run_pipeline.py --max-records 3000    # quick first test
python run_pipeline.py                       # default 20,000 reports
streamlit run app.py                         # http://localhost:8501
```
Optional BTP extras (pair PRR/chi-square rules, regression, ML models): `python run_pipeline.py --skip-fetch --full`
Re-mine without re-downloading: `python run_pipeline.py --skip-fetch`  (or `python process_and_mine.py --help`).

## The app
| Page | What it does |
|---|---|
| Overview | Counts, who-is-responsible breakdown, histogram showing where ROR values really sit |
| **Drug search** | One page: pick a drug → its reactions → pick a reaction → *drug alone, partner alone, pair, or 3-drug combination?* with evidence chart and plain-language justification |
| Combination signals | Every 3-drug signal with verdict, filters and evidence |
| **Glossary & formulas** | Plain definitions, every formula, how verdicts are decided, what is excluded, limitations |
| ML models | Results of `advanced_models.py` (after `--full`) |

## Files
| File | Role |
|---|---|
| `fetch_fda_api.py` | openFDA download (one row per report; keeps drug roles, indications, generic names) |
| `exclusions.py` | **Editable** list of non-adverse-event terms ("Feeling abnormal", "Drug ineffective", …) |
| `pharmasense_core.py` | All statistics: ROR / PRR / IC, name clean-up, attribution, partner analysis |
| `process_and_mine.py` | Clean → FP-Growth → test → write `data/results/` |
| `app.py` | The dashboard |
| `load_to_duckdb.py`, `src/analysis/*` | DuckDB layer for PRR rules, regression, ML (`--full`) |
| `legacy/` | Your previous dashboards and miner, kept for reference only |
| `tests/check_formulas.py` | 10-second self-check of the formulas |

## Tuning
`python process_and_mine.py --min-triplet 15` (more triplets) · `--min-events 3` · `--drug-scope suspect`
(only drugs flagged suspect/interacting) · edit `exclusions.py` to change what counts as a non-event.
Close any script that writes `data/processed/faers.db` before starting another (DuckDB allows one writer).
