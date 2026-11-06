# PharmaSense – from "which drugs are reported together" to "why might they interact, and how sure are we?"

PharmaSense has two layers.

1. **Signal layer** (the original BTP): mines FDA FAERS reports (via openFDA) for harmful 3-drug combinations and decides whether a signal is
   caused by one drug, a pair, or only the combination (ROR/PRR/IC, exact tests, Benjamini–Hochberg, 8 mutually exclusive groups).
2. **Mechanism engine** (`pharmasense_ai/`, new): for any 2–3 drugs it asks *why* — molecular targets → metabolic enzymes → shared proteins →
   safety effect axes → literature — builds **competing hypotheses**, and shows **evidence for and against each**, with an honest probability
   and a list of what is missing.

Three capabilities, one evidence-grounded workflow:

| | What it does | Where |
|---|---|---|
| **Discover** | Finds combination signals that are *not documented* in the component drugs' FDA labels or the retrieved literature; ranks unrecorded drug→reaction links on a knowledge graph (hold-out tested) | `discover.py`, `kg.py` |
| **Explain** | Traces a relationship through enzymes, targets, pathways and label statements; every claim carries a citation id you can open | `agents/`, `reasoning.py`, `report.py` |
| **Challenge** | Lists contradictions, missing evidence and alternative explanations; flags FAERS clusters and one-drug explanations | `agents/nodes.py`, `cluster_check.py` |

It also shows **molecular structure, structurally similar drugs, what is observed for them, and what can only be inferred from them**
(`chem.py`, `similar.py`).

> I make no claim that knowledge graphs, hypergraphs or AI-assisted drug discovery are new. The contribution is combining them with
> **disclosed statistics and cited evidence** so that every output can be audited. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 1. Run it on your own laptop (step by step)

Works on Windows, macOS and Linux. You need **Python 3.10 or newer** (check with `python --version`; on macOS/Linux you may need `python3`),
**git**, and an internet connection (the mechanism engine calls free public APIs). Everything runs on `localhost`; nothing is deployed.

**Step 1 – get the code**
```bash
git clone https://github.com/HRD4317k/pharmasense.git
cd pharmasense
```

**Step 2 – create an isolated environment and install**
```bash
python -m venv .venv
# Windows (PowerShell):   .venv\Scripts\Activate.ps1        Windows (cmd):  .venv\Scripts\activate.bat
# macOS / Linux:          source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
(The first install takes a few minutes: it includes RDKit, LangGraph and XGBoost.)

**Step 3 – add your keys (all free, none strictly required)**
```bash
# Windows:  copy .env.example .env          macOS / Linux:  cp .env.example .env
```
Open `.env` in any editor. Paste your openFDA key after `OPENFDA_API_KEY=` (no quotes or spaces). `.env.example` explains where to get it
and the optional language-model keys. `.env` is git-ignored: never commit it or share it in a zip.

**Step 4 – check that every data source works from your machine**
```bash
python -m pharmasense_ai doctor
```
You should see `[OK ]` for PubChem, ChEMBL, UniProt, openFDA, Europe PMC, STRING and ClinicalTrials.gov. A `[FAIL]` source is simply skipped
(and shown as a gap in reports).

**Step 5 – start the app**
```bash
streamlit run app.py
```
Your browser opens at **http://localhost:8501** (open it yourself if it does not). Stop with `Ctrl+C`.

* **Mechanism explorer**, **Structure & analogues** and **Validation** work immediately (internet needed for the first lookup of each drug; answers are cached in `.cache/`).
* **Overview / Drug search / Combination signals / Discovery** need FAERS results. The repository ships results from the day-by-day sampler, so they
  appear straight away, but you can also regenerate them (Step 6) at any time.

**Step 6 – (re)build the FAERS results from fresh data**
```bash
python run_pipeline.py --max-records 3000     # quick test, a few minutes
python run_pipeline.py                        # 20,000 reports
python run_pipeline.py --skip-fetch --calibrate   # re-mine existing download + permutation-null false-alarm check
python run_pipeline.py --skip-fetch --full        # also the DuckDB analysis and ML models ("ML models" page)
```
Refresh the browser tab afterwards.

**Step 7 – (optional) run the checks**
```bash
python -m pytest tests -q                                  # 54 offline tests, ~10 s
python -m pharmasense_ai validate --calibrate --out data/benchmark/last_run.json   # fills the Validation page
```

### Troubleshooting
| Symptom | Fix |
|---|---|
| `pip install` fails on `rdkit` | Upgrade pip (`pip install --upgrade pip`) and use Python 3.10-3.13 (64-bit). Without RDKit the app still runs; only structures/analogues are unavailable. |
| `streamlit` is not recognised | The virtual environment is not active - repeat the activate line in Step 2. |
| `python` opens Windows Store or fails | Disable Windows App Execution Aliases (Settings -> Apps -> Advanced app settings -> App execution aliases -> disable `python.exe`), or run using `.\.venv\Scripts\python.exe`. |
| "Port 8501 is already in use" | `streamlit run app.py --server.port 8502` |
| Pages say "No results yet" | Run Step 6 first. |
| `[FAIL]` for every source in `doctor` | Check your internet / VPN / proxy. University or office networks sometimes block API hosts. |
| openFDA errors or HTTP 429 | Add your openFDA key (Step 3); without one you only get 1,000 requests/day. Wait a minute and retry. |
| DuckDB "could not set lock" | Another script is using `data/processed/faers.db`; close it first. |
| Old data keeps appearing | Delete the `.cache/` folder to force fresh API answers. |

## 2. Mechanism engine from the command line
```bash
python -m pharmasense_ai explain warfarin fluconazole --reaction "INTERNATIONAL NORMALISED RATIO INCREASED" --live-faers
python -m pharmasense_ai live-faers warfarin fluconazole simvastatin --reaction RHABDOMYOLYSIS   # whole-database counts + verdict
python -m pharmasense_ai discover -n 8                      # screen your local results for undocumented combination signals
python -m pharmasense_ai analogs ibuprofen                  # structure + look-alikes
python -m pharmasense_ai validate --calibrate --out data/benchmark/last_run.json
python -m pytest tests -q                                   # 54 offline tests
```

## 3. The app
| Page | What it does |
|---|---|
| Overview · Drug search · Combination signals | the original signal layer |
| **Glossary & formulas** | definitions, every formula (new tab: the mechanism-engine mathematics) |
| ML models | `advanced_models.py` results (after `--full`): Lasso, Random Forest and XGBoost scored with a random split and a batch-aware split |
| **Mechanism explorer** | Explain + Challenge for any 2–3 drugs, optional FAERS context (whole database or local), downloadable report |
| **Structure & analogues** | structure, descriptors, look-alikes under four fingerprints, observed vs *inferred* reactions, activity-cliff flags |
| **Discovery** | undocumented-combination screen of your results |
| **Validation** | benchmark results, the benchmark list itself, the permutation-null check |

## 4. How far to trust it (read this)
* **Verified here** (offline): the maths (unit tests, formula self-check), the Möbius-inversion counting against brute force on synthetic
  data with *planted* interactions, the workflow logic and the app against hand-written fixtures, the refactor of the original statistics
  (results identical to 1e-10 on your shipped data), and the permutation null (0 false alarms in 20 shuffles of your data).
* **Live API diagnostic passed**: all public APIs (openFDA, PubChem, ChEMBL, UniProt, Europe PMC, STRING, ClinicalTrials.gov) tested and passing via `python -m pharmasense_ai doctor`.
* A high "posterior" means *the retrieved evidence favours this mechanism*, not that a patient will be harmed. This is research software, not clinical advice.
* FAERS is voluntary-report data: it shows what was reported, not incidence, and clusters (litigation, studies) can fake combination signals. The
  cluster check flags this; the shipped sample bunched on 15 receive dates, which the new day-by-day fetcher fixes – **re-run the pipeline**.

## Files
| File / folder | Role |
|---|---|
| `fetch_fda_api.py`, `process_and_mine.py`, `pharmasense_core.py`, `exclusions.py` | signal layer (fetch day-by-day, clean, mine, test, verdicts) |
| `app.py`, `app_ai.py` | dashboard; `app_ai.py` holds the four mechanism-engine pages |
| `pharmasense_ai/` | clients (PubChem, ChEMBL, UniProt, Open Targets, Europe PMC, STRING, ClinicalTrials.gov, openFDA), label miner, RAG, hypotheses, evidence maths, LangGraph workflow, KG, analogues, discovery, validation, CLI |
| `data/benchmark/` | hand-curated known-interaction list (+ gold enzyme roles) |
| `docs/` | architecture and validation notes |
| `tests/` | 54 offline tests (`python -m pytest tests -q`) |
| `legacy/`, `src/analysis/*`, `load_to_duckdb.py` | earlier BTP material (DuckDB PRR rules, regression, ML) |
