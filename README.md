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

## 1. Setup
```bash
python -m venv .venv
# Windows:   .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```
Open `.env`. **All keys are free; none is mandatory** – `.env.example` says where to get each one (openFDA key, optional Groq / Gemini /
OpenRouter / local Ollama for a language model). Then check what works on your machine:
```bash
python -m pharmasense_ai doctor
```
The live behaviour of the web APIs could not be tested while this code was written (no network in the build environment), so `doctor` is
the first thing to run. A source that fails is skipped and shown as a *gap* in reports; it is never treated as evidence against a mechanism.

## 2. Run
```bash
python run_pipeline.py --max-records 3000     # quick test of the signal layer (day-by-day sampling across the year)
python run_pipeline.py                        # 20,000 reports
streamlit run app.py                          # http://localhost:8501
```

### Mechanism engine from the command line
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
| ML models | `advanced_models.py` results (after `--full`) |
| **Mechanism explorer** | Explain + Challenge for any 2–3 drugs, optional FAERS context (whole database or local), downloadable report |
| **Structure & analogues** | structure, descriptors, look-alikes under four fingerprints, observed vs *inferred* reactions, activity-cliff flags |
| **Discovery** | undocumented-combination screen of your results |
| **Validation** | benchmark results, the benchmark list itself, the permutation-null check |

## 4. How far to trust it (read this)
* **Verified here** (offline): the maths (unit tests, formula self-check), the Möbius-inversion counting against brute force on synthetic
  data with *planted* interactions, the workflow logic and the app against hand-written fixtures, the refactor of the original statistics
  (results identical to 1e-10 on your shipped data), and the permutation null (0 false alarms in 20 shuffles of your data).
* **Not verified**: every live API call (field names and query syntax follow the public documentation but were not exercised), the precision of the
  label text-miner on real labels, and the evidence weights – they are *judgement priors* until you run `validate --calibrate`, and even then
  they rest on a ~30-row hand-curated benchmark. See [docs/VALIDATION.md](docs/VALIDATION.md).
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
