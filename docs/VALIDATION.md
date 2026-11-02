# Validation: what was checked, how, and what it does not show

## Checked in the build environment (offline)
| Claim | How it was checked | Result |
|---|---|---|
| Original statistics unchanged by refactor (`add_exact_tests` extracted) | regression script on the shipped `triplet_results` | 344 rows, same keys, max numeric difference 1.2e-10, verdicts and q-values identical |
| Pipeline reports no signal when there is none | permutation null: shuffle reaction lists, rerun the same evaluation + verdict code, 20 times | 0 signals per shuffle on average (observed: 81 signals, 8 combination-like) |
| Möbius inversion rebuilds exact groups | synthetic FAERS (300k reports) vs brute force | identical for 2 and 3 drugs; 16 requests per triplet |
| Live-count verdict logic | synthetic worlds with a **planted** combination / single-drug / pair effect | recovered in all three; additive-scale test agrees |
| Math helpers | `tests/test_mathx.py`, `tests/check_formulas.py` | pass |
| Workflow logic, KG, analogues, discovery, validation, UI | hand-written API-shaped fixtures (`tests/fakes.py`) | 54 tests pass |

The fixtures are **hand-written to look like the documented API responses**; they are not captured live data and the label sentences are paraphrases.

## NOT checked (you must check these)
* Every live API call. Field names, query syntax and rate behaviour follow public documentation but were never exercised from here.
  Run `python -m pharmasense_ai doctor`. Known uncertain points: PubChem property names, Open Targets drug fields, openFDA `generic_name`
  matching, the Gemini OpenAI-compatible URL.
* Precision of the label text-miner on real labels. `validate` reports its **recall** against `data/benchmark/gold_roles.csv` (16 hand-written roles); precision is not estimated.
* Whether the evidence weights are well calibrated. Until you run `validate --calibrate` they are judgement priors and reports say so.

## The benchmark (`data/benchmark/known_interactions.csv`)
About 20 positive and 9 mechanism-specific negative pairs, written from standard pharmacology. Limits:
* Positives are in drug labels and the engine reads drug labels, so a high score is a **sanity check on recovering documented mechanisms**, not evidence that Discover works on unknown ones.
* A negative means "this mechanism is not established for this pair", not "the pair is safe". One negative (pantoprazole–clopidogrel/CYP2C19) is arguable.
* One author plus an AI assistant; no independent audit. Please audit it before quoting any number.
* ~30 rows: every interval is wide. Calibration fits ~8 weights; it is adopted only if leave-one-out Brier beats the defaults, and the output includes an overfitting warning.

## How to read the output
`recall` = share of positives whose matching hypothesis reaches *supported*; `false_positive_rate` = share of negatives that do; `AUROC` uses the posterior (an unproposed hypothesis scores 0).
Compare against the listed `missed` and `false_alarms` rows rather than the headline numbers.

## Known limitations of the signal layer
FAERS is voluntary; duplicates and litigation clusters can fake combination signals; the shipped sample was bunched on 15 receive dates (an artefact of the old fetcher) — re-fetch
with the day-by-day sampler and run `python run_pipeline.py --skip-fetch --calibrate`.
