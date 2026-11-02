# Architecture

## The question the system answers
*Why might two (or three) drugs produce an unexpected adverse effect — and how sure are we?*
Each step maps to code, and each step leaves citable evidence rather than a bare assertion.

| Step | Module | Source (free, no paid key) |
|---|---|---|
| 1 molecular targets | `profile.py`, `clients/chembl.py`, `clients/opentargets.py` | ChEMBL, Open Targets |
| 2 metabolic pathways | `clients/uniprot.py` (Reactome cross-references), `label_parse.py` | UniProt, FDA label |
| 3 shared enzymes / transporters | `label_parse.extract_roles` → `reasoning.generate` | FDA label (structured class fields first, then sentences) |
| 4 relevant proteins | `clients/others.StringDB` | STRING |
| 5 clinical literature | `rag.py`, `clients/europepmc.py` | Europe PMC (indexes PubMed) |
| 6 indirect mechanisms | `kg.py`, `reasoning` (`indirect-network`, `PD-pathway`) | the graph built above |
| 7 competing hypotheses | `reasoning.generate` | rules (+ optional LLM proposals, validated) |
| 8 evidence for / against | `reasoning.attach_*`, `finalize`, `score`; `agents/nodes.challenge` | all of the above + FAERS |

## LangGraph workflow (`agents/workflow.py`)
```
START → profile → network → hypothesize → retrieve → challenge → score ─┬─ (some hypothesis untested, loops left) → bump ─┐
                                                       ▲                 │                                                  │
                                                       └─────────────────┼──────────────────────────────────────────────────┘
                                                                         └─ otherwise → critic → report → END
```
* `profile` builds a `DrugProfile` per drug (structure, targets, enzyme roles, safety axes, indications). A source that fails becomes a **gap**,
  not a silent omission.
* `hypothesize` proposes mechanism families: `PK-enzyme`, `PK-transporter`, `PD-shared-target`, `PD-effect-axis`, `PD-pathway`, `indirect-network`,
  `documented`, and (only if an LLM is configured) `llm-proposed`.
* `retrieve` → BM25 over label sections and abstracts; keeps only sentences naming **both** drugs and the mechanism term.
* `challenge` looks for contradictions (explicit negation naming both drugs), one-drug explanations in FAERS, missing links to the reported reaction.
* `bump → retrieve` is the **feedback edge**: if a hypothesis is still untested for lack of literature, search again with its specific enzyme/axis
  terms (once) before reporting the gap.
* `critic` runs once; `report` assembles the JSON/Markdown report and validates any LLM text.

Why a graph and not a script: the loop, the conditional routing and the once-only critic are real control flow, and each node can be tested alone.

## The mathematics, and where it earns its place
* **Disproportionality + exact tests + BH** (`pharmasense_core`): the signal layer; permutation-null checked.
* **Möbius inversion** (`faers_live.py`): openFDA has no NOT operator, so exact exposure groups are recovered from "at least these drugs" counts,
  16 requests per triplet. Tested against brute force.
* **Additive-scale interaction** (`mathx/interaction.py`): RERI, synergy index, third-order contrast I₃ with parametric bootstrap — because an effect can be
  super-multiplicative yet sub-additive, and the original test was multiplicative only.
* **Dependence-aware evidence combination** (`mathx/evidence.py`): n_eff = n / (1 + ρ(n−1)) so ten sentences of one label are not ten confirmations;
  Monte-Carlo credible interval over uncertainty in the weights; noisy-OR for independent routes.
* **Calibration** (`validate.py`, `mathx/uncertainty.py`): L2-regularised log-LR fit with leave-one-out comparison against the defaults; split-conformal
  acceptance; Brier / ECE / Beta and bootstrap intervals.
* **Graph maths** (`mathx/graphmath.py`, `kg.py`): personalised PageRank, spectral embedding, Adamic–Adar, hold-out AUROC. A scorer is only trusted if it beats
  chance on hidden edges of *this* graph.
* **Structure** (`chem.py`): four fingerprints (Morgan, feature-Morgan, atom-pair, torsion), Murcko scaffolds; similarity is reported per fingerprint because one number hides disagreement.
* **Cluster diagnostics** (`cluster_check.py`): Monte Carlo against the exposed-report pool — a combination "signal" made of one litigation batch is flagged.

## Design rules that keep it honest
1. Numbers come from code, not from a language model. The LLM (optional) can word things and propose hypotheses; every citation it uses must exist in the
   retrieved evidence or the text is discarded.
2. Missing data is a **gap**, never evidence against.
3. "Novel" is defined operationally in `discover.py` (not in the component labels, not in the retrieved abstracts) and never means "unknown to science".
4. Results are labelled *uncalibrated* until `validate --calibrate` has been run, and calibration is adopted only if it beats the defaults under leave-one-out.
5. Standing caveats are part of every report.

## Data sources and licences
openFDA (public domain), PubChem (public domain), UniProt (CC BY 4.0), ChEMBL (CC BY-SA 3.0), Open Targets (CC0), Europe PMC (abstract text
subject to publisher terms — the system stores only short quoted sentences with their link), STRING (CC BY 4.0), ClinicalTrials.gov (public domain).
Check each licence before redistributing derived data.
