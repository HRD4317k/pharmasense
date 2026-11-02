"""Reaction terms that are NOT genuine adverse events, and why.

EDIT THIS FILE to tune what the app ignores. Every term is an exact, UPPER-CASE MedDRA
Preferred Term as it appears in openFDA. A report's reaction is dropped before any statistics
are computed if it matches here, or if it equals the indication of any drug in the same report
(done in pharmasense_core.prepare_reports - that rule is data-driven, no list needed).

Why: "Feeling abnormal", "Drug ineffective", "Death" ... describe the patient's circumstances,
the disease, or the paperwork - not a specific medical event caused by a drug. They are very
common and highly clustered, so they produce large, meaningless odds ratios.
(Mapping terms to MedDRA System Organ Classes would be cleaner, but MedDRA is licensed, so this
list is a curated heuristic.)
"""
import re

CATEGORIES = {
    "Vague or non-specific wording": {
        "FEELING ABNORMAL", "FEELING HOT", "FEELING COLD", "FEELING JITTERY", "FEELING DRUNK",
        "FEELING OF RELAXATION", "FEELING OF BODY TEMPERATURE CHANGE", "MALAISE", "ILLNESS",
        "GENERAL PHYSICAL HEALTH DETERIORATION", "GENERAL PHYSICAL CONDITION ABNORMAL",
        "GENERAL PHYSICAL CONDITION DECREASED", "ADVERSE EVENT", "ADVERSE REACTION",
        "ADVERSE DRUG REACTION", "NO ADVERSE EVENT", "UNEVALUABLE EVENT", "DRUG INTERACTION",
        "ASYMPTOMATIC", "SYMPTOM RECURRENCE",
    },
    "Lack of effect / disease course": {
        "DRUG INEFFECTIVE", "DRUG EFFECT DECREASED", "DRUG EFFECT INCOMPLETE", "DRUG EFFECT DELAYED",
        "THERAPEUTIC RESPONSE DECREASED", "THERAPEUTIC RESPONSE SHORTENED",
        "THERAPEUTIC RESPONSE UNEXPECTED", "TREATMENT FAILURE", "THERAPY NON-RESPONDER",
        "DISEASE PROGRESSION", "DISEASE RECURRENCE", "CONDITION AGGRAVATED",
    },
    "Medication errors & product issues": {
        "OFF LABEL USE", "PRODUCT USE IN UNAPPROVED INDICATION", "PRODUCT USE ISSUE",
        "PRODUCT DOSE OMISSION ISSUE", "INCORRECT DOSE ADMINISTERED",
        "INAPPROPRIATE SCHEDULE OF PRODUCT ADMINISTRATION", "UNDERDOSE", "EXTRA DOSE",
        "MEDICATION ERROR", "DRUG ADMINISTRATION ERROR", "NO ADVERSE EFFECT",
    },
    "Misuse, overdose & self-harm": {
        "OVERDOSE", "INTENTIONAL OVERDOSE", "ACCIDENTAL OVERDOSE", "DRUG ABUSE", "DRUG DEPENDENCE",
        "SUBSTANCE ABUSE", "TOXICITY TO VARIOUS AGENTS", "SUICIDE ATTEMPT", "COMPLETED SUICIDE",
        "INTENTIONAL PRODUCT MISUSE", "INTENTIONAL PRODUCT USE ISSUE",
    },
    "Outcomes (not events)": {
        "DEATH", "HOSPITALISATION", "HOSPITALIZATION", "SUDDEN DEATH",
    },
    "Underlying diseases": {
        "PLASMA CELL MYELOMA", "COVID-19", "ASTHMA", "RHEUMATOID ARTHRITIS", "PSORIASIS",
    },
}

# Patterns for families of terms (checked after the exact lists)
PATTERNS = [
    (re.compile(r"^PRODUCT "), "Medication errors & product issues"),
    (re.compile(r"^(INCORRECT|WRONG) "), "Medication errors & product issues"),
    (re.compile(r"^ACCIDENTAL (EXPOSURE|INGESTION)"), "Medication errors & product issues"),
    (re.compile(r"^INTENTIONAL "), "Misuse, overdose & self-harm"),
    (re.compile(r"^(MATERNAL|PATERNAL|FOETAL|FETAL) EXPOSURE"), "Exposure / circumstance only"),
    (re.compile(r"^EXPOSURE (DURING|VIA|TO)"), "Exposure / circumstance only"),
]

_EXACT = {t: cat for cat, terms in CATEGORIES.items() for t in terms}


def classify(term):
    """Return the exclusion category for a reaction term, or None if it is a real event."""
    t = term.strip().upper()
    if t in _EXACT:
        return _EXACT[t]
    for rx, cat in PATTERNS:
        if rx.search(t):
            return cat
    return None
