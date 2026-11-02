"""Mine FDA label text for the facts the mechanism engine needs: which enzymes / transporters a drug is a
substrate, inhibitor or inducer of, which safety "axes" it carries (QT, bleeding, serotonin syndrome ...), and
whether one drug's label names another.

This is rule-based text mining, not a database. Every result keeps its source sentence and a confidence tag:
  structured  openFDA `pharm_class_moa` field ("Cytochrome P450 3A4 Inhibitors [MoA]")          most reliable
  explicit    a sentence whose subject is this drug ("X is a potent inhibitor of CYP3A4")
  context     indirect wording ("strong CYP3A4 inhibitors increase X exposure" -> X is a substrate)  weakest
`validate.py` measures how often these rules agree with a hand-checked list; do not treat them as ground truth.
"""
from __future__ import annotations

import re

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")

ENZ = (r"(?:CYP\s?-?\d[A-Z]\d{0,2}|cytochrome\s+P-?450\s+(?:isoenzymes?\s+|isozymes?\s+|enzymes?\s+)?\d[A-Z]\d{0,2}"
       r"|UGT\s?\d[A-Z]\d{0,2}|P-?\s?glycoprotein|P-?gp|BCRP|OATP\s?\d[A-Z]\d|MAO-?\s?[AB]|xanthine\s+oxidase)")
ENZ_RE = re.compile(ENZ, re.I)
ENZLIST = ENZ + r"(?:\s*(?:,\s*(?:and\s+|or\s+)?|\s+and\s+|\s+or\s+)" + ENZ + r")*"      # "CYP1A2, CYP3A4 and CYP2C9"


def _enzs(group: str) -> list[str]:
    return list(dict.fromkeys(norm_enzyme(t) for t in ENZ_RE.findall(group)))
TRANSPORTERS = {"P-gp", "BCRP", "OATP1B1", "OATP1B3", "OATP2B1"}


def norm_enzyme(tok: str) -> str:
    t = re.sub(r"[\s\-]+", "", tok.upper())
    t = t.replace("CYTOCHROMEP450", "CYP").replace("ISOENZYMES", "").replace("ISOENZYME", "").replace("ISOZYMES", "")
    t = t.replace("ISOZYME", "").replace("ENZYMES", "").replace("ENZYME", "")
    if t in ("PGLYCOPROTEIN", "PGP"):
        return "P-gp"
    if t == "XANTHINEOXIDASE":
        return "XO"
    if t.startswith("MAO"):
        return "MAO-" + t[-1]
    return t


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(re.sub(r"\s+", " ", text or "")) if len(s.strip()) > 12]


def _subj(names: list[str]) -> str:
    toks = sorted({re.escape(n.lower()) for n in names if n} | {"it", "this drug", "the drug", "this medication",
                                                               "this product", "the product"}, key=len, reverse=True)
    return "(?:" + "|".join(toks) + ")"


_ADJ = r"(?:[\w\-]+\s+){0,3}"
SUBSTRATE_SECTIONS = {"clinical_pharmacology", "pharmacokinetics", "description", "mechanism_of_action"}


def extract_roles(names: list[str], sections: dict, classes: dict | None = None) -> list[dict]:
    """-> [{"enzyme","role","confidence","sentence","section"}] (deduplicated, structured first)."""
    out, seen = [], set()

    def add(enz, role, conf, sent, sec):
        k = (enz, role)
        if k in seen:
            return
        seen.add(k)
        out.append({"enzyme": enz, "role": role, "confidence": conf, "sentence": sent, "section": sec})

    # 1) structured classes
    for cls in (classes or {}).get("pharm_class_moa", []):
        m = re.match(r"(.+?)\s+(Inhibitors?|Inducers?)\b", cls, re.I)
        if m and ENZ_RE.search(m.group(1)):
            add(norm_enzyme(ENZ_RE.search(m.group(1)).group(0)), "inhibitor" if m.group(2).lower().startswith("inhib") else "inducer",
                "structured", cls, "openfda.pharm_class_moa")
    subj = _subj(names)
    # 2) sentences
    for sec, text in sections.items():
        for s in sentences(text):
            if not ENZ_RE.search(s):
                continue
            low = s.lower()
            # explicit perpetrator:  "<drug> is a potent inhibitor of CYP3A4 and CYP2C9" / "<drug> inhibits CYP2D6"
            hit = False
            for m in re.finditer(rf"\b{subj}\s+(?:is|are|was|were|acts?\s+as)\s+(?:an?\s+)?{_ADJ}(inhibitor|inducer)s?\s+of\s+(?:the\s+)?(?:activity\s+of\s+)?({ENZLIST})", s, re.I):
                for e in _enzs(m.group(2)):
                    add(e, "inhibitor" if m.group(1).lower().startswith("inhib") else "inducer", "explicit", s, sec)
                    hit = True
            for m in re.finditer(rf"\b{subj}\s+(?:\w+\s+){{0,3}}?(inhibits|induces)\s+(?:the\s+)?(?:activity\s+of\s+)?({ENZLIST})", s, re.I):
                for e in _enzs(m.group(2)):
                    add(e, "inhibitor" if m.group(1).lower().startswith("inhib") else "inducer", "explicit", s, sec)
                    hit = True
            # a second, coordinated clause: "... and a moderate inhibitor of CYP2C19"
            if hit:
                for m in re.finditer(rf"(?:and|,)\s+(?:an?\s+)?{_ADJ}(inhibitor|inducer)s?\s+of\s+(?:the\s+)?({ENZLIST})", s, re.I):
                    for e in _enzs(m.group(2)):
                        add(e, "inhibitor" if m.group(1).lower().startswith("inhib") else "inducer", "explicit", s, sec)
                continue
            # explicit substrate:  "<drug> is metabolized by CYP2C9" / "is a substrate of CYP3A4"
            m = re.search(rf"\b{subj}\s+(?:is|are|was|were)\s+(?:\w+\s+){{0,4}}?(?:metaboli[sz]ed|oxidi[sz]ed|biotransformed|hydroxylated|demethylated|cleared)\s+"
                          rf"(?:\w+\s+){{0,2}}?(?:by|via|through)\s+(?:the\s+)?(?:\w+\s+){{0,3}}?({ENZLIST})", s, re.I) \
                or re.search(rf"\b{subj}\s+(?:is|are)\s+(?:an?\s+)?(?:\w+\s+){{0,2}}substrate\s+(?:of|for)\s+(?:the\s+)?({ENZLIST})", s, re.I)
            if m:
                for e in _enzs(m.group(1)):
                    add(e, "substrate", "explicit", s, sec)
                continue
            # PK sections talk about this drug: "... is metabolized primarily by CYP2C9"
            if sec in SUBSTRATE_SECTIONS:
                m = re.search(rf"(?:metaboli[sz]ed|oxidi[sz]ed|biotransformed|hydroxylated|demethylated)\s+(?:\w+\s+){{0,2}}?(?:by|via|through)\s+(?:the\s+)?(?:\w+\s+){{0,3}}?({ENZLIST})", s, re.I) \
                    or re.search(rf"substrate\s+(?:of|for)\s+(?:the\s+)?({ENZLIST})", s, re.I)
                if m:
                    for e in _enzs(m.group(1)):
                        add(e, "substrate", "context", s, sec)
                    continue
            # perpetrator wording: "increase(s) plasma concentrations of drugs metabolized by CYP2D6"
            m = re.search(rf"\b(increase|decrease|reduce|raise)s?\s+(?:the\s+)?(?:plasma\s+)?(?:concentrations?|exposure|levels?)\s+of\s+"
                          rf"(?:\w+\s+){{0,3}}?(?:metaboli[sz]ed\s+by|substrates?\s+of)\s+(?:the\s+)?({ENZLIST})", s, re.I)
            if m and re.search(rf"\b{subj}\b", low) and sec in ("drug_interactions", "clinical_pharmacology", "pharmacokinetics", "warnings"):
                for e in _enzs(m.group(2)):
                    add(e, "inhibitor" if m.group(1).lower() in ("increase", "raise") else "inducer", "context", s, sec)
                continue
            # victim wording: "strong CYP3A4 inhibitors increase <drug> exposure" -> this drug is a substrate
            if sec in ("drug_interactions", "warnings", "warnings_and_cautions", "precautions", "contraindications", "boxed_warning"):
                m = re.search(rf"({ENZLIST})\s+(?:inhibitors?|inducers?)|(?:inhibitors?|inducers?)\s+of\s+(?:the\s+)?({ENZLIST})"
                              rf"|(?:drugs?|agents?|medications?|medicines?)\s+that\s+(?:inhibit|induce)\s+(?:the\s+)?({ENZLIST})", s, re.I)
                if m and re.search(r"(increas|decreas|reduc|rais|alter|elevat|lower|higher|contraindicat|avoid)", low):
                    for e in _enzs(m.group(1) or m.group(2) or m.group(3)):
                        add(e, "substrate", "context", s, sec)
    order = {"structured": 0, "explicit": 1, "context": 2}
    return sorted(out, key=lambda r: order[r["confidence"]])


# ------------------------------------------------------------------ safety "axes"
EFFECT_AXES = {
    "QT prolongation": r"QT[c]?[\s-]*(?:interval\s+)?prolong|prolong\w*\s+(?:of\s+)?(?:the\s+)?QT|torsade|torsades",
    "Serotonin toxicity": r"serotonin\s+syndrome|serotonin\s+toxicity|serotonergic",
    "CNS / respiratory depression": r"respiratory\s+depress|CNS\s+depress|profound\s+sedation|concomitant\s+use\s+of\s+benzodiazepines",
    "Bleeding": r"bleed|h[a]?emorrhag|anticoagulant|antiplatelet|prothrombin|international\s+normali[sz]ed\s+ratio|\bINR\b",
    "Hyperkalaemia": r"hyperkal[a]?emia|potassium[\s-]sparing|potassium\s+supplement",
    "Hypotension": r"hypotens|orthostatic",
    "Nephrotoxicity": r"nephrotox|acute\s+kidney\s+injury|renal\s+(?:failure|toxicity)",
    "Hepatotoxicity": r"hepatotox|liver\s+(?:injury|failure|damage)|hepatic\s+(?:failure|necrosis)",
    "Myelosuppression": r"myelosuppress|pancytopenia|bone\s+marrow\s+(?:suppress|depress)",
    "Hypoglycaemia": r"hypoglyc[a]?emia",
    "Myopathy / rhabdomyolysis": r"myopath|rhabdomyol",
    "Anticholinergic effects": r"anticholinergic|urinary\s+retention",
    "Seizure threshold": r"seizure\s+threshold|convulsion",
    "Bradycardia / AV block": r"bradycard|AV\s+block|atrioventricular\s+block|heart\s+block",
}
_AXIS_RE = {k: re.compile(v, re.I) for k, v in EFFECT_AXES.items()}
AXIS_SECTIONS_STRONG = ("boxed_warning", "warnings", "warnings_and_cautions", "drug_interactions", "contraindications",
                        "pharmacodynamics", "precautions")


def effect_axes(sections: dict) -> dict[str, list[tuple[str, str]]]:
    """axis -> [(section, sentence)] for the sections where a safety statement is deliberate (not every AE list)."""
    out: dict[str, list] = {}
    for sec in AXIS_SECTIONS_STRONG:
        for s in sentences(sections.get(sec, "")):
            for axis, rx in _AXIS_RE.items():
                if rx.search(s) and len(out.setdefault(axis, [])) < 3:
                    out[axis].append((sec, s[:400]))
    return out


def reaction_axes(reaction: str) -> list[str]:
    """Which safety axes does a MedDRA reaction term belong to? (e.g. 'TORSADE DE POINTES' -> QT prolongation)"""
    extra = {"Hyperkalaemia": r"hyperkal", "Hypoglycaemia": r"hypoglyc", "Bleeding": r"haemorrhag|hemorrhag|bleed|haematoma|international normali[sz]ed ratio",
             "Myopathy / rhabdomyolysis": r"rhabdo|myopath|blood creatine phosphokinase", "Hepatotoxicity": r"hepat|liver|transaminase|alanine amino|aspartate amino",
             "Nephrotoxicity": r"renal|kidney|creatinine", "Myelosuppression": r"neutropeni|pancytopeni|thrombocytopeni|leukopeni|agranulocyt|bone marrow",
             "Serotonin toxicity": r"serotonin", "CNS / respiratory depression": r"respiratory (?:depression|arrest|failure)|somnolence|sedation|coma|loss of consciousness",
             "QT prolongation": r"qt|torsade|ventricular (?:tachycardia|fibrillation|arrhythmia)", "Hypotension": r"hypotens|syncope",
             "Bradycardia / AV block": r"bradycard|atrioventricular|heart block", "Seizure threshold": r"seizure|convuls"}
    return [a for a, rx in extra.items() if re.search(rx, reaction, re.I)] or \
           [a for a, rx in _AXIS_RE.items() if rx.search(reaction)]


# ------------------------------------------------------------------ naming / mentions
CLASS_ALIASES = {
    "nonsteroidal anti-inflammatory drug": ["nsaid", "nsaids"],
    "angiotensin converting enzyme inhibitor": ["ace inhibitor", "ace inhibitors", "angiotensin-converting enzyme inhibitor"],
    "angiotensin 2 receptor antagonist": ["angiotensin receptor blocker", "arb", "arbs"],
    "serotonin reuptake inhibitor": ["ssri", "ssris", "selective serotonin reuptake inhibitor"],
    "monoamine oxidase inhibitor": ["maoi", "maois", "monoamine oxidase inhibitor"],
    "hmg-coa reductase inhibitor": ["statin", "statins"],
    "proton pump inhibitor": ["ppi", "ppis", "proton pump inhibitor"],
    "benzodiazepine": ["benzodiazepines"],
    "opioid agonist": ["opioid", "opioids"],
    "azole antifungal": ["azole antifungals", "azole"],
    "macrolide antimicrobial": ["macrolide", "macrolides"],
}


def name_variants(generic: str | None, brands: list[str] | None, classes: dict | None, query: str) -> list[str]:
    """Lower-case strings that mean 'this drug (or its class)' when found in another drug's label."""
    vs = {query.lower().strip()}
    for g in (generic or "").lower().replace(",", " ").split(" and "):
        g = g.strip()
        if g:
            vs.add(g)
            vs.add(re.sub(r"\s+(sodium|hydrochloride|hcl|sulfate|tartrate|maleate|succinate|mesylate|besylate|potassium|calcium)$", "", g))
    for b in brands or []:
        vs.add(b.lower().strip())
    for c in (classes or {}).get("pharm_class_epc", []):
        c = re.sub(r"\s*\[.*?\]", "", c).lower().strip()
        if c:
            vs.add(c)
            vs.update(CLASS_ALIASES.get(c, []))
    return sorted(v for v in vs if len(v) >= 4)


def find_mentions(text: str, terms: list[str], limit: int = 3) -> list[str]:
    """Sentences of `text` that name any term (whole-word match, so 'ace' does not match 'place')."""
    if not terms:
        return []
    rx = re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")(?![a-z0-9])", re.I)
    return [s[:500] for s in sentences(text) if rx.search(s)][:limit]


NEGATION = re.compile(
    r"\bno\s+(?:clinically\s+)?(?:significant|relevant|meaningful|apparent|pharmacokinetic|pharmacodynamic)?\s*(?:effect|interaction|change|difference|alteration|influence)"
    r"|\bdid\s+not\s+(?:significantly\s+)?(?:alter|affect|change|influence|increase|decrease|modify|inhibit)"
    r"|\b(?:was|were|is|are)\s+not\s+(?:significantly\s+)?(?:affected|altered|changed|associated|increased|decreased)"
    r"|\bwithout\s+(?:any\s+)?(?:clinically\s+)?significant|\bnot\s+clinically\s+(?:significant|relevant)|\bno\s+evidence\s+of\s+(?:an?\s+)?interaction"
    r"|\bdoes\s+not\s+(?:significantly\s+)?(?:inhibit|induce|affect|alter|interact)", re.I)


def is_negated(sentence: str) -> bool:
    return bool(NEGATION.search(sentence))
