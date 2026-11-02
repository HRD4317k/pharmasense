"""Plain data objects shared by the whole package."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field


@dataclass
class Evidence:
    """One citable fact. Every claim in a report points at one or more of these by id."""
    id: str
    kind: str            # label_class | label_role | label_context | label_ddi | label_text | database | literature | faers | network
    source: str          # human-readable source, e.g. "openFDA label: warfarin"
    ref: str             # URL or identifier a person can open
    text: str            # the sentence / record, verbatim or a short factual rendering
    drugs: list = field(default_factory=list)
    section: str = ""
    group: str = ""      # evidence from the same document is NOT independent: the scorer discounts by group
    meta: dict = field(default_factory=dict)


class EvidenceStore:
    def __init__(self):
        self.items: dict[str, Evidence] = {}
        self._seen: dict[str, str] = {}

    def add(self, kind, source, ref, text, drugs=(), section="", group="", **meta) -> str:
        key = hashlib.sha1(f"{kind}|{source}|{text}".encode()).hexdigest()
        if key in self._seen:
            return self._seen[key]
        eid = f"E{len(self.items) + 1:03d}"
        self.items[eid] = Evidence(eid, kind, source, ref, text, list(drugs), section, group or source, meta)
        self._seen[key] = eid
        return eid

    def get(self, eid) -> Evidence | None:
        return self.items.get(eid)

    def __len__(self):
        return len(self.items)


@dataclass
class Target:
    name: str
    chembl_id: str = ""
    action_type: str = ""
    mechanism: str = ""
    uniprot: list = field(default_factory=list)
    genes: list = field(default_factory=list)
    pathways: list = field(default_factory=list)       # [{"id","name"}]
    evidence: list = field(default_factory=list)


@dataclass
class EnzymeRole:
    enzyme: str              # CYP3A4, P-gp, ...
    role: str                # substrate | inhibitor | inducer
    confidence: str          # structured | explicit | context
    evidence: str            # evidence id
    sentence: str = ""


@dataclass
class DrugProfile:
    name: str
    smiles: str = ""
    structure: dict = field(default_factory=dict)       # PubChem properties
    chembl_id: str = ""
    max_phase: object = None
    targets: list = field(default_factory=list)         # list[Target]
    roles: list = field(default_factory=list)           # list[EnzymeRole]
    axes: dict = field(default_factory=dict)            # effect axis -> [evidence id]
    classes: dict = field(default_factory=dict)         # pharm_class_* from the label
    names: list = field(default_factory=list)           # name variants used for text matching
    label: dict = field(default_factory=dict)           # {"url","set_id","sections":{...}} (sections kept for RAG)
    ot_adverse_events: list = field(default_factory=list)
    trials: list = field(default_factory=list)
    indications: list = field(default_factory=list)
    gaps: list = field(default_factory=list)            # what could NOT be found, and why
    sources: list = field(default_factory=list)         # sources that did answer

    def roles_for(self, enzyme: str, role: str):
        return [r for r in self.roles if r.enzyme == enzyme and r.role == role]

    def to_dict(self):
        return asdict(self)


@dataclass
class Hypothesis:
    id: str
    mechanism_type: str      # PK-enzyme | PK-transporter | PD-shared-target | PD-effect-axis | PD-pathway | indirect-network | documented
    title: str
    claim: str
    drugs: list
    victim: str = ""
    perpetrator: str = ""
    term: str = ""           # enzyme / axis / target the mechanism turns on
    direction: str = ""      # e.g. "exposure of warfarin increases"
    supporting: list = field(default_factory=list)       # evidence ids
    contradicting: list = field(default_factory=list)
    gaps: list = field(default_factory=list)
    pre_gaps: list = field(default_factory=list)          # gaps known before finalisation (kept across loop passes)
    alternatives: list = field(default_factory=list)
    explains_reaction: object = None                      # True / False / None (unknown or no reaction given)
    reaction_evidence: list = field(default_factory=list)  # label sentences linking the victim drug to the reaction
    origin: str = "rules"                                  # rules | llm-proposed
    prior: float = 0.1
    log_odds: float = 0.0
    posterior: float = 0.0
    interval: tuple = (0.0, 1.0)
    verdict: str = "untested"       # supported | contested | untested | unsupported
    conformal: dict = field(default_factory=dict)
