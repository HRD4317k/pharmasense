"""One object that holds every client, so the rest of the code takes a single `services` argument
(and tests can hand in fakes)."""
from __future__ import annotations

from dataclasses import dataclass

from .clients.chembl import ChEMBL
from .clients.europepmc import EuropePMC
from .clients.openfda import OpenFDA
from .clients.opentargets import OpenTargets
from .clients.others import ClinicalTrials, StringDB
from .clients.pubchem import PubChem
from .clients.uniprot import UniProt
from .config import Settings
from .http import Http
from .llm import LLM


@dataclass
class Services:
    settings: Settings
    http: Http
    pubchem: PubChem
    chembl: ChEMBL
    uniprot: UniProt
    opentargets: OpenTargets
    europepmc: EuropePMC
    openfda: OpenFDA
    string: StringDB
    ctgov: ClinicalTrials
    llm: LLM

    @classmethod
    def build(cls, settings: Settings | None = None, http: Http | None = None) -> "Services":
        settings = settings or Settings.load()
        http = http or Http(settings)
        return cls(settings, http, PubChem(http), ChEMBL(http), UniProt(http), OpenTargets(http), EuropePMC(http),
                   OpenFDA(http, settings.openfda_api_key), StringDB(http), ClinicalTrials(http), LLM(settings))
