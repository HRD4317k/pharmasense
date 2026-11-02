"""State passed between the LangGraph nodes."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class State(TypedDict, total=False):
    query: dict                                   # {"drugs": [...], "reaction": str | None, "faers": dict | None}
    profiles: dict                                # drug -> DrugProfile
    string_edges: list                            # protein-network links between the drugs' targets
    hypotheses: list                              # list[Hypothesis]
    lit_counts: dict                              # (drugA, drugB) -> abstracts retrieved
    report: dict
    loops: int                                    # how many extra literature passes were made
    warnings: Annotated[list, operator.add]       # accumulates across nodes
    trace: Annotated[list, operator.add]          # one line per step: what ran and what it found
