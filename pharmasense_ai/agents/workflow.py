"""The LangGraph workflow.

    profile -> network -> hypothesize -> retrieve -> challenge -> score --(untested hypotheses & loops left)--> bump -> retrieve
                                                                      \\--(otherwise)--------------------------> critic -> report

`bump -> retrieve` is the feedback edge: when some hypothesis ends up untested for lack of literature, the graph searches
again with that hypothesis' specific enzyme / axis terms before it gives up and reports the gap.
"""
from __future__ import annotations

import json
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from ..config import ROOT
from ..mathx.uncertainty import SplitConformal
from ..models import EvidenceStore
from ..services import Services
from .nodes import make_nodes
from .state import State

CALIBRATION_FILE = ROOT / "data" / "benchmark" / "calibration.json"


def load_calibration():
    """-> (evidence weight table | None, SplitConformal | None) written by `python -m pharmasense_ai validate --calibrate`."""
    try:
        d = json.loads(Path(CALIBRATION_FILE).read_text())
    except Exception:
        return None, None
    table = {**d.get("log_lr", {}), "_calibrated": True} if d.get("log_lr") else None
    cf = SplitConformal.from_dict(d["conformal"]) if d.get("conformal") else None
    return table, cf


def build_graph(svc: Services, store: EvidenceStore | None = None, table=None, conformal=None, max_loops: int = 1,
                cache: dict | None = None):
    store = store if store is not None else EvidenceStore()
    n = make_nodes(svc, store, table, conformal, max_loops, cache)
    g = StateGraph(State)
    for name in ("profile", "network", "hypothesize", "retrieve", "challenge", "score", "critic", "bump", "report"):
        g.add_node(name, n[name])
    g.add_edge(START, "profile")
    g.add_edge("profile", "network")
    g.add_edge("network", "hypothesize")
    g.add_edge("hypothesize", "retrieve")
    g.add_edge("retrieve", "challenge")
    g.add_edge("challenge", "score")
    g.add_conditional_edges("score", n["route"], {"retrieve": "bump", "critic": "critic"})
    g.add_edge("critic", "report")
    g.add_edge("bump", "retrieve")
    g.add_edge("report", END)
    return g.compile(), store


def explain(drugs: list[str], reaction: str | None = None, faers: dict | None = None, svc: Services | None = None,
            use_calibration: bool = True) -> dict:
    """Run the full workflow and return the report dict."""
    svc = svc or Services.build()
    table, cf = load_calibration() if use_calibration else (None, None)
    app, _ = build_graph(svc, table=table, conformal=cf)
    out = app.invoke({"query": {"drugs": drugs, "reaction": reaction, "faers": faers}, "loops": 0, "warnings": [], "trace": []})
    return out["report"]
