"""A small evidence-carrying knowledge graph built from the drug profiles and the FAERS statistics.

Node types: drug, gene, pathway, enzyme, axis (safety axis), reaction, class, combo (a hyperedge node joining the drugs
of a combination to a reaction - the 3-drug signals are genuinely n-ary, not pairwise).
Every edge keeps its relation, a confidence, and the evidence ids behind it, so any path can be audited edge by edge.

Missing is not the same as false: an absent edge only means *no source recorded it*. `predict_links` therefore proposes
candidates and `discover.py` then checks each one against labels and literature before calling it novel.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .mathx import evidence as em
from .mathx import graphmath as gm

CONF = {"structured": 0.9, "explicit": 0.8, "context": 0.5, "database": 0.8, "faers": 0.6, "label": 0.7, "pathway": 0.6}


class KnowledgeGraph:
    def __init__(self):
        self.G = nx.Graph()

    # ------------------------------------------------------------------ building
    def _edge(self, u, v, relation, conf, evidence=(), **attrs):
        if self.G.has_edge(u, v):
            d = self.G[u][v]
            d["relations"] = sorted(set(d["relations"]) | {relation})
            d["evidence"] = sorted(set(d["evidence"]) | set(evidence))
            d["conf"] = max(d["conf"], conf)
        else:
            self.G.add_edge(u, v, relations=[relation], conf=conf, evidence=sorted(set(evidence)), weight=1.0, **attrs)

    def add_profile(self, p):
        d = f"drug:{p.name.upper()}"
        self.G.add_node(d, kind="drug", label=p.name)
        for t in p.targets:
            for g in t.genes or []:
                self.G.add_node(f"gene:{g}", kind="gene", label=g)
                self._edge(d, f"gene:{g}", f"targets ({t.action_type or 'acts on'})", CONF["database"], t.evidence)
                for pw in t.pathways:
                    self.G.add_node(f"pathway:{pw['id']}", kind="pathway", label=pw["name"])
                    self._edge(f"gene:{g}", f"pathway:{pw['id']}", "in pathway", CONF["pathway"])
        for r in p.roles:
            self.G.add_node(f"enzyme:{r.enzyme}", kind="enzyme", label=r.enzyme)
            self._edge(d, f"enzyme:{r.enzyme}", r.role, CONF[r.confidence], [r.evidence])
        for axis, ids in p.axes.items():
            self.G.add_node(f"axis:{axis}", kind="axis", label=axis)
            self._edge(d, f"axis:{axis}", "label warns", CONF["label"], ids)
        for c in (p.classes or {}).get("pharm_class_epc", []):
            self.G.add_node(f"class:{c}", kind="class", label=c)
            self._edge(d, f"class:{c}", "member of", CONF["label"])

    def add_faers(self, stats: pd.DataFrame, min_events: int = 3):
        """stats: pharmasense_core.drug_reaction_stats output. Only statistically clear signals become edges."""
        s = stats[(stats["signal"]) & (stats["a"] >= min_events)]
        for r in s.itertuples():
            d, x = f"drug:{r.drug}", f"reaction:{r.reaction}"
            self.G.add_node(d, kind="drug", label=r.drug.title())
            self.G.add_node(x, kind="reaction", label=r.reaction.capitalize())
            self._edge(d, x, "FAERS signal", CONF["faers"], [], ror=float(r.ror), n=int(r.a))

    def add_combo(self, drugs, reaction, verdict, ror):
        node = "combo:" + "|".join(sorted(d.upper() for d in drugs))
        self.G.add_node(node, kind="combo", label=" + ".join(drugs), verdict=verdict)
        for d in drugs:
            self._edge(node, f"drug:{d.upper()}", "member", 1.0)
        self.G.add_node(f"reaction:{reaction}", kind="reaction", label=reaction.capitalize())
        self._edge(node, f"reaction:{reaction}", f"FAERS {verdict}", CONF["faers"], [], ror=float(ror))

    # ------------------------------------------------------------------ explaining
    def paths(self, src: str, dst: str, max_hops: int = 4, limit: int = 8) -> list[dict]:
        """Typed simple paths, best first. Score = product of edge confidences x 0.9^(hops-1): short, well-sourced beats long."""
        out = []
        if src not in self.G or dst not in self.G:
            return out
        for path in nx.all_simple_paths(self.G, src, dst, cutoff=max_hops):
            edges, score = [], 1.0
            for u, v in zip(path, path[1:]):
                d = self.G[u][v]
                edges.append({"from": self.G.nodes[u].get("label", u), "to": self.G.nodes[v].get("label", v), "relation": "/".join(d["relations"]),
                              "conf": d["conf"], "evidence": d["evidence"]})
                score *= d["conf"]
            out.append({"nodes": [self.G.nodes[n].get("label", n) for n in path], "edges": edges, "score": score * 0.9 ** (len(path) - 2)})
        return sorted(out, key=lambda p: -p["score"])[:limit]

    def path_support(self, paths: list[dict]) -> float:
        """Probability that at least one path is real, treating paths as independent (noisy-OR, an assumption - paths that
        share edges are not independent, so this is an upper bound)."""
        return em.noisy_or([min(p["score"], 0.99) for p in paths])

    # ------------------------------------------------------------------ discovering
    def predict_links(self, drug: str, kind: str = "reaction", top_k: int = 10, alpha: float = 0.15) -> list[dict]:
        """Rank nodes of `kind` that the drug is NOT yet linked to by personalised PageRank from the drug node."""
        src = f"drug:{drug.upper()}"
        if src not in self.G:
            return []
        A, nodes = gm.adjacency(self.G)
        idx = {n: i for i, n in enumerate(nodes)}
        r = gm.personalized_pagerank(A, {idx[src]: 1.0}, alpha)
        cands = []
        for n in nodes:
            if self.G.nodes[n].get("kind") == kind and not self.G.has_edge(src, n):
                cands.append((float(r[idx[n]]), n))
        out = []
        for score, n in sorted(cands, reverse=True)[:top_k]:
            if score <= 0:
                continue
            ps = self.paths(src, n, max_hops=4, limit=3)
            out.append({"node": self.G.nodes[n].get("label", n), "ppr": score, "paths": ps})
        return out

    def holdout_auc(self, method: str = "ppr", frac: float = 0.2, seed: int = 0) -> float:
        """Can this graph recover its own hidden edges better than chance? (0.5 = no.)"""
        return gm.holdout_auc(self.G, method, frac, seed)

    def stats(self) -> dict:
        kinds = pd.Series([d.get("kind") for _, d in self.G.nodes(data=True)]).value_counts().to_dict()
        return {"nodes": self.G.number_of_nodes(), "edges": self.G.number_of_edges(), "by_kind": kinds}
