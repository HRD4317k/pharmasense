"""Retrieval over label sections and literature abstracts (BM25, no embeddings, no API key).

Drug names, enzyme names and numbers matter more than paraphrase here, which is where lexical BM25 is strong; a dense
retriever can be fused in later through `Retriever.search`'s interface. Every chunk keeps its source and reference,
so whatever is retrieved can be cited.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .label_parse import sentences

_TOK = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
STOP = set("a an the of and or to in on for with by is are was were be been this that these those as at from it its may can will "
           "not no if than then also such into which who whom whose their there has have had do does did".split())


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOK.findall((text or "").lower()):
        if t in STOP:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    ref: str
    section: str = ""
    drugs: list = field(default_factory=list)
    kind: str = "label_text"
    group: str = ""
    meta: dict = field(default_factory=dict)


def chunk_text(text: str, size: int = 3, overlap: int = 1) -> list[str]:
    ss = sentences(text)
    if not ss:
        return []
    step = max(1, size - overlap)
    return [" ".join(ss[i:i + size]) for i in range(0, max(1, len(ss) - overlap), step)]


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf = [Counter(tokenize(c.text)) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df = Counter(w for t in self.tf for w in t)
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def search(self, query: str, k: int = 5, where=None) -> list[tuple[float, Chunk]]:
        q = tokenize(query)
        scored = []
        for i, c in enumerate(self.chunks):
            if where and not where(c):
                continue
            tf, dl = self.tf[i], self.len[i]
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avg or 1)))
            if s > 0:
                scored.append((s, c))
        return sorted(scored, key=lambda x: -x[0])[:k]


def label_chunks(drug: str, label: dict) -> list[Chunk]:
    out = []
    for sec, text in (label.get("sections") or {}).items():
        for j, t in enumerate(chunk_text(text)):
            out.append(Chunk(id=f"{drug}:{sec}:{j}", text=t, source=f"openFDA label: {drug}", ref=label.get("url", ""),
                             section=sec, drugs=[drug], kind="label_text", group=f"label:{drug}"))
    return out


def literature_chunks(papers: list[dict], drugs: list[str]) -> list[Chunk]:
    out = []
    for p in papers:
        for j, t in enumerate(chunk_text(f"{p.get('title', '')}. {p.get('abstract', '')}", size=2, overlap=1)):
            out.append(Chunk(id=f"PMID{p.get('pmid')}:{j}", text=t, source=f"PubMed {p.get('pmid')} ({p.get('year')}) {p.get('journal') or ''}".strip(),
                             ref=p.get("url", ""), section="abstract", drugs=list(drugs), kind="literature",
                             group=f"pmid:{p.get('pmid')}", meta={"year": p.get("year"), "title": p.get("title")}))
    return out
