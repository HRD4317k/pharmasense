"""Graph mathematics for DISCOVER: score candidate edges that are not recorded.

 * personalised PageRank   how strongly is v connected to u through *many short paths*?
 * spectral embedding       nodes close in the leading eigenvectors of D^-1/2 A D^-1/2 share neighbourhoods
 * Adamic-Adar              common neighbours, weighted down when the neighbour is a hub (a generic pathway)
 * hold-out AUROC           hide a fraction of edges, then ask whether the scorer ranks them above non-edges

A scorer is only useful if it beats chance on hidden edges of *your* graph; `holdout_auc` is how to check.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.metrics import roc_auc_score


def adjacency(G: nx.Graph, nodes=None):
    nodes = list(nodes if nodes is not None else G.nodes())
    return nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", format="csr").astype(float), nodes


def personalized_pagerank(A, seeds: dict[int, float], alpha: float = 0.15, tol: float = 1e-10, max_iter: int = 500):
    """r = alpha*s + (1-alpha) * P r, with P the column-stochastic transition matrix. Dangling mass returns to the seed."""
    n = A.shape[0]
    s = np.zeros(n)
    for i, w in seeds.items():
        s[i] = w
    s /= s.sum()
    deg = np.asarray(A.sum(axis=0)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    P = A @ sparse.diags(inv)
    dangling = (deg == 0).astype(float)
    r = s.copy()
    for _ in range(max_iter):
        new = alpha * s + (1 - alpha) * (P @ r + dangling @ r * s)
        if np.abs(new - r).sum() < tol:
            r = new
            break
        r = new
    return r


def spectral_embedding(A, dim: int = 8) -> np.ndarray:
    """Row-normalised leading non-trivial eigenvectors of the symmetric normalised adjacency."""
    n = A.shape[0]
    dim = max(1, min(dim, n - 2))
    deg = np.asarray(A.sum(axis=1)).ravel()
    d = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0)
    M = sparse.diags(d) @ A @ sparse.diags(d)
    if n <= 600:
        w, v = np.linalg.eigh(M.toarray())
        order = np.argsort(-w)
        V = v[:, order[1:dim + 1]]
    else:
        w, v = eigsh(M, k=dim + 1, which="LA")
        V = v[:, np.argsort(-w)[1:dim + 1]]
    norm = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.where(norm == 0, 1, norm)


def adamic_adar_matrix(A) -> np.ndarray:
    deg = np.asarray(A.sum(axis=1)).ravel()
    w = np.divide(1.0, np.log(np.maximum(deg, 2)), out=np.zeros_like(deg), where=deg > 1)
    B = (A > 0).astype(float)
    return (B @ sparse.diags(w) @ B).toarray()


def score_pairs(A, pairs, method: str = "ppr", dim: int = 8):
    if method == "aa":
        S = adamic_adar_matrix(A)
        return np.array([S[u, v] for u, v in pairs])
    if method == "spectral":
        E = spectral_embedding(A, dim)
        return np.array([float(E[u] @ E[v]) for u, v in pairs])
    if method == "ppr":
        cache, out = {}, []
        for u, v in pairs:
            for a in (u, v):
                if a not in cache:
                    cache[a] = personalized_pagerank(A, {a: 1.0})
            out.append(0.5 * (cache[u][v] + cache[v][u]))
        return np.array(out)
    raise ValueError(method)


def holdout_auc(G: nx.Graph, method: str = "ppr", frac: float = 0.2, seed: int = 0, dim: int = 8) -> float:
    """Remove `frac` of the edges (keeping the graph's nodes), score removed edges vs. an equal number of random
    non-edges, return AUROC. 0.5 = no better than chance."""
    rng = np.random.default_rng(seed)
    edges = list(G.edges())
    if len(edges) < 10:
        return float("nan")
    k = max(1, int(frac * len(edges)))
    pick = rng.choice(len(edges), size=k, replace=False)
    removed = [edges[i] for i in pick]
    H = G.copy()
    H.remove_edges_from(removed)
    nodes = list(H.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    A, _ = adjacency(H, nodes)
    nonedges, tries = [], 0
    while len(nonedges) < k and tries < 50 * k:
        u, v = rng.choice(len(nodes), 2, replace=False)
        tries += 1
        if not G.has_edge(nodes[u], nodes[v]):
            nonedges.append((nodes[u], nodes[v]))
    pos = [(idx[u], idx[v]) for u, v in removed]
    neg = [(idx[u], idx[v]) for u, v in nonedges]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = score_pairs(A, pos + neg, method, dim)
    return float(roc_auc_score(y, s))
