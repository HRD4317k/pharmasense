"""Whole-FAERS counts for any drug set + reaction, straight from openFDA, without downloading reports.

The local pipeline analyses a sample (a few thousand reports). openFDA can *count* over the entire database, so for
one chosen combination we can rebuild the same eight exclusive groups (none / A only / ... / all three) from
"contains at least these drugs" counts and Moebius inversion:

        exact(T) = sum over S superset-of T of (-1)^(|S|-|T|) * atleast(S)

openFDA has no NOT operator, which is why inclusion-exclusion is used. Cost: 2^k - 1 drug-set queries x 2 (with and
without the reaction) + 2 totals = 16 calls for a triplet; cached after the first run.

Caveats (also shown in the app): matching uses openFDA's harmonised `generic_name`, so reports whose product name
could not be mapped are missed (counts are lower bounds); duplicates are not removed; all roles (suspect and
concomitant) are included.
"""
from __future__ import annotations

from itertools import combinations

CATS3 = ["REF", "A", "B", "AB", "C", "AC", "BC", "ABC"]        # index == bitmask (A=1, B=2, C=4)
CATS2 = ["REF", "A", "B", "AB"]


def mobius_exclusive(atleast: dict[int, int], k: int) -> dict[int, int]:
    """atleast[mask] = reports containing every drug in mask (atleast[0] = all reports).
    Returns exact[mask] = reports containing exactly the drugs in mask."""
    exact = {}
    for t in range(2 ** k):
        total = 0
        for s in range(2 ** k):
            if s & t == t:
                total += (-1) ** (bin(s).count("1") - bin(t).count("1")) * atleast[s]
        exact[t] = total
    return exact


def _drugs_clause(openfda, names, mask):
    return "+AND+".join(openfda.drug_clause(names[i]) for i in range(len(names)) if mask >> i & 1)


def _atleast(openfda, names, reaction):
    k = len(names)
    n_all, a_all = {}, {}
    n_all[0] = openfda.event_total(None)
    rx = f'patient.reaction.reactionmeddrapt.exact:"{reaction.upper()}"' if reaction else None
    a_all[0] = openfda.event_total(rx) if rx else 0
    for mask in range(1, 2 ** k):
        clause = _drugs_clause(openfda, names, mask)
        n_all[mask] = openfda.event_total(clause)
        a_all[mask] = openfda.event_total(f"{clause}+AND+{rx}") if rx else 0
    return n_all, a_all


def live_group_counts(openfda, drugs: list[str], reaction: str | None) -> dict:
    """{cat: (events, n)} in the same layout the rest of the package uses, for 2 or 3 drugs."""
    k = len(drugs)
    if k not in (2, 3):
        raise ValueError("2 or 3 drugs")
    cats = CATS2 if k == 2 else CATS3
    n_all, a_all = _atleast(openfda, drugs, reaction)
    n_ex, a_ex = mobius_exclusive(n_all, k), mobius_exclusive(a_all, k)
    return {cats[m]: (max(a_ex[m], 0), max(n_ex[m], 0)) for m in range(2 ** k)}
