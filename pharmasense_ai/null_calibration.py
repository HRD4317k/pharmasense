"""Permutation null for the triplet pipeline: how many "signals" does it report when there is NO signal?

Method. Keep every report's drug list; shuffle which reaction list goes with which report. That destroys every
drug<->reaction association but keeps (a) how often each reaction occurs and (b) which drugs travel together. Run the
SAME evaluation + verdict code on the shuffled data, B times. Anything it still calls a signal is, by construction,
a false alarm of the pipeline. Then

        empirical FDR = mean(false alarms per shuffle) / (signals in the real data)

This tests the whole chain (exact test, BH correction, verdict rules) rather than trusting the theory.
Limitation: shuffling whole reaction lists cannot create the *clustering* that real duplicate reports have, so the
null is optimistic about clustered data - pair this with `cluster_check`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import pharmasense_core as core


def permutation_null(reports: pd.DataFrame, observed: pd.DataFrame, n_perm: int = 20, min_events: int = 3,
                     seed: int = 0, log=print) -> dict:
    X, Y, drugs, rxs = core.build_matrices(reports)
    d_idx = {d: i for i, d in enumerate(drugs)}
    triplets = sorted({tuple(sorted((d_idx[r.drug_a], d_idx[r.drug_b], d_idx[r.drug_c]))) for r in observed.itertuples()})
    rng = np.random.default_rng(seed)
    Ycsr = Y.tocsr()
    rows = []
    for b in range(n_perm):
        Yp = Ycsr[rng.permutation(Ycsr.shape[0])].tocsc()
        res = core.evaluate_triplets(X, Yp, drugs, rxs, triplets, min_events, log=lambda *_: None)
        if len(res):
            res = core.classify_triplets(res, min_events)
        sig = res[res["verdict"] != "No signal"] if len(res) else res
        rows.append({"perm": b, "tested": int(len(res)), "signals": int(len(sig)),
                     "combination_effects": int((sig["verdict"] == "Combination effect").sum()) if len(sig) else 0,
                     "probable_or_combination": int(sig["verdict"].isin(["Combination effect", "Probable combination"]).sum()) if len(sig) else 0})
        if (b + 1) % 5 == 0:
            log(f"  permutation {b + 1}/{n_perm}")
    null = pd.DataFrame(rows)
    obs_sig = int((observed["verdict"] != "No signal").sum())
    obs_comb = int(observed["verdict"].isin(["Combination effect", "Probable combination"]).sum())
    return {
        "null": null, "observed_signals": obs_sig, "observed_combination_like": obs_comb,
        "null_mean_signals": float(null["signals"].mean()), "null_mean_combination_like": float(null["probable_or_combination"].mean()),
        "empirical_fdr_signals": float(null["signals"].mean() / obs_sig) if obs_sig else float("nan"),
        "empirical_fdr_combination_like": float(null["probable_or_combination"].mean() / obs_comb) if obs_comb else float("nan"),
    }
