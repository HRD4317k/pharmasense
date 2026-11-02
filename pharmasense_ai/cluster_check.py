"""Are the reports behind a signal more clustered than the reports they were drawn from?

Why this exists. In the shipped sample, the ML target "ILL-DEFINED DISORDER" was carried by a block of opioid-product
reports (104 of its 146 reports contain Percocet; Percocet appears in only 5 of the other 1,862). A batch of
near-identical reports submitted together gives huge odds ratios that say nothing about biology.

Why it is NOT a simple threshold. Our first version flagged "most events share one receive date" - and flagged 79 of
81 signals, because the sample itself contains only 15 distinct receive dates. Likewise, patients on one regimen
(transplant immunosuppressants) legitimately share drug lists. So each diagnostic is compared with the *exposed pool*
(all reports listing the same drugs) by Monte-Carlo:

    p_lists = P( n random exposed reports show <= this many distinct drug lists / n )
    p_dates = P( n random exposed reports show >= this top-date share )

A signal is flagged only if the events are more clustered than a random draw of the same size from the exposed pool
(p < 0.05, uncorrected). This is a prompt for a human look, not a duplicate detector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALPHA = 0.05


def data_quality(reports: pd.DataFrame) -> dict:
    """Sanity checks on how the sample was drawn."""
    d = reports["date"].astype(str)
    months = d.str[:6].nunique()
    n_dates = int(d.nunique())
    top = float(d.value_counts().iloc[0] / len(d)) if len(d) else float("nan")
    warn = ""
    if months and n_dates < 12 * max(months, 1) * 0.5 and len(d) > 200:
        warn = (f"Only {n_dates} distinct receive dates across {months} months (busiest date holds {top:.0%} of reports): "
                "the sample is not spread over the period. Re-fetch with the day-by-day sampler (fetch_fda_api.py).")
    return {"n_reports": int(len(reports)), "distinct_dates": n_dates, "months": int(months),
            "busiest_date_share": top, "warning": warn}


def _stats(lists: pd.Series, dates: pd.Series):
    n = len(lists)
    return lists.nunique() / n, dates.value_counts().iloc[0] / n


def cluster_concentration(reports: pd.DataFrame, drugs: tuple[str, ...], reaction: str, n_mc: int = 2000,
                          seed: int = 0) -> dict:
    want = set(drugs)
    exposed = reports[reports["drugs"].apply(lambda d: want <= set(d))]
    ev = exposed[exposed["reactions"].apply(lambda r: reaction in set(r))]
    n, pool = len(ev), len(exposed)
    empty = {"n_events": int(n), "pool": int(pool), "distinct_druglist_share": float("nan"), "top_date_share": float("nan"),
             "p_lists": float("nan"), "p_dates": float("nan"), "flag": ""}
    if n == 0:
        return empty
    lists_all = exposed["drugs"].apply(lambda d: tuple(sorted(d))).reset_index(drop=True)
    dates_all = exposed["date"].astype(str).reset_index(drop=True)
    ev_lists = ev["drugs"].apply(lambda d: tuple(sorted(d)))
    obs_distinct, obs_top = _stats(ev_lists, ev["date"].astype(str))
    out = {**empty, "distinct_druglist_share": float(obs_distinct), "top_date_share": float(obs_top)}
    if n < 3 or pool <= n:                       # nothing to compare with: every exposed report has the event
        out.update(p_lists=1.0, p_dates=1.0)
        return out
    rng = np.random.default_rng(seed)
    le = dh = 0
    for _ in range(n_mc):
        idx = rng.choice(pool, n, replace=False)
        d, t = _stats(lists_all.iloc[idx], dates_all.iloc[idx])
        le += d <= obs_distinct + 1e-12
        dh += t >= obs_top - 1e-12
    out["p_lists"], out["p_dates"] = le / n_mc, dh / n_mc
    msgs = []
    if out["p_lists"] < ALPHA:
        msgs.append("events share drug lists more often than the exposed reports as a whole (possible duplicate/batch reports)")
    if out["p_dates"] < ALPHA:
        msgs.append("events are bunched on fewer receive dates than the exposed reports as a whole (possible batch submission)")
    out["flag"] = "; ".join(msgs)
    return out


def annotate(results: pd.DataFrame, reports: pd.DataFrame, n_mc: int = 1000) -> pd.DataFrame:
    rows = [cluster_concentration(reports, (r.drug_a, r.drug_b, r.drug_c), r.reaction, n_mc) for r in results.itertuples()]
    out = results.copy()
    out["cluster_distinct_druglists"] = [r["distinct_druglist_share"] for r in rows]
    out["cluster_top_date_share"] = [r["top_date_share"] for r in rows]
    out["cluster_p_lists"] = [r["p_lists"] for r in rows]
    out["cluster_p_dates"] = [r["p_dates"] for r in rows]
    out["cluster_flag"] = [r["flag"] for r in rows]
    return out
