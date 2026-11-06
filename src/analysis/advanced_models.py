"""Predict whether a report contains a target reaction from which of the top-N drugs it lists.

Trains Lasso (L1 logistic), Random Forest and XGBoost and scores every model TWO ways:

  random      ordinary 5-fold cross-validation (reports shuffled)                 - the usual, optimistic number
  batch-aware 5-fold cross-validation where all reports received on the same date stay together
              (GroupKFold). FAERS arrives in batches (one study, one litigation, one manufacturer upload), so near-duplicate
              reports share a date; a random split puts some of them in training and some in testing, and the model
              "predicts" the test copies it has effectively already seen. The gap between the two numbers is the leakage.

The headline numbers (auc / auprc / f1 in models/metrics.csv) are the batch-aware ones.

The target reaction is never a term that exclusions.py discards as non-specific ("ILL-DEFINED DISORDER", "DRUG INEFFECTIVE",
...), because predicting vague wording says nothing about drugs.

Writes data/processed/model_metrics.json, models/metrics.csv and models/feature_importance.csv
(the two CSVs are what the app's "ML models" page reads). Needs data/processed/faers.db (python load_to_duckdb.py).
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import exclusions  # noqa: E402

DB_PATH = ROOT / "data" / "processed" / "faers.db"
METRICS_PATH = ROOT / "data" / "processed" / "model_metrics.json"
MODELS_DIR = ROOT / "models"
FOLDS, SEED = 5, 42


# ------------------------------------------------------------------------------------------------ data
MAX_TOP_DATE_SHARE = 0.5       # a reaction with more than half its reports on ONE receive date is a batch, not a drug pattern


def date_concentration(con):
    """reaction -> (reports, share of those reports received on its single busiest date)."""
    d = con.execute("""SELECT r.reaction, d.receive_date, COUNT(DISTINCT r.caseid) n FROM clean_reac r
                       JOIN demo d USING (caseid) GROUP BY 1, 2""").fetchdf()
    g = d.groupby("reaction")["n"]
    return pd.DataFrame({"n": g.sum(), "top_share": g.max() / g.sum()}).sort_values(["n"], ascending=False)


def pick_target(con, requested=None):
    """The requested reaction, or the most frequent one that is (1) a real medical event per exclusions.py and
    (2) spread over several receive dates, so that a batch-aware evaluation is possible."""
    conc = date_concentration(con)
    if requested:
        t = requested.upper()
        cat = exclusions.classify(t)
        if cat:
            print(f"WARNING: '{t}' is excluded elsewhere in PharmaSense as '{cat}'. Predicting it says little about drugs.")
        if t in conc.index and conc.loc[t, "top_share"] > MAX_TOP_DATE_SHARE:
            print(f"WARNING: {conc.loc[t, 'top_share']:.0%} of '{t}' reports were received on one date - likely one batch, "
                  "so a good score may reflect that batch and not a drug pattern.")
        return t
    for term, row in conc.iterrows():
        if exclusions.classify(term):
            continue
        if row["top_share"] > MAX_TOP_DATE_SHARE:
            print(f"Skipping '{term}' ({int(row['n'])} reports): {row['top_share']:.0%} received on one date - a batch, not a drug pattern.")
            continue
        return term
    raise SystemExit("No reaction term is both a real medical event and spread over several receive dates.")


def prepare_data(con, target, num_drugs=50):
    """X: one 0/1 column per top-N drug, y: target reaction present, groups: receive date (the batch proxy)."""
    print(f"Preparing data for target reaction: {target}")
    top = con.execute(f"""SELECT drugname FROM clean_drug GROUP BY drugname
                          ORDER BY COUNT(DISTINCT caseid) DESC, drugname LIMIT {int(num_drugs)}""").fetchdf()["drugname"].tolist()
    demo = con.execute("SELECT DISTINCT caseid, receive_date FROM demo").fetchdf().drop_duplicates("caseid").set_index("caseid")
    pairs = con.execute("SELECT DISTINCT caseid, drugname FROM clean_drug").fetchdf()
    pairs = pairs[pairs["drugname"].isin(top)]
    positives = set(con.execute("SELECT DISTINCT caseid FROM clean_reac WHERE reaction = ?", (target,)).fetchdf()["caseid"])

    X = (pd.crosstab(pairs["caseid"], pairs["drugname"]).clip(upper=1)
           .reindex(index=demo.index, columns=top, fill_value=0).fillna(0).astype(int))
    y = pd.Series(X.index.isin(positives).astype(int), index=X.index)
    groups = demo.loc[X.index, "receive_date"].astype(str).to_numpy()
    names = list(X.columns)
    X.columns = [f"f{i}" for i in range(len(names))]            # XGBoost rejects names containing [ ] <
    return X, y, groups, names


# ------------------------------------------------------------------------------------------------ models
def _lasso():
    """L1 logistic regression. scikit-learn 1.8 deprecated penalty='l1' in favour of l1_ratio=1; on older versions
    l1_ratio alone would silently give an L2 model, so the spelling is chosen by version."""
    import sklearn
    new = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
    kw = {"l1_ratio": 1} if new else {"penalty": "l1"}
    return LogisticRegression(solver="liblinear", C=0.1, class_weight="balanced", max_iter=1000, **kw)   # C=0.1: strong enough penalty to actually select


def make_models(y_train):
    ratio = float((y_train == 0).sum()) / max(1, int((y_train == 1).sum()))
    return {
        "Lasso": _lasso(),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(scale_pos_weight=ratio, random_state=SEED, eval_metric="logloss", verbosity=0),
    }


def out_of_fold(X, y, splitter, groups=None):
    """Pooled out-of-fold probabilities for every model. A fold whose training part lacks both classes is skipped (and counted)."""
    P = {k: np.full(len(y), np.nan) for k in make_models(y)}
    skipped = 0
    for tr, te in splitter.split(X, y, groups):
        if y.iloc[tr].sum() < 2 or y.iloc[te].sum() < 1 or y.iloc[tr].sum() == len(tr):
            skipped += 1
            continue
        for name, m in make_models(y.iloc[tr]).items():
            m.fit(X.iloc[tr], y.iloc[tr])
            P[name][te] = m.predict_proba(X.iloc[te])[:, 1]
    return P, skipped


def score(y, p):
    ok = ~np.isnan(p)
    y, p = np.asarray(y)[ok], p[ok]
    pred = (p >= 0.5).astype(int)
    m = {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
         "Recall": recall_score(y, pred, zero_division=0), "F1 Score": f1_score(y, pred, zero_division=0),
         "ROC AUC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p)}
    return {k: round(float(v), 4) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None, help="reaction term (upper case); default: most frequent real medical event")
    ap.add_argument("--num-drugs", type=int, default=50)
    args = ap.parse_args()

    if not DB_PATH.exists():
        raise SystemExit(f"{DB_PATH} not found. Run: python load_to_duckdb.py")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    target = pick_target(con, args.target)
    X, y, groups, names = prepare_data(con, target, args.num_drugs)
    con.close()

    n_groups = len(set(groups))
    print(f"{len(y):,} reports, {int(y.sum())} with '{target}' ({y.mean():.1%}); {n_groups} distinct receive dates")
    if y.sum() < 10:
        raise SystemExit(f"Only {int(y.sum())} reports mention '{target}'. Pass --target with a more common reaction.")

    random_cv = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    P_rand, skip_r = out_of_fold(X, y, random_cv)
    batch = None
    if n_groups >= 3:
        batch = GroupKFold(min(FOLDS, n_groups))
        P_batch, skip_b = out_of_fold(X, y, batch, groups)
        if all(np.isnan(p).all() for p in P_batch.values()):
            print(f"Batch-aware scoring impossible for '{target}': every fold lacked positives in training or testing "
                  "(the reaction is concentrated on too few dates). Reporting the random split only.")
            batch = None
        elif skip_b:
            print(f"Note: {skip_b} of {batch.n_splits} batch-aware folds skipped (no positives on one side).")
    else:
        print("Fewer than 3 distinct receive dates: batch-aware scoring is not possible, only the random split is reported.")

    results = {"Target": target, "n_reports": int(len(y)), "n_positive": int(y.sum()), "prevalence": round(float(y.mean()), 4),
               "n_receive_dates": n_groups, "folds": FOLDS, "Models": {},
               "note": ("batch-aware = whole receive dates held out together (GroupKFold); random = ordinary shuffled 5-fold. "
                        "A large gap means near-duplicate reports from one batch are inflating the random-split score.")}
    for name in P_rand:
        entry = {"random": score(y, P_rand[name])}
        if batch is not None:
            entry["batch_aware"] = score(y, P_batch[name])
        results["Models"][name] = entry
        head = entry.get("batch_aware", entry["random"])
        print(f"--- {name} ---  random AUC {entry['random']['ROC AUC']:.3f}"
              + (f" | batch-aware AUC {head['ROC AUC']:.3f}" if batch is not None else "") + f" | AUPRC {head['AUPRC']:.3f} (chance {y.mean():.3f})")

    # headline = batch-aware when available
    rows = []
    for k, v in results["Models"].items():
        h = v.get("batch_aware", v["random"])
        rows.append({"model": k, "auc": h["ROC AUC"], "auprc": h["AUPRC"], "f1": h["F1 Score"],
                     "auc_random": v["random"]["ROC AUC"], "auprc_random": v["random"]["AUPRC"]})

    # feature importance / direction from models fitted on all reports (for interpretation, not for scoring)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = make_models(y)
        for m in full.values():
            m.fit(X, y)
    fi = pd.DataFrame({"feature": names, "importance": full["XGBoost"].feature_importances_,
                       "rf_importance": full["Random Forest"].feature_importances_,
                       "lasso_coef": full["Lasso"].coef_[0]}).sort_values("importance", ascending=False)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=4))
    pd.DataFrame(rows).to_csv(MODELS_DIR / "metrics.csv", index=False)
    fi.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    print("Saved model_metrics.json, models/metrics.csv and models/feature_importance.csv")


if __name__ == "__main__":
    main()
