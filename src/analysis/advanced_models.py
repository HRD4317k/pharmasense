"""Predict whether a report contains a target reaction from which of the top-N drugs it lists.
Trains Lasso (L1 logistic), Random Forest and XGBoost; writes
  data/processed/model_metrics.json, models/metrics.csv, models/feature_importance.csv
(the two CSVs are what dashboard_v1/app.py's "ML models" page reads)."""
import argparse
import json
import os
from pathlib import Path

import duckdb
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

DB_PATH = Path("../../data/processed/faers.db")
METRICS_PATH = Path("../../data/processed/model_metrics.json")
MODELS_DIR = Path("../../models")


def prepare_data(con, target_reaction, num_drugs=50):
    print(f"Preparing data for target reaction: {target_reaction}")
    top = con.execute(f"""SELECT drugname FROM clean_drug GROUP BY drugname
                          ORDER BY COUNT(DISTINCT caseid) DESC, drugname LIMIT {int(num_drugs)}""").fetchdf()["drugname"].tolist()
    cases = con.execute("SELECT DISTINCT caseid FROM demo").fetchdf()["caseid"]
    pairs = con.execute("SELECT DISTINCT caseid, drugname FROM clean_drug").fetchdf()
    pairs = pairs[pairs["drugname"].isin(top)]
    positives = set(con.execute("SELECT DISTINCT caseid FROM clean_reac WHERE reaction = ?",
                                (target_reaction,)).fetchdf()["caseid"])

    X = (pd.crosstab(pairs["caseid"], pairs["drugname"]).clip(upper=1)
           .reindex(index=cases, columns=top, fill_value=0).fillna(0).astype(int))
    y = pd.Series(X.index.isin(positives).astype(int), index=X.index)
    names = list(X.columns)
    X.columns = [f"f{i}" for i in range(len(names))]   # XGBoost rejects names containing [ ] <
    stratify = y if y.sum() >= 2 else None
    return (*train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify), names)


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    m = {"Accuracy": accuracy_score(y_test, y_pred),
         "Precision": precision_score(y_test, y_pred, zero_division=0),
         "Recall": recall_score(y_test, y_pred, zero_division=0),
         "F1 Score": f1_score(y_test, y_pred, zero_division=0),
         "ROC AUC": roc_auc_score(y_test, y_prob),
         "AUPRC": average_precision_score(y_test, y_prob)}
    m = {k: round(float(v), 4) for k, v in m.items()}
    print(f"--- {name} ---\n{m}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None, help="reaction term (upper case); default: most frequent reaction")
    ap.add_argument("--num-drugs", type=int, default=50)
    args = ap.parse_args()

    os.chdir(Path(__file__).parent)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    target = args.target.upper() if args.target else con.execute(
        "SELECT reaction FROM clean_reac GROUP BY reaction ORDER BY COUNT(DISTINCT caseid) DESC LIMIT 1").fetchone()[0]
    X_train, X_test, y_train, y_test, names = prepare_data(con, target, args.num_drugs)
    con.close()
    args.target = target

    print(f"Train positives: {int(y_train.sum())}/{len(y_train)} | test positives: {int(y_test.sum())}/{len(y_test)}")
    if y_train.sum() < 2 or y_test.sum() < 1:
        print(f"Not enough '{args.target.upper()}' reports to train. Pass --target with a more common reaction.")
        return

    results = {"Target": args.target.upper(), "Models": {}}
    print("Training Lasso Regression...")
    lasso = LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=1000)
    lasso.fit(X_train, y_train)
    results["Models"]["Lasso"] = evaluate_model("Lasso", lasso, X_test, y_test)

    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["Models"]["Random Forest"] = evaluate_model("Random Forest", rf, X_test, y_test)

    print("Training XGBoost...")
    ratio = float((y_train == 0).sum()) / max(1, int((y_train == 1).sum()))
    xgb_model = xgb.XGBClassifier(scale_pos_weight=ratio, random_state=42, eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    results["Models"]["XGBoost"] = evaluate_model("XGBoost", xgb_model, X_test, y_test)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=4))
    pd.DataFrame([{"model": k, "auc": v["ROC AUC"], "auprc": v["AUPRC"], "f1": v["F1 Score"]}
                  for k, v in results["Models"].items()]).to_csv(MODELS_DIR / "metrics.csv", index=False)
    (pd.DataFrame({"feature": names, "importance": xgb_model.feature_importances_})
       .sort_values("importance", ascending=False).to_csv(MODELS_DIR / "feature_importance.csv", index=False))
    print("Saved model_metrics.json, models/metrics.csv and models/feature_importance.csv")


if __name__ == "__main__":
    main()
