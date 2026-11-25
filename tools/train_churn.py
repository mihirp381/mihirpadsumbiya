import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_gbm import boundary_rows, decisions_match, export_compact, node_count, write_fixture  # noqa: E402

SEED = 42
LABELS = {
    "current_balance": "Current balance",
    "previous_month_end_balance": "Balance at end of last month",
    "average_monthly_balance_prevQ": "Avg monthly balance, last quarter",
    "average_monthly_balance_prevQ2": "Avg monthly balance, quarter before",
    "current_month_debit": "Debits this month",
    "previous_month_debit": "Debits last month",
    "days_since_last_transaction": "Days since last transaction",
    "current_month_balance": "Avg balance this month",
    "previous_month_balance": "Avg balance last month",
    "branch_code": "Branch code",
}
A = ["current_balance", "previous_month_end_balance", "average_monthly_balance_prevQ", "average_monthly_balance_prevQ2",
     "current_month_debit", "previous_month_debit", "days_since_last_transaction"]
SETS = {"A": A, "B": A + ["current_month_balance", "previous_month_balance"],
        "C": ["current_balance", "current_month_debit", "previous_month_debit", "days_since_last_transaction",
              "average_monthly_balance_prevQ", "current_month_balance", "previous_month_balance",
              "average_monthly_balance_prevQ2", "previous_month_end_balance", "branch_code"]}
GRID = {"n_estimators": [150, 250], "max_depth": [2, 3], "learning_rate": [0.05, 0.1]}


def load(path):
    d = pd.read_csv(path)
    lt = pd.to_datetime(d["last_transaction"])
    n2018 = int((lt.dt.year == 2018).sum())
    lt = lt.where(lt.dt.year != 2018, lt + pd.DateOffset(years=4))
    end = lt.max()
    d["days_since_last_transaction"] = (end - lt).dt.days.astype(float)
    bal = [c for c in d.columns if "balance" in c]
    neg = {c: int((d[c] < 0).sum()) for c in bal if (d[c] < 0).any()}
    return d, n2018, str(end.date()), int((d[bal] < 0).any(axis=1).sum()), neg


def r(x, n=5):
    return round(float(x), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fixture", default=os.path.join(os.path.dirname(__file__), "fixtures", "gbm_parity.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "assets", "models", "churn.json"))
    a = ap.parse_args()
    d, n2018, end, n_neg, neg = load(a.csv)
    y = d["churn"].to_numpy(int)
    idx_tr, idx_te = train_test_split(np.arange(len(d)), test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    res = {}
    for name, cols in SETS.items():
        X = d[cols].to_numpy(float)
        gs = GridSearchCV(GradientBoostingClassifier(subsample=0.8, random_state=SEED), GRID,
                          scoring={"ll": "neg_log_loss", "auc": "roc_auc"}, refit="ll", cv=cv)
        gs.fit(X[idx_tr], y[idx_tr])
        i = gs.best_index_
        res[name] = {"cols": cols, "gs": gs, "X": X,
                     "cv_logloss": r(-gs.cv_results_["mean_test_ll"][i]), "cv_logloss_std": r(gs.cv_results_["std_test_ll"][i]),
                     "cv_auc": r(gs.cv_results_["mean_test_auc"][i]), "cv_auc_std": r(gs.cv_results_["std_test_auc"][i]),
                     "params": gs.best_params_}
        print(name, res[name]["cv_logloss"], res[name]["cv_auc"], gs.best_params_, flush=True)

    ship = "B" if res["B"]["cv_auc"] - res["A"]["cv_auc"] > 0.010 else "A"
    for name, rr in res.items():
        rr["test_auc"] = r(roc_auc_score(y[idx_te], rr["gs"].predict_proba(rr["X"][idx_te])[:, 1]))
    S = res[ship]
    m, X = S["gs"].best_estimator_, S["X"]
    Xtr, Xte, ytr, yte = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]
    p = m.predict_proba(Xte)[:, 1]
    metrics = {"test_roc_auc": r(roc_auc_score(yte, p)), "test_pr_auc_average_precision": r(average_precision_score(yte, p)),
               "test_brier": r(brier_score_loss(yte, p)), "test_logloss": r(log_loss(yte, p)),
               "test_base_rate": r(yte.mean()), "train_base_rate": r(ytr.mean()),
               "test_mean_predicted": r(p.mean()),
               "cv_logloss": S["cv_logloss"], "cv_logloss_std": S["cv_logloss_std"],
               "cv_roc_auc": S["cv_auc"], "cv_roc_auc_std": S["cv_auc_std"]}

    rng = np.random.default_rng(7)
    Xr = rng.uniform(X.min(0), X.max(0), size=(2000, X.shape[1]))
    Xc = np.vstack([X, Xr, boundary_rows(m, X, rng)])
    model, digits, err = export_compact(m, Xc)
    assert digits is not None and err <= 1e-10, (digits, err)
    assert decisions_match(m, model, Xc)

    feats = []
    for j, k in enumerate(S["cols"]):
        c = X[:, j]
        feats.append({"key": k, "label": LABELS[k], "min": float(c.min()), "max": float(c.max()),
                      **{f"p{q:02d}": r(np.percentile(c, q), 2) for q in (1, 5, 50, 95, 99)}})
    baseline = [float(np.median(Xtr[:, j])) for j in range(X.shape[1])]

    thresholds = []
    for t in np.round(np.arange(0.02, 0.981, 0.02), 2):
        pos = p >= t
        thresholds.append([float(t), int((pos & (yte == 1)).sum()), int((pos & (yte == 0)).sum()),
                           int((~pos & (yte == 1)).sum()), int((~pos & (yte == 0)).sum())])
    order = np.argsort(p, kind="stable")
    calibration = [[r(p[b].mean()), r(yte[b].mean()), int(len(b))] for b in np.array_split(order, 10)]

    samples = []
    for cls in (1, 0):
        ii = np.flatnonzero(yte == cls)
        ii = ii[np.argsort(p[ii], kind="stable")]
        for k in np.linspace(0, len(ii) - 1, 4).round().astype(int):
            samples.append({"x": [float(v) for v in Xte[ii[k]]], "churn": cls, "p": float(p[ii[k]])})

    comparison = {"cols": ["set", "n_features", "n_estimators", "max_depth", "learning_rate", "cv_logloss", "cv_logloss_std",
                           "cv_roc_auc", "cv_roc_auc_std", "test_roc_auc"],
                  "rows": [[n, len(rr["cols"]), rr["params"]["n_estimators"], rr["params"]["max_depth"], rr["params"]["learning_rate"],
                            rr["cv_logloss"], rr["cv_logloss_std"], rr["cv_auc"], rr["cv_auc_std"], rr["test_auc"]] for n, rr in res.items()]}
    out = {
        "meta": {
            "task": "Bank customer churn, binary classification (1 = churned)",
            "data": "Public bank customer churn dataset as cleaned by the ChurnGuard team for UTD FinHack 2024 (Cleaned_Modelling.csv)",
            "rows_used": int(len(d)), "split": {"train": int(len(idx_tr)), "test": int(len(idx_te)), "stratified": True, "seed": SEED},
            "handling": [
                "No missing values, no duplicate rows, no duplicate customer ids found; all rows kept.",
                f"Negative balances (likely overdrafts) in {n_neg} rows kept as-is (per column: {neg}); heavy right tails kept (trees are invariant to monotone scaling).",
                f"days_since_last_transaction recomputed from last_transaction as days before the latest date in the data ({end}); "
                f"the CSV's version was measured from the date the team ran their notebook (offset of 97 days).",
                f"{n2018} rows dated 2018-12-31 shifted by +4 years to match the team's 2019->2023 year rewrite (they would otherwise read 1826+ days).",
                "No class reweighting or resampling: probabilities are calibrated to the natural base rate.",
                "Identifier columns (customer_id, branch_code) are not used in the shipped model.",
            ],
            "model": "sklearn GradientBoostingClassifier (log-loss)",
            "selection": "5-fold stratified CV (shuffled, seed 42) log-loss on the training split; ship A unless B beats A by > 0.010 CV ROC-AUC",
            "shipped_set": ship,
            "hyperparameters": {**S["params"], "subsample": 0.8, "random_state": SEED},
            "trees": len(model["trees"]), "nodes": node_count(model),
            "metrics": metrics,
            "comparison": comparison,
            "base_rate": metrics["test_base_rate"],
            "parity_max_abs_raw_py": err, "parity_check_rows": int(len(Xc)), "leaf_value_sig_digits": digits,
            "threshold_encoding": "shortest decimal t' such that float32(x) <= t' iff float32(x) <= t for every float32 x (exact; leaf-index equality verified)",
            "thresholds_rule": "predicted positive when p >= t; rows are [t, tp, fp, fn, tn] on the test split",
            "calibration_rule": "10 equal-count bins of the test split sorted by p; rows are [mean_predicted, observed_rate, n]",
            "sklearn_version": __import__("sklearn").__version__,
            "inference": "raw = init + lr * sum(leaf values); p = sigmoid(raw). Cast inputs to float32 before comparing; go left when x[f] <= t.",
        },
        "features": feats,
        "baseline": baseline,
        "model": model,
        "thresholds": thresholds,
        "calibration": calibration,
        "samples": samples,
    }
    write_fixture(a.fixture, "churn", m, X.min(0), X.max(0), [s["x"] for s in samples])
    with open(a.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(json.dumps(out["meta"], indent=1))
    print("size", os.path.getsize(a.out))


if __name__ == "__main__":
    main()
