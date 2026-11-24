import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_gbm import boundary_rows, decisions_match, export_compact, node_count, write_fixture  # noqa: E402

FEATURES = [
    ("cement", "Cement", "kg/m³", 1),
    ("slag", "Blast furnace slag", "kg/m³", 1),
    ("flyash", "Fly ash", "kg/m³", 1),
    ("water", "Water", "kg/m³", 1),
    ("superplasticizer", "Superplasticizer", "kg/m³", 0.1),
    ("coarse", "Coarse aggregate", "kg/m³", 1),
    ("fine", "Fine aggregate", "kg/m³", 1),
    ("age", "Age", "days", None),
]
SEED = 42


def pct(a, q):
    return round(float(np.percentile(a, q)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xls", required=True)
    ap.add_argument("--fixture", default=os.path.join(os.path.dirname(__file__), "fixtures", "gbm_parity.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "assets", "models", "concrete.json"))
    a = ap.parse_args()

    raw = pd.read_excel(a.xls, engine="xlrd")
    raw.columns = [k for k, *_ in FEATURES] + ["strength"]
    df = raw.drop_duplicates().reset_index(drop=True)
    keys = [k for k, *_ in FEATURES]
    X, y = df[keys].to_numpy(float), df["strength"].to_numpy(float)
    same_x_diff_y = int(df.duplicated(subset=keys).sum())

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    grid = {"n_estimators": [200, 300, 400], "max_depth": [3, 4], "learning_rate": [0.05, 0.1]}
    gs = GridSearchCV(GradientBoostingRegressor(subsample=0.8, random_state=SEED), grid,
                      scoring="neg_root_mean_squared_error", cv=KFold(5, shuffle=True, random_state=SEED), refit=True)
    gs.fit(Xtr, ytr)
    m = gs.best_estimator_
    i = gs.best_index_
    pred = m.predict(Xte)
    metrics = {
        "test_r2": round(float(r2_score(yte, pred)), 4),
        "test_rmse": round(float(np.sqrt(mean_squared_error(yte, pred))), 4),
        "test_mae": round(float(mean_absolute_error(yte, pred)), 4),
        "test_median_ae": round(float(median_absolute_error(yte, pred)), 4),
        "cv_rmse_mean": round(float(-gs.cv_results_["mean_test_score"][i]), 4),
        "cv_rmse_std": round(float(gs.cv_results_["std_test_score"][i]), 4),
    }
    cv_table = [[p["n_estimators"], p["max_depth"], p["learning_rate"], round(float(-s), 4), round(float(sd), 4)]
                for p, s, sd in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"], gs.cv_results_["std_test_score"])]

    rng = np.random.default_rng(7)
    Xr = rng.uniform(X.min(0), X.max(0), size=(2000, X.shape[1]))
    Xb = boundary_rows(m, X, rng)
    Xc = np.vstack([X, Xr, Xb])
    model, digits, err = export_compact(m, Xc)
    assert digits is not None and err <= 1e-10, (digits, err)
    assert decisions_match(m, model, Xc)

    feats = []
    for j, (k, label, unit, step) in enumerate(FEATURES):
        col = X[:, j]
        f = {"key": k, "label": label, "unit": unit, "min": float(col.min()), "max": float(col.max()),
             "p01": pct(col, 1), "p50": pct(col, 50), "p99": pct(col, 99), "step": step}
        feats.append(f)
    age_values = sorted(int(v) for v in np.unique(X[:, 7]))
    binder = X[:, 0] + X[:, 1] + X[:, 2]
    wb = X[:, 3] / binder
    mass = X[:, :7].sum(1)

    order = np.argsort(yte, kind="stable")
    ranks = np.linspace(0, len(order) - 1, 10).round().astype(int)
    samples = [{"x": [float(v) for v in Xte[order[r]]], "y": round(float(yte[order[r]]), 4),
                "pred": float(pred[order[r]])} for r in ranks]

    out = {
        "meta": {
            "task": "Concrete compressive strength regression (MPa)",
            "dataset": "I-Cheng Yeh, \"Modeling of strength of high performance concrete using artificial neural networks\", "
                       "Cement and Concrete Research 28(12), 1998; UCI ML Repository, Concrete Compressive Strength, CC BY 4.0",
            "rows_raw": int(len(raw)),
            "rows_dedup": int(len(df)),
            "exact_duplicates_dropped": int(len(raw) - len(df)),
            "rows_same_inputs_different_strength_kept": same_x_diff_y,
            "split": {"train": int(len(Xtr)), "test": int(len(Xte)), "seed": SEED, "test_fraction": 0.2},
            "model": "sklearn GradientBoostingRegressor (squared error)",
            "selection": "5-fold CV (shuffled, seed 42) RMSE on the training split; refit on full training split; test evaluated once",
            "hyperparameters": {**gs.best_params_, "subsample": 0.8, "random_state": SEED},
            "cv_grid": {"cols": ["n_estimators", "max_depth", "learning_rate", "cv_rmse_mean", "cv_rmse_std"], "rows": cv_table},
            "trees": len(model["trees"]),
            "nodes": node_count(model),
            "metrics": metrics,
            "parity_max_abs_raw_py": err,
            "parity_check_rows": int(len(Xc)),
            "leaf_value_sig_digits": digits,
            "threshold_encoding": "shortest decimal t' such that float32(x) <= t' iff float32(x) <= t for every float32 x (exact; leaf-index equality verified)",
            "sklearn_version": __import__("sklearn").__version__,
            "inference": "raw = init + lr * sum(leaf values); prediction = raw (MPa). Cast inputs to float32 before comparing; go left when x[f] <= t.",
        },
        "features": feats,
        "ageValues": age_values,
        "derived": {
            "water_binder": {"p01": pct(wb, 1), "p50": pct(wb, 50), "p99": pct(wb, 99)},
            "total_mass": {"p01": pct(mass, 1), "p50": pct(mass, 50), "p99": pct(mass, 99)},
        },
        "model": model,
        "samples": samples,
    }
    write_fixture(a.fixture, "concrete", m, X.min(0), X.max(0), [s["x"] for s in samples])
    with open(a.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(json.dumps(out["meta"], indent=1))
    print("size", os.path.getsize(a.out))


if __name__ == "__main__":
    main()
