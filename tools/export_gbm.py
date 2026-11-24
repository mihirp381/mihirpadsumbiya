import numpy as np


def raw_init(model, n_features):
    return float(model._raw_predict_init(np.zeros((1, n_features)))[0, 0])


def compact_threshold(t):
    # Shortest decimal d with {float32 x: x <= d} == {float32 x: x <= t}.
    lo = np.float32(t)
    if float(lo) > t:
        lo = np.nextafter(lo, np.float32(-np.inf))
    hi = float(np.nextafter(lo, np.float32(np.inf)))
    lo = float(lo)
    for k in range(1, 18):
        for base in (lo, (lo + hi) / 2):
            c = float(f"{base:.{k}g}")
            if lo <= c < hi:
                return c
    return float(t)


def export_gbm(model, value_digits=None):
    trees = []
    for est in model.estimators_[:, 0]:
        tr = est.tree_
        leaf = tr.children_left == -1
        f = np.where(leaf, -1, tr.feature).astype(int).tolist()
        t = [0 if lf else compact_threshold(float(th)) for lf, th in zip(leaf, tr.threshold)]
        v = [float(val) if lf else 0 for lf, val in zip(leaf, tr.value[:, 0, 0])]
        if value_digits is not None:
            v = [float(f"{x:.{value_digits}g}") if x else 0 for x in v]
        trees.append({"f": f, "t": t, "l": tr.children_left.astype(int).tolist(),
                      "r": tr.children_right.astype(int).tolist(), "v": v})
    return {"init": raw_init(model, model.n_features_in_), "lr": float(model.learning_rate), "trees": trees}


def export_compact(model, X_check, tol=1e-10):
    for digits in range(8, 18):
        d = export_gbm(model, digits)
        err, _ = check(model, d, X_check)
        if err <= tol:
            return d, digits, err
    d = export_gbm(model)
    return d, None, check(model, d, X_check)[0]


def predict_raw(d, X):
    X32 = np.asarray(X, dtype=np.float32).astype(np.float64)
    out = np.empty(len(X32))
    for i, x in enumerate(X32):
        s = d["init"]
        for tr in d["trees"]:
            n = 0
            while tr["f"][n] != -1:
                n = tr["l"][n] if x[tr["f"][n]] <= tr["t"][n] else tr["r"][n]
            s += d["lr"] * tr["v"][n]
        out[i] = s
    return out


def sklearn_raw(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def check(model, d, X):
    ours = predict_raw(d, X)
    ref = sklearn_raw(model, X)
    err = float(np.max(np.abs(ours - ref)))
    if hasattr(model, "predict_proba"):
        perr = float(np.max(np.abs(1 / (1 + np.exp(-ours)) - model.predict_proba(X)[:, 1])))
    else:
        perr = float(np.max(np.abs(ours - model.predict(X))))
    return err, perr


def decisions_match(model, d, X):
    X32 = np.asarray(X, dtype=np.float32)
    for est, tr in zip(model.estimators_[:, 0], d["trees"]):
        ref = est.tree_.apply(X32)
        x64 = X32.astype(np.float64)
        n = np.zeros(len(X32), int)
        for _ in range(64):
            f = np.array(tr["f"])[n]
            inner = f != -1
            if not inner.any():
                break
            go = x64[np.arange(len(n)), np.maximum(f, 0)] <= np.array(tr["t"])[n]
            n = np.where(inner, np.where(go, np.array(tr["l"])[n], np.array(tr["r"])[n]), n)
        if not np.array_equal(n, ref):
            return False
    return True


def boundary_rows(model, base, rng, per_tree=4):
    rows = []
    for est in model.estimators_[:, 0]:
        tr = est.tree_
        inner = np.flatnonzero(tr.children_left != -1)
        for n in rng.choice(inner, size=min(per_tree, len(inner)), replace=False):
            lo = np.float32(tr.threshold[n])
            if float(lo) > tr.threshold[n]:
                lo = np.nextafter(lo, np.float32(-np.inf))
            for val in (lo, np.nextafter(lo, np.float32(np.inf))):
                r = base[rng.integers(len(base))].copy()
                r[tr.feature[n]] = float(val)
                rows.append(r)
    return np.array(rows)


def write_fixture(path, key, model, lo, hi, sample_x, n=200, seed=7):
    import json
    import os
    X = np.vstack([np.random.default_rng(seed).uniform(lo, hi, size=(n, len(lo))), np.asarray(sample_x, float)])
    entry = {"x": X.tolist(), "raw": sklearn_raw(model, X).tolist()}
    if hasattr(model, "predict_proba"):
        entry["p"] = model.predict_proba(X)[:, 1].tolist()
    else:
        entry["pred"] = model.predict(X).tolist()
    data = json.load(open(path)) if os.path.exists(path) else {}
    data[key] = entry
    with open(path, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))


def node_count(d):
    return sum(len(t["f"]) for t in d["trees"])
