import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_churn import SEED, load  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def r(x):
    return round(float(x), 2)


def box(c):
    q1, q2, q3 = np.percentile(c, [25, 50, 75])
    iqr = q3 - q1
    inside = c[(c >= q1 - 1.5 * iqr) & (c <= q3 + 1.5 * iqr)]
    return {"n": int(len(c)), "lo": r(inside.min()), "q1": r(q1), "median": r(q2), "q3": r(q3), "hi": r(inside.max()),
            "outliers": int(len(c) - len(inside))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default=os.path.join(ROOT, "assets", "models", "churn.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "assets", "apps", "churnguard", "population.json"))
    a = ap.parse_args()
    with open(a.model) as fh:
        m = json.load(fh)
    d = load(a.csv)[0]
    y = d["churn"].to_numpy(int)
    idx_tr, _ = train_test_split(np.arange(len(d)), test_size=0.2, stratify=y, random_state=SEED)
    assert len(idx_tr) == m["meta"]["split"]["train"]
    keys = [f["key"] for f in m["features"]]
    X = d[keys].to_numpy(float)[idx_tr]
    yt = y[idx_tr]
    assert [float(np.median(X[:, j])) for j in range(len(keys))] == m["baseline"], "split does not match churn.json"

    feats = []
    for j, k in enumerate(keys):
        c = X[:, j]
        feats.append({"key": k, "percentiles": [r(v) for v in np.percentile(c, np.arange(101))],
                      "churned": box(c[yt == 1]), "retained": box(c[yt == 0])})
    out = {
        "meta": {
            "source": "Training split of churn.json (stratified 80/20, seed 42), the same rows the model was fitted on",
            "rows": int(len(idx_tr)), "churned": int(yt.sum()), "retained": int(len(yt) - yt.sum()),
            "percentiles_rule": "value at each whole percentile 0..100 of all training customers (numpy linear interpolation)",
            "box_rule": "quartiles by numpy linear interpolation; whiskers at the most extreme value within 1.5 IQR of the box",
        },
        "features": feats,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
        fh.write("\n")
    print(json.dumps(out["meta"], indent=1))
    for f in feats:
        print(f["key"], f["churned"], f["retained"])
    print("size", os.path.getsize(a.out))


if __name__ == "__main__":
    main()
