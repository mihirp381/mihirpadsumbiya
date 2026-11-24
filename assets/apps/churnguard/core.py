import json
import math
import struct
from bisect import bisect_left, bisect_right

DAYS = "days_since_last_transaction"
QUARTERS = ("bottom quarter", "second quarter", "third quarter", "top quarter")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def f32(v):
    return struct.unpack("f", struct.pack("f", v))[0]


def predict_raw(model, x):
    x32 = [f32(v) for v in x]
    lr = model["lr"]
    s = model["init"]
    for tr in model["trees"]:
        f, t, l, r = tr["f"], tr["t"], tr["l"], tr["r"]
        n = 0
        while f[n] != -1:
            n = l[n] if x32[f[n]] <= t[n] else r[n]
        s += lr * tr["v"][n]
    return s


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def proba(model, x):
    return sigmoid(predict_raw(model, x))


def drivers(model, x, baseline):
    r0 = predict_raw(model, x)
    out = []
    for j, v in enumerate(x):
        y = list(x)
        y[j] = baseline[j]
        z = predict_raw(model, y)
        out.append({"j": j, "value": v, "delta": r0 - z, "p_swap": sigmoid(z)})
    out.sort(key=lambda d: (-abs(d["delta"]), d["j"]))
    return out


def percentile(v, table):
    lo, hi = bisect_left(table, v), bisect_right(table, v)
    if hi > lo:
        return (lo + hi - 1) / (len(table) - 1) * 100
    if lo == 0:
        return 0.0
    if lo == len(table):
        return 100.0
    a, b = table[lo - 1], table[lo]
    return (lo - 1 + (v - a) / (b - a)) / (len(table) - 1) * 100


def quarter(v, box):
    if v < box["q1"]:
        return 0
    if v < box["median"]:
        return 1
    if v < box["q3"]:
        return 2
    return 3


def threshold_row(m, t):
    for row in m["thresholds"]:
        if abs(row[0] - t) < 1e-9:
            tp, fp, fn, tn = row[1:]
            return {"t": row[0], "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": tp + fp + fn + tn, "flagged": tp + fp}
    raise KeyError(t)


def num(v, d=0):
    s = f"{v:,.{d}f}"
    if d and "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace("-", "−")


def money(v):
    a = abs(v)
    return num(v, 2 if a < 10 else 1 if a < 100 else 0)


def value_text(key, v):
    if key == DAYS:
        return f"{num(v)} day" if v == 1 else f"{num(v)} days"
    return money(v)


def pct(p, d=1):
    return f"{100 * p:.{d}f}%"


def notes(m, pop, x, n=5):
    feats = m["features"]
    p = proba(m["model"], x)
    out = []
    for d in drivers(m["model"], x, m["baseline"])[:n]:
        j, v = d["j"], d["value"]
        ft, st = feats[j], pop["features"][j]
        ch, rt = st["churned"], st["retained"]
        tbl = st["percentiles"]
        here = percentile(v, tbl)
        near = "churned" if abs(here - percentile(ch["median"], tbl)) < abs(here - percentile(rt["median"], tbl)) else "retained"
        place = (f"is {value_text(ft['key'], v)}, in the {QUARTERS[quarter(v, rt)]} of retained customers and the "
                 f"{QUARTERS[quarter(v, ch)]} of churned ones, nearer the typical {'churner' if near == 'churned' else 'retained customer'} "
                 f"(medians {value_text(ft['key'], ch['median'])} churned, {value_text(ft['key'], rt['median'])} retained).")
        b = m["baseline"][j]
        if v == b:
            effect = "This is the median customer's own value, the reference point for the what-if."
        elif abs(d["p_swap"] - p) < 0.0005:
            effect = f"Swapping in the median customer's {value_text(ft['key'], b)} leaves the score at {pct(p)}, so it barely matters here."
        else:
            effect = (f"Swapping in the median customer's {value_text(ft['key'], b)} moves the score from {pct(p)} to "
                      f"{pct(d['p_swap'])}, so this value {'raises' if d['delta'] > 0 else 'lowers'} the risk.")
        out.append({"j": j, "key": ft["key"], "label": ft["label"], "delta": d["delta"], "text": f"{place} {effect}"})
    return out
