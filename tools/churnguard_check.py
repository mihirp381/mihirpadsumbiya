import json
import os
import random
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
APP = os.path.join(ROOT, "assets", "apps", "churnguard")
sys.path.insert(0, APP)
import core  # noqa: E402

JS = """
import { readFileSync } from "node:fs";
import { predictRaw, sigmoid } from "./assets/js/lib/gbm.js";
import { drivers } from "./assets/js/lib/demo-math.js";
const { xs, base } = JSON.parse(readFileSync(0, "utf8"));
const m = JSON.parse(readFileSync("./assets/models/churn.json", "utf8"));
const raw = (x) => predictRaw(m.model, x);
console.log(JSON.stringify(xs.map((x) => ({ p: sigmoid(raw(x)), d: drivers(raw, x, base).map((e) => [e.j, e.delta]) }))));
"""

failed = 0


def check(name, ok, detail=""):
    global failed
    failed += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")


m = core.load(os.path.join(ROOT, "assets", "models", "churn.json"))
pop = core.load(os.path.join(APP, "population.json"))
model, samples, base, feats = m["model"], m["samples"], m["baseline"], m["features"]

err = max(abs(core.proba(model, s["x"]) - s["p"]) for s in samples)
check(f"core.proba = stored p ({len(samples)} samples)", err <= 1e-9, f"max diff {err:.1e}")

rng = random.Random(42)
xs = [s["x"] for s in samples] + [base]
for _ in range(500):
    xs.append([round(rng.uniform(f["p01"], f["p99"]), 2) if f["key"] != core.DAYS else float(rng.randint(0, 365)) for f in feats])
for s in samples:
    for j in range(len(feats)):
        y = list(s["x"])
        y[j] = base[j]
        xs.append(y)
out = subprocess.run(["node", "--input-type=module", "-e", JS], input=json.dumps({"xs": xs, "base": base}),
                     capture_output=True, text=True, cwd=ROOT, check=True)
js = json.loads(out.stdout)
err = max(abs(core.proba(model, x) - r["p"]) for x, r in zip(xs, js))
check(f"core.proba = gbm.js ({len(xs)} inputs)", err <= 1e-9, f"max diff {err:.1e}")
same = all([d["j"] for d in core.drivers(model, x, base)] == [e[0] for e in r["d"]] for x, r in zip(xs, js))
derr = max(abs(d["delta"] - dict(r["d"])[d["j"]]) for x, r in zip(xs, js) for d in core.drivers(model, x, base))
check("core.drivers order and deltas = demo-math drivers", same and derr <= 1e-9, f"max diff {derr:.1e}")

counts = [len(core.notes(m, pop, x)) for x in [s["x"] for s in samples] + [base]]
check("notes: 5 per held-out customer and for the median", counts == [5] * (len(samples) + 1), str(counts))
check("population.json covers the model's features", [f["key"] for f in pop["features"]] == [f["key"] for f in feats])
check("population.json rows = training split", pop["meta"]["rows"] == m["meta"]["split"]["train"],
      f"{pop['meta']['rows']} of {m['meta']['split']['train']}")
k = next(i for i, s in enumerate(samples) if s["churn"] == 1 and s["p"] >= 0.3)
check("default customer matches Figure 3", k == 2 and core.pct(samples[k]["p"]) == "63.1%", f"Customer {chr(65 + k)} at {core.pct(samples[k]['p'])}")
check("threshold rows at every slider step", all(core.threshold_row(m, round(0.02 * i, 2)) for i in range(1, 50)))

sys.exit(1 if failed else 0)
