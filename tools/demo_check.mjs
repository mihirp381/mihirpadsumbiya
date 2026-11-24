import { readFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { predictRaw, sigmoid } from "../assets/js/lib/gbm.js";
import { scaleAsinh } from "../assets/js/lib/plot.js";
import * as M from "../assets/js/lib/demo-math.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const json = (p) => JSON.parse(readFileSync(join(root, p), "utf8"));
let failed = 0;
const check = (name, ok, detail = "") => {
  if (!ok) failed++;
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? `: ${detail}` : ""}`);
};
const maxAbs = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

const concrete = json("assets/models/concrete.json");
{
  const { features: F, samples, derived } = concrete;
  const f = (x) => predictRaw(concrete.model, x);
  check("concrete live prediction = stored pred (10 samples)", maxAbs(samples.map((s) => f(s.x)), samples.map((s) => s.pred)) <= 1e-9);
  const s7 = samples[6];
  check("concrete Lab mix 7 line", `${M.fmt(s7.y, 1)} / ${M.fmt(f(s7.x), 1)} / ${M.signed(f(s7.x) - s7.y, 1)}` === "43.7 / 39.0 / −4.7", `${M.fmt(s7.y, 1)} MPa measured, ${M.fmt(f(s7.x), 1)} model, ${M.signed(f(s7.x) - s7.y, 1)}`);
  const err = (s) => Math.abs(f(s.x) - s.y);
  const dk = M.defaultMix(samples, 7);
  const at28 = samples.filter((s) => s.x[7] === 28).map(err).sort((a, b) => a - b);
  check("concrete default mix is the lower-median |error| of the 28-day mixes", at28.length > 0 && samples[dk].x[7] === 28 && err(samples[dk]) === at28[Math.floor((at28.length - 1) / 2)], `Lab mix ${dk + 1} of ${at28.length} at 28 d, ${M.fmt(samples[dk].y, 1)} measured, ${M.fmt(f(samples[dk].x), 1)} model, ${M.signed(f(samples[dk].x) - samples[dk].y, 1)}`);
  let dErr = 0;
  for (const s of samples) {
    const d = M.concreteDerived(s.x);
    const b = s.x[0] + s.x[1] + s.x[2];
    dErr = Math.max(dErr, Math.abs(d.wb - s.x[3] / b), Math.abs(d.mass - s.x.slice(0, 7).reduce((a, v) => a + v, 0)));
  }
  check("concrete w/b and total mass", dErr < 1e-12);
  const zero = F.map((ft) => ft.p50);
  zero[0] = zero[1] = zero[2] = 0;
  check("concrete binder 0 flags extrapolation", M.outsideTraining(M.concreteDerived(zero), derived));
  const base = F.map((ft) => ft.p50);
  base[7] = 28;
  check("concrete p50 mix at 28 d is inside training band", !M.outsideTraining(M.concreteDerived(base), derived));
  let sErr = 0, sortOk = true;
  for (const x of [base, ...samples.map((s) => s.x)]) {
    const rows = M.sensitivity(f, x, F, [0, 1, 2, 3, 4, 5, 6]);
    const y0 = f(x);
    for (const r of rows) {
      const ft = F[r.i];
      const h = 0.1 * (ft.p99 - ft.p01);
      const lo = x.slice(), hi = x.slice();
      lo[r.i] = Math.min(ft.max, Math.max(ft.min, x[r.i] - h));
      hi[r.i] = Math.min(ft.max, Math.max(ft.min, x[r.i] + h));
      sErr = Math.max(sErr, Math.abs(r.h - h), Math.abs(r.down - (f(lo) - y0)), Math.abs(r.up - (f(hi) - y0)));
    }
    const mags = rows.map((r) => Math.max(Math.abs(r.down), Math.abs(r.up)));
    sortOk &&= mags.every((v, i) => i === 0 || mags[i - 1] >= v);
  }
  check("concrete sensitivity, two ways (11 mixes)", sErr === 0 && sortOk, `max |d| ${sErr}`);
  const sweep = M.ageSweep(f, base, 7, 365);
  const y28 = base.slice();
  check("concrete age sweep matches point prediction at 28 d", sweep.length === 365 && sweep[27] === f(y28));
  const ages = Array.from({ length: 365 }, (_, i) => i + 1);
  check("age slider log mapping round-trips 1..365 for tick days", [1, 3, 7, 28, 90, 365].every((d) => M.posToAge(M.ageToPos(d)) === d) && ages.every((d) => Math.abs(M.posToAge(M.ageToPos(d)) - d) <= 1));
}

const churn = json("assets/models/churn.json");
{
  const { features: F, samples, baseline, thresholds, meta } = churn;
  const raw = (x) => predictRaw(churn.model, x);
  let rt = 0;
  F.forEach((ft, i) => {
    if (i === 6) return;
    const sc = scaleAsinh(ft.p50, 0, ft.p99, 0, 200);
    for (const v of [0, ft.p01, ft.p05, ft.p50, ft.p95, ft.p99]) rt = Math.max(rt, Math.abs(sc.invert(sc(v)) - v) / Math.max(1, Math.abs(v)));
    check(`churn asinh scale ${ft.label}`, Math.abs(sc(0)) < 1e-9 && Math.abs(sc(ft.p99) - 200) < 1e-9 && Math.abs(sc(ft.p50) - 200 * Math.asinh(1) / Math.asinh(ft.p99 / ft.p50)) < 1e-9);
  });
  check("churn asinh round-trips (relative)", rt < 1e-12, rt.toExponential(2));
  check("churn live p = stored p (8 samples)", maxAbs(samples.map((s) => sigmoid(raw(s.x))), samples.map((s) => s.p)) <= 1e-9);
  check("churn drivers are 0 at the baseline", M.drivers(raw, baseline, baseline).every((d) => d.delta === 0));
  let dErr = 0;
  for (const s of samples) for (const d of M.drivers(raw, s.x, baseline)) {
    const y = s.x.slice();
    y[d.j] = baseline[d.j];
    dErr = Math.max(dErr, Math.abs(d.delta - (raw(s.x) - raw(y))));
  }
  check("churn drivers match direct swap", dErr === 0);
  const n = meta.split.test;
  check(`churn every threshold row sums to ${n}`, thresholds.length === 49 && thresholds.every((r) => r[1] + r[2] + r[3] + r[4] === n));
  const t30 = M.thresholdStats(thresholds.find((r) => Math.abs(r[0] - 0.3) < 1e-9));
  const sent30 = M.thresholdSentence(t30);
  check("churn threshold 0.30 sentence", sent30 === `At 0.30: flags ${M.fmt(t30.tp + t30.fp)} of 4,882 held-out customers, catches ${M.pct(t30.tp / (t30.tp + t30.fn))} of the churners, and ${M.pct(t30.tp / (t30.tp + t30.fp))} of the flags are real.`, sent30);
  const none = thresholds.find((r) => r[1] + r[2] === 0);
  const sNone = M.thresholdStats(none);
  check("churn threshold with nobody flagged", sNone.precision === null && /flags nobody/.test(M.thresholdSentence(sNone)) && /n\/a/.test(M.thresholdSentence(sNone)), M.thresholdSentence(sNone));
  const pos = thresholds[0][1] + thresholds[0][3];
  check("churn positives constant across thresholds and match base rate", thresholds.every((r) => r[1] + r[3] === pos) && Math.abs(pos / n - meta.metrics.test_base_rate) < 1e-4, `${pos}/${n}`);
  const ck = M.caughtChurner(samples, 0.3);
  check("churn default customer is the first churner caught at 0.30", ck >= 0 && samples[ck].churn === 1 && samples[ck].p >= 0.3 && samples.slice(0, ck).every((s) => !(s.churn === 1 && s.p >= 0.3)) && M.drivers(raw, samples[ck].x, baseline).some((d) => d.delta !== 0), `Customer ${"ABCDEFGH"[ck]}, p ${M.fmt(samples[ck].p, 3)}`);
  const outcomes = samples.map((s, k) => `${"ABCDEFGH"[k]}: ${M.outcomeText(s.churn, s.p >= 0.3)}`);
  check("churn sample outcomes at 0.30 cover all four cases", new Set(outcomes.map((o) => o.split(" ").slice(-2).join(" "))).size === 4, outcomes.join(" | "));
}

{
  const b = json("assets/models/bcrp.json");
  const idx = M.compoundIndex(b.rows);
  const inh = b.rows.filter((r) => r[1] === 1).length;
  check("bcrp counts match meta", b.rows.length === b.meta.total && inh === b.meta.inhibitors && b.rows.length - inh === b.meta.substrates, `${b.rows.length} = ${inh} + ${b.rows.length - inh}`);
  check("bcrp size for caption", statSync(join(root, "assets/models/bcrp.json")).size > 0, `${M.fmt(statSync(join(root, "assets/models/bcrp.json")).size / 1e6, 1)} MB`);
  const rank = (c, q) => (c.lower === q ? 0 : c.lower.startsWith(q) ? 1 : 2);
  let ordered = true;
  for (const q of ["mitox", "topotecan", "an", "Hoechst", "e"]) {
    const { n, items } = M.searchCompounds(idx, q, "all", 1e6);
    const ql = q.toLowerCase();
    const brute = idx.filter((c) => c.lower.includes(ql)).length;
    ordered &&= n === brute && items.every((c, i) => i === 0 || (() => {
      const a = items[i - 1];
      return rank(a, ql) < rank(c, ql) || (rank(a, ql) === rank(c, ql) && a.nrefs >= c.nrefs);
    })());
  }
  check("bcrp ranking exact > prefix > substring, then nrefs", ordered);
  const mitox = M.searchCompounds(idx, "mitox");
  check("bcrp 'mitox' first prefix match is Mitoxantrone", mitox.items.find((c) => c.lower !== "mitox")?.name === "Mitoxantrone", mitox.items.slice(0, 3).map((c) => `${c.name} (${c.nrefs})`).join(", "));
  check("bcrp 'MITOXANTRONE' returns Mitoxantrone first", M.searchCompounds(idx, "MITOXANTRONE").items[0].name === "Mitoxantrone");
  const fi = M.searchCompounds(idx, "", "inhibitor", 1e6), fs = M.searchCompounds(idx, "", "substrate", 1e6);
  check("bcrp filter", fi.n === b.meta.inhibitors && fs.n === b.meta.substrates && fi.items.every((c) => c.act === 1) && fs.items.every((c) => c.act === 0));
  const top = M.searchCompounds(idx, "");
  check("bcrp empty query lists 50 most-referenced", top.items.length === 50 && top.items[0].nrefs === Math.max(...idx.map((c) => c.nrefs)), `${top.items[0].name} (${top.items[0].nrefs})`);
  check("bcrp no-match query", M.searchCompounds(idx, "zzzzqqq").n === 0);
}

{
  const c = json("assets/models/crack-cnn.json");
  const blocks = M.cnnBlocks(c.tensors);
  const params = blocks.reduce((a, b) => a + (b.params || 0), 0);
  check("crack architecture params sum to meta.params", params === c.meta.params, blocks.map((b) => `${b.name} ${b.shape || ""} ${b.params || ""}`.trim()).join(" | "));
  const cm = c.meta.metrics_shipped.confusion;
  const st = M.confusionStats(cm);
  check("crack confusion matches shipped accuracy", Math.abs(st.accuracy - c.meta.metrics_shipped.accuracy) < 1e-9 && st.wrong === 48, `wrong ${st.wrong}, P ${M.pct(st.precision, 2)}, R ${M.pct(st.recall, 2)}, F1 ${M.pct(st.f1, 2)}`);
  const wrong = c.samples.filter((s) => (s.p >= 0.5 ? 1 : 0) !== s.label);
  check("crack exactly one shipped sample is misclassified (s07)", wrong.length === 1 && wrong[0].misclassified && /s07/.test(wrong[0].file));
}

check("fmtProb extremes", M.fmtProb(0.99996) === ">99.9%" && M.fmtProb(0.00004) === "<0.1%" && M.fmtProb(0.5643) === "56.4%");
check("true minus sign", M.signed(-4.7, 1) === "−4.7" && M.fmt(-1234.5, 1) === "−1,234.5" && M.signed(0, 2) === "0.00");

console.log(failed ? `\n${failed} check(s) FAILED` : "\nall demo checks passed");
process.exit(failed ? 1 : 0);
