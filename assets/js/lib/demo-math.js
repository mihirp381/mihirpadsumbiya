const formats = new Map();
const nf = (min, max, sign) => {
  const key = `${min}|${max}|${sign}`;
  if (!formats.has(key)) formats.set(key, new Intl.NumberFormat("en-US", { minimumFractionDigits: min, maximumFractionDigits: max, signDisplay: sign }));
  return formats.get(key);
};
const minus = (s) => s.replace(/-/g, "−");

export const fmt = (v, d = 0, max = d) => minus(nf(d, max, "negative").format(v));
export const signed = (v, d = 2) => minus(nf(d, d, "exceptZero").format(v));
export const pct = (v, d = 1) => `${fmt(100 * v, d)}%`;
export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export function money(v) {
  const a = Math.abs(v);
  return fmt(v, 0, a < 10 ? 2 : a < 100 ? 1 : 0);
}

export function roundMoney(v) {
  const a = Math.abs(v);
  const k = a < 10 ? 100 : a < 100 ? 10 : 1;
  return Math.round(v * k) / k;
}

export function fmtProb(p) {
  const v = Math.round(p * 1000) / 10;
  if (v > 99.9) return ">99.9%";
  if (v < 0.1) return "<0.1%";
  return `${fmt(v, 1)}%`;
}

export const AGE_POS = 1000;
export const posToAge = (pos, max = 365) => Math.round(max ** (pos / AGE_POS));
export const ageToPos = (d, max = 365) => Math.round((Math.log(d) / Math.log(max)) * AGE_POS);

export function concreteDerived(x) {
  const binder = x[0] + x[1] + x[2];
  let mass = 0;
  for (let i = 0; i < 7; i++) mass += x[i];
  return { binder, wb: binder > 0 ? x[3] / binder : NaN, mass };
}

export function outsideTraining(d, ranges) {
  const out = (v, r) => !(v >= r.p01 && v <= r.p99);
  return !(d.binder > 0) || out(d.wb, ranges.water_binder) || out(d.mass, ranges.total_mass);
}

export function sensitivity(f, x, features, idx, frac = 0.1) {
  const base = f(x);
  const at = (i, v) => {
    const y = x.slice();
    y[i] = v;
    return f(y) - base;
  };
  return idx
    .map((i) => {
      const ft = features[i];
      const h = frac * (ft.p99 - ft.p01);
      return { i, h, down: at(i, clamp(x[i] - h, ft.min, ft.max)), up: at(i, clamp(x[i] + h, ft.min, ft.max)) };
    })
    .sort((a, b) => Math.max(Math.abs(b.down), Math.abs(b.up)) - Math.max(Math.abs(a.down), Math.abs(a.up)) || a.i - b.i);
}

export function ageSweep(f, x, ageIdx, maxAge = 365) {
  const y = x.slice();
  const out = new Float64Array(maxAge);
  for (let d = 1; d <= maxAge; d++) {
    y[ageIdx] = d;
    out[d - 1] = f(y);
  }
  return out;
}

export function defaultMix(samples, ageIdx, age = 28) {
  const all = samples.map((_, i) => i);
  const at = all.filter((i) => samples[i].x[ageIdx] === age);
  const order = (at.length ? at : all).map((i) => [Math.abs(samples[i].pred - samples[i].y), i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return order[Math.floor((order.length - 1) / 2)][1];
}

export const caughtChurner = (samples, t) => samples.findIndex((s) => s.churn === 1 && s.p >= t);

export function drivers(raw, x, baseline) {
  const r0 = raw(x);
  return x
    .map((v, j) => {
      const y = x.slice();
      y[j] = baseline[j];
      return { j, value: v, delta: r0 - raw(y) };
    })
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta) || a.j - b.j);
}

export function thresholdStats([t, tp, fp, fn, tn]) {
  const flagged = tp + fp;
  return {
    t, tp, fp, fn, tn, flagged,
    n: tp + fp + fn + tn,
    recall: tp + fn ? tp / (tp + fn) : NaN,
    precision: flagged ? tp / flagged : null,
  };
}

export function thresholdSentence(s) {
  const head = `At ${s.t.toFixed(2)}: `;
  const n = fmt(s.n);
  if (!s.flagged) return `${head}flags nobody of ${n} held-out customers, catches ${pct(s.recall)} of the churners, and precision is n/a.`;
  return `${head}flags ${fmt(s.flagged)} of ${n} held-out customers, catches ${pct(s.recall)} of the churners, and ${pct(s.precision)} of the flags are real.`;
}

export function outcomeText(churned, flagged) {
  const truth = churned ? "This customer churned." : "This customer stayed.";
  const call = churned ? (flagged ? "Flagged: caught." : "Not flagged: missed.") : flagged ? "Flagged: false alarm." : "Not flagged: correct.";
  return `${truth} ${call}`;
}

export function compoundIndex(rows) {
  return rows.map((r, i) => ({ i, name: r[0], lower: r[0].toLowerCase(), act: r[1], nrefs: r[3] }));
}

export function searchCompounds(index, query, filter = "all", limit = 50) {
  const q = query.trim().toLowerCase();
  const want = filter === "inhibitor" ? 1 : filter === "substrate" ? 0 : null;
  const hits = [];
  for (const c of index) {
    if (want !== null && c.act !== want) continue;
    if (!q) {
      hits.push([2, c]);
      continue;
    }
    const k = c.lower.indexOf(q);
    if (k >= 0) hits.push([c.lower === q ? 0 : k === 0 ? 1 : 2, c]);
  }
  hits.sort((a, b) => a[0] - b[0] || b[1].nrefs - a[1].nrefs || a[1].i - b[1].i);
  return { n: hits.length, items: hits.slice(0, limit).map((h) => h[1]) };
}

export function cnnBlocks(tensors, size = 128) {
  const p = Object.fromEntries(tensors.map((t) => [t.name, t.shape.reduce((a, b) => a * b, 1)]));
  const params = (name) => p[`${name}.weight`] + p[`${name}.bias`];
  const ch = tensors.find((t) => t.name === "conv1.weight").shape[0];
  const hid = tensors.find((t) => t.name === "fc1.weight").shape[0];
  const c1 = size - 2, p1 = Math.floor(c1 / 4), c2 = p1 - 2, p2 = Math.floor(c2 / 4);
  const dim = (n) => `${n}×${n}×${ch}`;
  return [
    { name: "input", shape: `${size}×${size}×3`, stage: null },
    { name: `conv 3×3, ${ch} filters`, shape: dim(c1), params: params("conv1"), stage: "conv1" },
    { name: "max-pool 4×4", shape: dim(p1), stage: "conv1" },
    { name: `conv 3×3, ${ch} filters`, shape: dim(c2), params: params("conv2"), stage: "conv2" },
    { name: "max-pool 4×4", shape: dim(p2), stage: "conv2" },
    { name: `dense ${hid}`, params: params("fc1"), stage: "dense" },
    { name: "sigmoid", params: params("fc2"), stage: "dense" },
  ];
}

export function channelMean(a, cells, ch) {
  const out = new Float32Array(cells);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < cells; i++) {
    let s = 0;
    for (let c = 0; c < ch; c++) s += a[i * ch + c];
    out[i] = s / ch;
    if (out[i] < lo) lo = out[i];
    if (out[i] > hi) hi = out[i];
  }
  return normalize(out, lo, hi);
}

export function normalize(a, lo, hi) {
  if (lo === undefined) {
    lo = Infinity;
    hi = -Infinity;
    for (const v of a) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  const span = hi - lo;
  const out = new Float32Array(a.length);
  if (span > 0) for (let i = 0; i < a.length; i++) out[i] = (a[i] - lo) / span;
  return out;
}

export function confusionStats({ tp, fp, fn, tn }) {
  const precision = tp / (tp + fp), recall = tp / (tp + fn);
  return { precision, recall, f1: (2 * precision * recall) / (precision + recall), accuracy: (tp + tn) / (tp + fp + fn + tn), wrong: fp + fn, n: tp + fp + fn + tn };
}

export function squareCrop(w, h) {
  const s = Math.min(w, h);
  return { sx: (w - s) / 2, sy: (h - s) / 2, s };
}
