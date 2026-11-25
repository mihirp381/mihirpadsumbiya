import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { predictRaw, sigmoid } from "../assets/js/lib/gbm.js";
import { createModel, forward } from "../assets/js/lib/crack-cnn.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const json = (p) => JSON.parse(readFileSync(join(root, p), "utf8"));
let failed = false;
const report = (name, err, tol, extra = "") => {
  const ok = err <= tol;
  if (!ok) failed = true;
  console.log(`${ok ? "PASS" : "FAIL"} ${name}: max |d| = ${err.toExponential(3)} (tol ${tol})${extra}`);
};

const fx = json("tools/fixtures/gbm_parity.json");
for (const key of ["concrete", "churn"]) {
  const m = json(`assets/models/${key}.json`);
  const f = fx[key];
  let err = 0, errOut = 0;
  const t0 = performance.now();
  f.x.forEach((x, i) => {
    const raw = predictRaw(m.model, x);
    err = Math.max(err, Math.abs(raw - f.raw[i]));
    errOut = Math.max(errOut, Math.abs(key === "churn" ? sigmoid(raw) - f.p[i] : raw - f.pred[i]));
  });
  const us = ((performance.now() - t0) / f.x.length) * 1000;
  report(`${key} raw score (${f.x.length} rows)`, err, 1e-9, `; output max |d| = ${errOut.toExponential(3)}; ${us.toFixed(1)} us/row`);
  let errS = 0;
  for (const s of m.samples) {
    const raw = predictRaw(m.model, s.x);
    errS = Math.max(errS, Math.abs(key === "churn" ? sigmoid(raw) - s.p : raw - s.pred));
  }
  report(`${key} shipped samples (${m.samples.length})`, errS, 1e-9);
}

const meta = json("assets/models/crack-cnn.json");
const buf = readFileSync(join(root, "assets/models/crack-cnn.bin"));
let t0 = performance.now();
const cnn = createModel(meta, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
console.log(`crack-cnn createModel ${(performance.now() - t0).toFixed(1)} ms`);
const fc = json("tools/fixtures/crack_parity.json");
let errP = 0;
const times = [];
for (const im of fc.images) {
  const rgb = new Uint8Array(Buffer.from(im.rgb, "base64"));
  const stages = [];
  const out = forward(cnn, rgb, (s) => stages.push(s));
  times.push(out.ms);
  errP = Math.max(errP, Math.abs(out.p - im.p));
  const shapeOk = stages.join() === "conv1,conv2,dense" && out.conv2.length === 29 * 29 * 128 &&
    out.conv1Top.channels.length === 8 && out.conv1Top.maps.every((m) => m.length === 126 * 126);
  if (!shapeOk) { failed = true; console.log("FAIL crack-cnn output shape/stages"); }
  console.log(`  ${im.source} label=${im.label} p_js=${out.p.toFixed(6)} p_py=${im.p.toFixed(6)} ${out.ms.toFixed(0)} ms`);
}
const warm = times.slice(1);
report(`crack-cnn p (${fc.images.length} images)`, errP, 1e-4,
  `; forward first ${times[0].toFixed(0)} ms, warm mean ${(warm.reduce((a, b) => a + b, 0) / warm.length).toFixed(0)} ms, max ${Math.max(...times).toFixed(0)} ms`);

process.exit(failed ? 1 : 0);
