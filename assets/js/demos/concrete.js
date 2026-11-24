import { predictRaw } from "../lib/gbm.js";
import { syncRange } from "../site.js";
import { chart, scaleLog, scaleLinear, ticks, niceMax, stepPath, axisX, gridY, text, line } from "../lib/plot.js";
import { loadJSON, refs, scheduler, sliderHTML, chipsHTML, pressChip, sampleHook, divergingRows } from "../lib/lab-ui.js";
import { fmt, signed, clamp, concreteDerived, outsideTraining, sensitivity, ageSweep, defaultMix, AGE_POS, posToAge, ageToPos } from "../lib/demo-math.js";

const GROUPS = [["Binder", [0, 1, 2]], ["Water and admixture", [3, 4]], ["Aggregates", [5, 6]]];
const AGE_TICKS = [1, 3, 7, 28, 90, 365];
const AGE = 7;
const REPO = "https://github.com/mihirp381/mihirpadsumbiya/blob/main/tools/train_concrete.py";

export async function mount(fig, hooks = {}) {
  const m = await loadJSON(new URL("../../models/concrete.json", import.meta.url));
  const { features: F, samples, meta, derived } = m;
  const f = (x) => predictRaw(m.model, x);
  const body = fig.querySelector(".fig-body");
  const ageMax = F[AGE].max;
  const valFmt = (i, v) => fmt(v, F[i].step < 1 ? 1 : 0, 2);

  body.innerHTML = `<div class="lab">
  <section class="lab-block" aria-labelledby="cc-samples-h">
    <div class="lab-h"><h3 id="cc-samples-h">Held-out lab mixes</h3><span class="lab-h-sub">test rows the model never saw</span></div>
    ${chipsHTML("chips", samples.map((_, i) => `Lab mix ${i + 1}`))}
    <p class="lab-line" data-ref="compare"></p>
  </section>
  <div class="lab-grid">
    <section class="lab-col lab-controls" aria-labelledby="cc-in-h">
      <div class="lab-h"><h3 id="cc-in-h">Mix, per cubic metre</h3><button type="button" class="btn btn-ghost btn-sm" data-ref="reset">Reset</button></div>
      ${GROUPS.map(([name, idx]) => `<div class="lab-group"><p class="lab-sub">${name}</p>${idx.map((i) => sliderHTML(`s${i}`, F[i].label, F[i])).join("")}</div>`).join("")}
      <div class="lab-group">
        ${sliderHTML("age", "Age", { min: 0, max: AGE_POS, step: 1 })}
        <div class="lab-ticks" aria-hidden="true">${AGE_TICKS.map((d) => `<span style="--at:${ageToPos(d, ageMax) / AGE_POS}">${d}</span>`).join("")}</div>
      </div>
    </section>
    <section class="lab-col lab-out" aria-label="Prediction">
      <div class="lab-readout" aria-live="polite">
        <p class="readout-label">Predicted compressive strength</p>
        <p class="lab-big"><span data-ref="mpa"></span><span class="lab-big-unit">MPa</span></p>
        <p class="lab-meta"><span data-ref="psi"></span> · ±${fmt(meta.metrics.test_rmse, 1)} MPa typical error on held-out mixes</p>
      </div>
      <dl class="kv">
        <div><dt>water / binder</dt><dd data-ref="wb"></dd></div>
        <div><dt>total mass</dt><dd data-ref="mass"></dd></div>
      </dl>
      <p class="note lab-warn" data-ref="warn">Outside the range of mixes the model was trained on. Treat this as extrapolation.</p>
    </section>
    <section class="lab-col lab-charts">
      <div class="lab-block">
        <div class="lab-h"><h3>Predicted strength by age</h3><span class="lab-h-sub">days, log scale</span></div>
        <div class="lab-plot" data-ref="age-plot"></div>
        <p class="note">The steps are real: tree ensembles predict piecewise-constant surfaces.</p>
      </div>
      <div class="lab-block">
        <div class="lab-h"><h3>What moves this mix</h3><span class="lab-key"><i class="sw sw-down"></i>down<i class="sw sw-up"></i>up</span></div>
        <ul class="dv" data-ref="sens"></ul>
        <p class="note">Change in predicted strength when one ingredient moves ±10% of its observed range.</p>
      </div>
    </section>
  </div>
</div>`;
  const r = refs(body);
  const inputs = F.map((_, i) => (i === AGE ? r.age : r[`s${i}`]));
  const outs = F.map((_, i) => (i === AGE ? r["age-val"] : r[`s${i}-val`]));
  const s = { x: F.map((ft) => ft.p50), sample: -1, edited: false, sweep: null, pred: 0 };
  s.x[AGE] = 28;

  const ageChart = chart(r["age-plot"], 210, (w, h) => {
    const ml = 36, mr = 16, mt = 20, mb = 26;
    const sample = s.sample >= 0 && !s.edited ? samples[s.sample] : null;
    const top = niceMax(Math.max(...s.sweep, sample ? sample.y : 0, 10) * 1.04, 4);
    const x = scaleLog(1, ageMax, ml, w - mr);
    const y = scaleLinear(0, top, h - mb, mt);
    const days = Array.from(s.sweep, (_, i) => x(i + 1));
    const age = s.x[AGE], cx = x(age), cy = y(s.pred);
    let out = gridY(y, ticks(0, top, 4), ml, w - mr, (v) => fmt(v)) + text(ml - 6, mt - 9, "MPa", "tk", "end");
    out += line(x(28), mt - 4, x(28), h - mb, "guide") + text(x(28) + 4, mt - 9, "28 d", "tk");
    out += axisX(x, AGE_TICKS, h - mb);
    out += `<path class="ln ln-a" d="${stepPath(days, Array.from(s.sweep, y))}"/>`;
    out += `<circle class="pt pt-a" cx="${cx}" cy="${cy}" r="4"/>` + text(cx - 7, cy - 7, fmt(s.pred, 1), "val", "end");
    if (sample) {
      const lx = x(sample.x[AGE]), ly = y(sample.y), right = lx < w - mr - 30;
      out += `<circle class="pt pt-b" cx="${lx}" cy="${ly}" r="4"/>` + text(lx + (right ? 8 : -8), ly + 4, "lab", "val val-b", right ? "start" : "end");
    }
    return out;
  });

  const syncInputs = () => {
    F.forEach((ft, i) => {
      inputs[i].value = i === AGE ? ageToPos(s.x[i], ageMax) : clamp(s.x[i], ft.min, ft.max);
      syncRange(inputs[i]);
    });
  };

  const render = () => {
    const x = s.x;
    s.pred = f(x);
    s.sweep = ageSweep(f, x, AGE, ageMax);
    F.forEach((ft, i) => {
      const t = i === AGE ? `${x[i]} ${x[i] === 1 ? "day" : "days"}` : `${valFmt(i, x[i])} ${ft.unit}`;
      outs[i].textContent = t;
      inputs[i].setAttribute("aria-valuetext", t);
    });
    r.mpa.textContent = fmt(s.pred, 1);
    r.psi.textContent = `≈ ${fmt(Math.round((s.pred * 145.038) / 10) * 10)} psi`;
    const d = concreteDerived(x);
    r.wb.textContent = d.binder > 0 ? fmt(d.wb, 2) : "n/a";
    r.mass.textContent = `${fmt(d.mass)} kg/m³`;
    r.warn.dataset.on = String(outsideTraining(d, derived));

    if (s.sample < 0) r.compare.textContent = "Pick a held-out mix to compare the model with the lab.";
    else if (s.edited) r.compare.textContent = `Lab mix ${s.sample + 1} has been edited, so its lab measurement no longer applies.`;
    else {
      const y = samples[s.sample].y;
      r.compare.textContent = `Measured in the lab: ${fmt(y, 1)} MPa · model: ${fmt(s.pred, 1)} MPa · error ${signed(s.pred - y, 1)} MPa`;
    }

    const sens = sensitivity(f, x, F, [0, 1, 2, 3, 4, 5, 6]);
    const max = Math.max(1, ...sens.flatMap((e) => [Math.abs(e.down), Math.abs(e.up)]));
    r.sens.innerHTML = divergingRows(sens.map((e) => ({
      name: F[e.i].label,
      sub: `±${fmt(e.h, e.h < 10 ? 1 : 0)} ${F[e.i].unit}`,
      bars: [{ v: e.down, cls: "dv-down", label: signed(e.down, 1) }, { v: e.up, cls: "dv-up", label: signed(e.up, 1) }],
    })), max);

    ageChart.svg.setAttribute("aria-label", `Predicted strength of this mix from 1 to ${ageMax} days; ${fmt(s.pred, 1)} MPa at ${x[AGE]} days.`);
    ageChart.render();
  };
  const schedule = scheduler(render);

  const edit = (i, v) => {
    s.x[i] = v;
    if (s.sample >= 0 && !s.edited) {
      s.edited = true;
      pressChip(r.chips, -1);
    }
    schedule();
  };
  F.forEach((ft, i) => inputs[i].addEventListener("input", () => edit(i, i === AGE ? posToAge(+inputs[i].value, ageMax) : +inputs[i].value)));
  r.age.addEventListener("keydown", (e) => {
    const k = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key];
    if (!k) return;
    e.preventDefault();
    edit(AGE, clamp(s.x[AGE] + k, 1, ageMax));
    r.age.value = ageToPos(s.x[AGE], ageMax);
    syncRange(r.age);
  });

  const load = (k) => {
    s.x = samples[k].x.slice();
    s.sample = k;
    s.edited = false;
    pressChip(r.chips, k);
    syncInputs();
    schedule();
  };
  r.chips.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b) load(+b.dataset.i);
  });
  r.reset.addEventListener("click", () => {
    s.x = F.map((ft) => ft.p50);
    s.x[AGE] = 28;
    s.sample = -1;
    s.edited = false;
    pressChip(r.chips, -1);
    syncInputs();
    schedule();
  });

  const hp = meta.hyperparameters, mt = meta.metrics;
  fig.querySelector(".fig-cap").innerHTML =
    `<span class="fig-num">Fig. 1</span>Gradient-boosted trees (${fmt(meta.trees)} trees, depth ${hp.max_depth}) trained on ${fmt(meta.split.train)} lab mixes and tested once on ${fmt(meta.split.test)} held-out mixes: ` +
    `R² ${fmt(mt.test_r2, 2)}, RMSE ${fmt(mt.test_rmse, 1)} MPa. In my manuscript the best of thirty models, a neural network, reached R² 0.93 and RMSE 4.9 MPa on its own split. ` +
    `Data: Yeh (1998), UCI Machine Learning Repository, CC BY 4.0. Reproduce with <a href="${REPO}" target="_blank" rel="noopener">tools/train_concrete.py</a>.`;

  const k = sampleHook(hooks, samples.length);
  load(k >= 0 ? k : defaultMix(samples, AGE));
  render();
}
