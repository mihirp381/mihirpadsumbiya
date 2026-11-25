import { predictRaw, sigmoid } from "../lib/gbm.js";
import { syncRange } from "../site.js";
import { chart, scaleAsinh, scaleLinear, axisX, axisY, gridY, linePath, text, line } from "../lib/plot.js";
import { loadJSON, refs, scheduler, sliderHTML, chipsHTML, pressChip, sampleHook, divergingRows, scaleHTML, place, edgeOf } from "../lib/lab-ui.js";
import { fmt, signed, pct, clamp, money, roundMoney, drivers, thresholdStats, thresholdSentence, outcomeText, caughtChurner } from "../lib/demo-math.js";

const POS = 200;
const DAYS = 6;
const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
const REPO = "https://github.com/mihirp381/mihirpadsumbiya/blob/main/tools/train_churn.py";

export async function mount(fig, hooks = {}) {
  const m = await loadJSON(new URL("../../models/churn.json", import.meta.url));
  const { features: F, samples, meta, baseline, thresholds, calibration } = m;
  const raw = (x) => predictRaw(m.model, x);
  const body = fig.querySelector(".fig-body");
  const scales = F.map((ft, i) => (i === DAYS ? null : scaleAsinh(ft.p50, 0, ft.p99, 0, POS)));
  const valText = (i, v) => (i === DAYS ? `${fmt(v)} ${v === 1 ? "day" : "days"}` : money(v));
  const base = meta.metrics.test_base_rate;
  const names = samples.map((_, i) => `Customer ${String.fromCharCode(65 + i)}`);

  body.innerHTML = `<div class="lab">
  <section class="lab-block" aria-labelledby="ch-samples-h">
    <div class="lab-h"><h3 id="ch-samples-h">Held-out customers</h3><span class="lab-h-sub">true outcome known, never seen in training</span></div>
    ${chipsHTML("chips", names)}
    <p class="lab-line" data-ref="outcome"></p>
  </section>
  <div class="lab-grid">
    <section class="lab-col lab-controls" aria-labelledby="ch-in-h">
      <div class="lab-h"><h3 id="ch-in-h">Account</h3><button type="button" class="btn btn-ghost btn-sm" data-ref="reset">Reset to the median customer</button></div>
      <div class="lab-group">${F.map((ft, i) => sliderHTML(`s${i}`, ft.label, i === DAYS ? { min: 0, max: 365, step: 1 } : { min: 0, max: POS, step: 1 })).join("")}</div>
      <p class="note">Money is in account currency units; the dataset does not name a currency. Money sliders run from 0 to the 99th percentile on an asinh scale.</p>
    </section>
    <section class="lab-col lab-out" aria-label="Prediction">
      <div class="lab-readout" aria-live="polite">
        <p class="readout-label">Probability this customer churns</p>
        <p class="lab-big"><span data-ref="p"></span></p>
        <p class="lab-verdict" data-ref="verdict"></p>
      </div>
      ${scaleHTML([{ at: base, cls: "scale-m-base", label: `base rate ${pct(base)}` }, { cls: "scale-m-t", tick: "ttick", lbl: "tlbl" }])}
    </section>
    <section class="lab-col lab-charts">
      <div class="lab-block">
        <div class="lab-h"><h3>Why this score</h3><span class="lab-key"><i class="sw sw-b"></i>raises risk<i class="sw sw-a"></i>lowers</span></div>
        <ul class="dv" data-ref="drv"></ul>
        <p class="note">Change in log-odds when one input is swapped for the median customer's value. A local what-if, not SHAP; the bars do not sum to the score.</p>
      </div>
    </section>
  </div>
  <div class="lab-split">
    <section class="lab-block" aria-labelledby="ch-thr-h">
      <div class="lab-h"><h3 id="ch-thr-h">Decision threshold</h3><span class="lab-h-sub">held-out customers</span></div>
      ${sliderHTML("t", "Flag when P(churn) ≥", { min: 0, max: thresholds.length - 1, step: 1 })}
      <p class="lab-line" data-ref="tsent" aria-live="polite"></p>
      <div class="lab-thr">
        <table class="cm">
          <caption class="sr">Confusion matrix at the current threshold</caption>
          <thead><tr><th scope="col"></th><th scope="col">flagged</th><th scope="col">not flagged</th></tr></thead>
          <tbody>
            <tr><th scope="row">churned</th><td class="cm-hit" data-ref="tp"></td><td data-ref="fn"></td></tr>
            <tr><th scope="row">stayed</th><td data-ref="fp"></td><td class="cm-ok" data-ref="tn"></td></tr>
          </tbody>
        </table>
        <div class="lab-plot" data-ref="pr"></div>
      </div>
    </section>
    <section class="lab-block" aria-labelledby="ch-cal-h">
      <div class="lab-h"><h3 id="ch-cal-h">Calibration on held-out customers</h3></div>
      <div class="lab-plot lab-plot-sq" data-ref="cal"></div>
      <p class="note">${calibration.length} equal-count bins of ${fmt(meta.split.test)} customers. Points on the diagonal mean the stated probability matches how often those customers churned. Brier score ${fmt(meta.metrics.test_brier, 3)}.</p>
    </section>
  </div>
</div>`;
  const r = refs(body);
  const inputs = F.map((_, i) => r[`s${i}`]);
  const stats = thresholds.map(thresholdStats);
  const s = { x: baseline.slice(), sample: -1, edited: false, t: Math.max(0, stats.findIndex((e) => Math.abs(e.t - 0.3) < 1e-9)), p: 0 };

  const prChart = chart(r.pr, 196, (w, h) => {
    const ml = 40, mr = 12, mt = 24, mb = 42;
    const x = scaleLinear(0, 1, ml, w - mr), y = scaleLinear(0, 1, h - mb, mt);
    const rec = stats.map((e) => [x(e.t), y(e.recall)]);
    const pre = stats.filter((e) => e.precision !== null).map((e) => [x(e.t), y(e.precision)]);
    const cur = stats[s.t];
    let out = gridY(y, [0.5, 1], ml + 1, w - mr) + axisY(y, [0, 0.5, 1], ml, (v) => pct(v, 0)) + axisX(x, [0, 0.5, 1], h - mb, (v) => v.toFixed(1));
    out += text(w - mr, h - 3, "THRESHOLD", "ttl", "end");
    out += `<path class="ln ln-dash" d="${linePath(rec)}"/><path class="ln" d="${linePath(pre)}"/>`;
    const k = stats.findIndex((e) => e.t >= 0.72);
    out += text(x(stats[k].t) + 4, y(stats[k].recall) - 8, "recall", "val") + text(x(stats[k].t) - 4, y(stats[k].precision) - 8, "precision", "val", "end");
    const e = edgeOf(cur.t);
    out += line(x(cur.t), mt - 6, x(cur.t), h - mb, "guide") + text(x(cur.t) + (e === "r" ? 4 : e === "l" ? -4 : 0), mt - 12, `t ${cur.t.toFixed(2)}`, "tk", e === "r" ? "end" : e === "l" ? "start" : "middle");
    out += `<circle class="pt" cx="${x(cur.t)}" cy="${y(cur.recall)}" r="3.5"/>`;
    if (cur.precision !== null) out += `<circle class="pt" cx="${x(cur.t)}" cy="${y(cur.precision)}" r="3.5"/>`;
    return out;
  }, "Recall and precision on held-out customers against the decision threshold.");

  chart(r.cal, (w) => Math.min(w, 300), (w, h) => {
    const ml = 44, mr = 12, mt = 24, mb = 42;
    const side = Math.min(w - ml - mr, h - mt - mb);
    const x = scaleLinear(0, 1, ml, ml + side), y = scaleLinear(0, 1, mt + side, mt);
    const pts = calibration.map(([p, o]) => [x(p), y(o)]);
    let out = gridY(y, [0.5, 1], ml + 1, ml + side) + axisY(y, [0, 0.5, 1], ml, (v) => pct(v, 0)) + axisX(x, [0, 0.5, 1], mt + side, (v) => pct(v, 0));
    out += line(x(0), y(0), x(1), y(1), "guide");
    out += text(ml + side, mt + side + 39, "MEAN PREDICTED", "ttl", "end") + text(ml - 8, mt - 12, "OBSERVED", "ttl");
    out += `<path class="ln" d="${linePath(pts)}"/>` + pts.map(([cx, cy]) => `<circle class="pt-o" cx="${cx}" cy="${cy}" r="3.5"/>`).join("");
    return out;
  }, `Calibration: ${calibration.map(([p, o]) => `predicted ${pct(p)}, observed ${pct(o)}`).join("; ")}.`);

  const syncInputs = () => {
    F.forEach((ft, i) => {
      inputs[i].value = i === DAYS ? clamp(s.x[i], 0, 365) : clamp(Math.round(scales[i](s.x[i])), 0, POS);
      syncRange(inputs[i]);
    });
  };
  const syncThreshold = () => {
    r.t.value = s.t;
    syncRange(r.t);
  };

  const render = () => {
    const x = s.x, st = stats[s.t];
    const z = raw(x);
    s.p = sigmoid(z);
    const flagged = s.p >= st.t;
    F.forEach((ft, i) => {
      const t = valText(i, x[i]);
      r[`s${i}-val`].textContent = t;
      inputs[i].setAttribute("aria-valuetext", t);
    });
    r.p.textContent = pct(s.p);
    r.verdict.textContent = flagged ? "Flag for retention" : "No action";
    r.verdict.dataset.v = flagged ? "b" : "a";
    place(r.fill, s.p);
    r.fill.dataset.v = r.verdict.dataset.v;
    r["fill-v"].textContent = pct(s.p);
    place(r.ttick, st.t);
    r.tlbl.textContent = `threshold ${pct(st.t, 0)}`;
    place(r.tlbl, st.t);

    if (s.sample < 0) r.outcome.textContent = "Pick a held-out customer to see what really happened.";
    else if (s.edited) r.outcome.textContent = `${names[s.sample]} has been edited, so the real outcome no longer applies.`;
    else r.outcome.textContent = outcomeText(samples[s.sample].churn === 1, flagged);

    const drv = drivers(raw, x, baseline);
    const max = Math.max(0.25, ...drv.map((d) => Math.abs(d.delta)));
    r.drv.innerHTML = divergingRows(drv.map((d) => ({
      name: F[d.j].label,
      sub: valText(d.j, d.value),
      bars: [{ v: d.delta, cls: d.delta > 0 ? "dv-pos" : "dv-neg", label: signed(d.delta, 2) }],
    })), max);

    r["t-val"].textContent = st.t.toFixed(2);
    r.t.setAttribute("aria-valuetext", st.t.toFixed(2));
    r.tsent.textContent = thresholdSentence(st);
    for (const k of ["tp", "fp", "fn", "tn"]) r[k].textContent = fmt(st[k]);
    prChart.render();
  };
  const schedule = scheduler(render);

  F.forEach((ft, i) =>
    inputs[i].addEventListener("input", () => {
      s.x[i] = i === DAYS ? +inputs[i].value : roundMoney(scales[i].invert(+inputs[i].value));
      if (s.sample >= 0 && !s.edited) {
        s.edited = true;
        pressChip(r.chips, -1);
      }
      schedule();
    })
  );
  r.t.addEventListener("input", () => {
    s.t = +r.t.value;
    schedule();
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
    s.x = baseline.slice();
    s.sample = -1;
    s.edited = false;
    pressChip(r.chips, -1);
    syncInputs();
    schedule();
  });

  const hp = meta.hyperparameters, mt = meta.metrics;
  fig.querySelector(".fig-cap").innerHTML =
    `<span class="fig-num">Figure 3.</span>A compact refit of ChurnGuard, my team's winning entry at UTD FinHack 2024, sized for the browser: gradient-boosted trees (${fmt(meta.trees)} trees, depth ${hp.max_depth}) ` +
    `on ${WORDS[F.length] || F.length} account features, trained on ${fmt(meta.split.train)} customers and tested once on ${fmt(meta.split.test)} at the natural ${pct(meta.base_rate)} churn rate. ` +
    `ROC-AUC ${fmt(mt.test_roc_auc, 2)}, PR-AUC ${fmt(mt.test_pr_auc_average_precision, 2)}. No resampling, so the probabilities are calibrated. The hackathon model was XGBoost on ten features. ` +
    `Reproduce with <a href="${REPO}" target="_blank" rel="noopener">tools/train_churn.py</a>.`;

  const hook = sampleHook(hooks, samples.length);
  const k = hook >= 0 ? hook : caughtChurner(samples, stats[s.t].t);
  if (k >= 0) load(k);
  else syncInputs();
  syncThreshold();
  render();
}
