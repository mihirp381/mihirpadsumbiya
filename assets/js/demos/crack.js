import { loadJSON, refs, cssVar, hexRgb, onTheme, sampleHook, scaleHTML, place } from "../lib/lab-ui.js";
import { fmt, fmtProb, pct, cnnBlocks, channelMean, normalize, confusionStats, squareCrop } from "../lib/demo-math.js";

const S = 128, G = 29, MAP = 126;
const PAPER = [["96.32%", "97.80%"], ["97.93%", "97.94%"], ["98.32%", "98.32%"], ["98.59%", "98.46%"], ["98.9%", "98.3%"]];
const DOI = "https://doi.org/10.22115/scce.2022.325596.1397";

function createEngine(onFallback) {
  const pending = new Map();
  let seq = 0, worker = null, local = null;
  const runLocal = async (job) => {
    local ||= import("./crack-worker.js");
    const eng = await local;
    await new Promise((r) => requestAnimationFrame(() => setTimeout(r)));
    return { ...(await eng.run(job.rgb, job.onStage)), engine: "main" };
  };
  const fallback = () => {
    worker?.terminate();
    worker = null;
    onFallback();
    for (const [id, job] of pending) {
      pending.delete(id);
      runLocal(job).then(job.resolve, job.reject);
    }
  };
  try {
    worker = new Worker(new URL("./crack-worker.js", import.meta.url), { type: "module" });
    worker.onmessage = ({ data }) => {
      const job = pending.get(data.id);
      if (!job) return;
      if (data.type === "stage") return job.onStage(data.stage);
      pending.delete(data.id);
      if (data.type === "result") job.resolve({ ...data, engine: "worker" });
      else job.reject(new Error(data.message));
    };
    worker.onerror = (e) => {
      e.preventDefault();
      fallback();
    };
  } catch {
    worker = null;
  }
  return {
    get inWorker() {
      return !!worker;
    },
    run: (rgb, onStage) =>
      new Promise((resolve, reject) => {
        const job = { rgb, onStage, resolve, reject };
        if (!worker) return runLocal(job).then(resolve, reject);
        const id = ++seq;
        pending.set(id, job);
        worker.postMessage({ id, rgb });
      }),
  };
}

export async function mount(fig, hooks = {}) {
  const meta = await loadJSON(new URL("../../models/crack-cnn.json", import.meta.url));
  const M = meta.meta, conf = M.metrics_shipped.confusion, cs = confusionStats(conf);
  const samples = meta.samples;
  const blocks = cnnBlocks(meta.tensors, S);
  const mb = fmt(M.bin_bytes / 1e6, 2);
  const size = (t) => t.shape.reduce((a, b) => a * b, 1);
  const q8 = pct(meta.tensors.filter((t) => t.dtype === "int8").reduce((a, t) => a + size(t), 0) / meta.tensors.reduce((a, t) => a + size(t), 0));
  const body = fig.querySelector(".fig-body");
  const asset = (p) => new URL(`../../../${p}`, import.meta.url).href;

  body.innerHTML = `<div class="lab">
  <div class="lab-grid">
    <section class="lab-col lab-controls" aria-labelledby="ck-pick-h">
      <div class="lab-h"><h3 id="ck-pick-h">Test images</h3><span class="lab-h-sub">held out from training</span></div>
      <div class="crack-picks" data-ref="picks">${samples.map((e, i) => `<button type="button" class="crack-pick" data-i="${i}" aria-pressed="false"><img src="${asset(e.file)}" alt="Sample ${i + 1}, concrete surface" width="${S}" height="${S}"></button>`).join("")}</div>
      <div class="crack-upload">
        <button type="button" class="btn btn-ghost btn-sm" data-ref="upload">Use your own photo</button>
        <input type="file" accept="image/*" data-ref="file" hidden>
      </div>
      <p class="note">Your photo never leaves this device. The network has only ever seen close-ups of concrete like these, so anything else is out of distribution.</p>
    </section>
    <section class="lab-col lab-out" aria-label="Prediction">
      <div class="crack-stage">
        <div class="crack-view" data-ref="view"><canvas data-ref="base"></canvas><canvas data-ref="ov"></canvas></div>
        <div class="crack-result">
          <div class="lab-readout" aria-live="polite">
            <p class="readout-label" data-ref="what">Sample 1</p>
            <p class="lab-big lab-word" data-ref="verdict">&nbsp;</p>
            <p class="lab-meta">P(crack) <span data-ref="p">&nbsp;</span></p>
          </div>
          ${scaleHTML()}
          <p class="note" data-ref="status">Loading weights, ${mb} MB</p>
          <p class="lab-line" data-ref="truth"></p>
          <p class="note" data-ref="own" hidden>This network has only two answers and no way to say that a photo is not concrete. Away from close-ups like the test images, ignore its confidence.</p>
          <p class="note" data-ref="kept" hidden></p>
        </div>
      </div>
      <label class="lab-check"><input type="checkbox" data-ref="toggle" checked> Activation map</label>
      <p class="note">Mean activation of the second convolutional layer. It shows where the network responds, not a formal attribution.</p>
    </section>
  </div>
  <section class="lab-block" aria-labelledby="ck-arch-h">
    <div class="lab-h"><h3 id="ck-arch-h">Architecture</h3><span class="lab-h-sub">${fmt(M.params)} parameters</span></div>
    <ol class="arch" data-ref="arch">${blocks.map((b) => `<li class="arch-b" data-stage="${b.stage || ""}"><span class="arch-name">${b.name}</span>${b.shape ? `<span class="arch-dim">${b.shape}</span>` : ""}${b.params ? `<span class="arch-dim">${fmt(b.params)} params</span>` : ""}</li>`).join("")}</ol>
  </section>
  <section class="lab-block" aria-labelledby="ck-maps-h">
    <div class="lab-h"><h3 id="ck-maps-h">Eight most active of 128 first-layer filters</h3></div>
    <div class="crack-maps" data-ref="maps">${Array.from({ length: 8 }, (_, i) => `<figure class="crack-map"><canvas width="${MAP}" height="${MAP}"></canvas><figcaption data-map="${i}"></figcaption></figure>`).join("")}</div>
  </section>
  <details class="lab-details">
    <summary>Training log and test results</summary>
    <div class="lab-details-body">
      <div class="lab-scroll"><table class="lab-table">
        <caption class="sr">Accuracy per epoch, this retraining against the 2022 paper</caption>
        <thead><tr><th scope="col">epoch</th><th scope="col">train, this run</th><th scope="col">validation, this run</th><th scope="col">train, paper</th><th scope="col">validation, paper</th></tr></thead>
        <tbody>${M.epochs.map((e, i) => `<tr><th scope="row">${e.epoch}</th><td>${pct(e.train_acc_running, 2)}</td><td>${pct(e.val_acc, 2)}</td><td>${PAPER[i]?.[0] ?? "n/a"}</td><td>${PAPER[i]?.[1] ?? "n/a"}</td></tr>`).join("")}</tbody>
      </table></div>
      <p class="note">Training accuracy is the running mean over each epoch, the same convention Keras used for the paper's numbers. Validation uses ${fmt(M.split.val)} images.</p>
      <div class="lab-thr">
        <table class="cm">
          <caption class="sr">Confusion matrix on the test images</caption>
          <thead><tr><th scope="col"></th><th scope="col">predicted crack</th><th scope="col">predicted no crack</th></tr></thead>
          <tbody>
            <tr><th scope="row">crack</th><td class="cm-hit">${fmt(conf.tp)}</td><td>${fmt(conf.fn)}</td></tr>
            <tr><th scope="row">no crack</th><td>${fmt(conf.fp)}</td><td class="cm-ok">${fmt(conf.tn)}</td></tr>
          </tbody>
        </table>
        <dl class="kv">
          <div><dt>precision</dt><dd>${pct(cs.precision, 2)}</dd></div>
          <div><dt>recall</dt><dd>${pct(cs.recall, 2)}</dd></div>
          <div><dt>F1</dt><dd>${pct(cs.f1, 2)}</dd></div>
          <div><dt>accuracy</dt><dd>${pct(cs.accuracy, 2)} of ${fmt(cs.n)}</dd></div>
        </dl>
      </div>
    </div>
  </details>
</div>`;
  const r = refs(body);
  const picks = [...r.picks.children];
  const archBlocks = [...r.arch.children];
  const mapCanvases = [...r.maps.querySelectorAll("canvas")];
  const src = document.createElement("canvas");
  src.width = src.height = S;
  const sctx = src.getContext("2d", { willReadFrequently: true });
  const s = { item: null, map: null, maps: null, channels: null, latest: 0, warm: false };

  const engine = createEngine(() => caption());
  const caption = () => {
    const sp = M.split;
    fig.querySelector(".fig-cap").innerHTML =
      `<span class="fig-num">Figure 3.</span>The four-layer CNN from my 2022 paper (${fmt(M.params)} parameters), retrained in PyTorch on the same ${fmt(sp.train + sp.val + sp.test)} images with the same split sizes and hyperparameters: ` +
      `${pct(M.metrics_shipped.accuracy)} accuracy on ${fmt(M.metrics_shipped.n)} held-out images (the paper reported 97.8%). ${q8} of the weights are stored as 8-bit integers (${mb} MB) and the forward pass is hand-written JavaScript${engine.inWorker ? " running in a Web Worker" : ""}. ` +
      `Images: Özgenel (2019), Mendeley Data, CC BY 4.0. Paper: <a href="${DOI}" target="_blank" rel="noopener">doi:10.22115/SCCE.2022.325596.1397</a>.`;
  };
  caption();

  const stage = (name) => archBlocks.forEach((b) => b.toggleAttribute("data-on", !!name && b.dataset.stage === name));

  const drawBase = () => {
    const c = r.base, ctx = c.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(src, 0, 0, c.width, c.height);
  };

  const drawOverlay = () => {
    const c = r.ov, W = c.width, ctx = c.getContext("2d");
    ctx.clearRect(0, 0, W, W);
    if (!s.map || !W) return;
    const [R, Gc, B] = hexRgb(cssVar("--ox"));
    const img = ctx.createImageData(W, W), d = img.data;
    const i0 = new Int32Array(W), fr = new Float32Array(W);
    for (let p = 0; p < W; p++) {
      const g = (((p + 0.5) * S) / W - 7) / 4;
      const gc = Math.min(Math.max(g, 0), G - 1);
      i0[p] = Math.min(Math.floor(gc), G - 2);
      fr[p] = gc - i0[p];
    }
    const m = s.map;
    for (let y = 0; y < W; y++) {
      const ya = i0[y], fy = fr[y];
      for (let x = 0; x < W; x++) {
        const xa = i0[x], fx = fr[x];
        const top = m[ya * G + xa] * (1 - fx) + m[ya * G + xa + 1] * fx;
        const bot = m[(ya + 1) * G + xa] * (1 - fx) + m[(ya + 1) * G + xa + 1] * fx;
        const j = 4 * (y * W + x);
        d[j] = R;
        d[j + 1] = Gc;
        d[j + 2] = B;
        d[j + 3] = 153 * (top * (1 - fy) + bot * fy);
      }
    }
    ctx.putImageData(img, 0, 0);
  };

  const drawMaps = () => {
    if (!s.maps) return;
    const bg = hexRgb(cssVar("--paper")), a = hexRgb(cssVar("--blue"));
    s.maps.forEach((map, k) => {
      const ctx = mapCanvases[k].getContext("2d");
      const n = normalize(map), img = ctx.createImageData(MAP, MAP), d = img.data;
      for (let i = 0; i < n.length; i++) {
        const v = n[i], j = 4 * i;
        d[j] = bg[0] + (a[0] - bg[0]) * v;
        d[j + 1] = bg[1] + (a[1] - bg[1]) * v;
        d[j + 2] = bg[2] + (a[2] - bg[2]) * v;
        d[j + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
      r.maps.querySelector(`[data-map="${k}"]`).textContent = `filter ${s.channels[k]}`;
    });
  };

  new ResizeObserver(() => {
    const dpr = Math.min(devicePixelRatio || 1, 2.5);
    const w = Math.round(r.view.clientWidth * dpr);
    if (!w || w === r.base.width) return;
    for (const c of [r.base, r.ov]) c.width = c.height = w;
    drawBase();
    drawOverlay();
  }).observe(r.view);
  onTheme(() => {
    drawOverlay();
    drawMaps();
  });
  r.toggle.addEventListener("change", () => (r.ov.hidden = !r.toggle.checked));

  const readRGB = () => {
    const d = sctx.getImageData(0, 0, S, S).data, rgb = new Uint8Array(S * S * 3);
    for (let i = 0, j = 0; i < d.length; i += 4, j += 3) {
      rgb[j] = d[i];
      rgb[j + 1] = d[i + 1];
      rgb[j + 2] = d[i + 2];
    }
    return rgb;
  };

  const classify = async (item) => {
    const id = ++s.latest;
    s.item = item;
    s.map = null;
    drawBase();
    drawOverlay();
    r.what.textContent = item.name;
    if (s.warm) r.status.textContent = engine.inWorker ? "Running in a Web Worker" : "Running on the main thread";
    r.truth.textContent = "";
    r.kept.hidden = true;
    r.own.hidden = true;
    const res = await engine.run(readRGB(), (name) => id === s.latest && stage(name));
    if (id !== s.latest) return;
    s.warm = true;
    stage(null);
    fig.dataset.engine = res.engine;
    s.map = channelMean(res.conv2, G * G, res.conv2.length / (G * G));
    s.maps = res.maps;
    s.channels = res.channels;
    drawOverlay();
    drawMaps();
    const crack = res.p > 0.5;
    r.verdict.textContent = crack ? "Crack" : "No crack";
    r.verdict.dataset.v = crack ? "b" : "a";
    r.fill.dataset.v = r.verdict.dataset.v;
    r.p.textContent = fmtProb(res.p);
    place(r.fill, res.p);
    r["fill-v"].textContent = fmtProb(res.p);
    r.status.textContent = `${fmt(res.ms)} ms on your device`;
    if (item.sample) {
      const truth = item.sample.label === 1;
      r.truth.textContent = `Dataset label: ${truth ? "crack" : "no crack"}. ${truth === crack ? "The model agrees." : "The model is wrong here."}`;
      if (item.sample.misclassified) {
        r.kept.textContent = `This one is kept on purpose: the dataset labels it intact and the model is confident it is cracked. ${fmt(cs.wrong)} of the ${fmt(cs.n)} test images end up on the wrong side.`;
        r.kept.hidden = false;
      }
    } else {
      r.truth.textContent = "Your photo has no dataset label.";
      r.own.hidden = false;
    }
  };

  const fail = (msg) => {
    s.latest++;
    stage(null);
    r.status.textContent = msg;
  };

  const pickSample = async (k) => {
    picks.forEach((b, i) => b.setAttribute("aria-pressed", String(i === k)));
    const img = picks[k].querySelector("img");
    await img.decode();
    sctx.clearRect(0, 0, S, S);
    sctx.drawImage(img, 0, 0, S, S);
    return classify({ name: `Sample ${k + 1}`, sample: samples[k] });
  };

  const pickFile = async (file) => {
    if (!file) return;
    let bmp;
    try {
      bmp = await createImageBitmap(file);
    } catch {
      return fail("This file could not be read as an image.");
    }
    picks.forEach((b) => b.setAttribute("aria-pressed", "false"));
    const { sx, sy, s: side } = squareCrop(bmp.width, bmp.height);
    sctx.fillStyle = "#fff";
    sctx.fillRect(0, 0, S, S);
    sctx.imageSmoothingEnabled = true;
    sctx.imageSmoothingQuality = "high";
    sctx.drawImage(bmp, sx, sy, side, side, 0, 0, S, S);
    bmp.close();
    return classify({ name: "Your photo", sample: null });
  };

  const guard = (p) => p.catch((err) => {
    console.error(err);
    fail("This image could not be processed.");
  });
  r.picks.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b) guard(pickSample(+b.dataset.i));
  });
  r.upload.addEventListener("click", () => r.file.click());
  r.file.addEventListener("change", () => {
    guard(pickFile(r.file.files[0]));
    r.file.value = "";
  });
  r.view.addEventListener("dragover", (e) => {
    e.preventDefault();
    r.view.dataset.drag = "";
  });
  r.view.addEventListener("dragleave", () => delete r.view.dataset.drag);
  r.view.addEventListener("drop", (e) => {
    e.preventDefault();
    delete r.view.dataset.drag;
    guard(pickFile(e.dataTransfer.files[0]));
  });

  const k = sampleHook(hooks, samples.length);
  await pickSample(Math.max(k, 0));
}
