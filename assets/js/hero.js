import { HERO, makeData, createNet, trainStep, predict, predictGrid, paramCount } from "./lib/mlp.js";
import { syncRange } from "./site.js";

const COLS = 96;
const ROWS = 64;
const EXT = [-1.725, 1.725, -1.15, 1.15];
const SPEEDS = [0.5, 1, 2, 4, 8];
const RATE = 22;
const RAMP = 400;
const MAX_STEPS_PER_FRAME = 40;
const DRAW_EVERY_MS = 33;
const PITCH = 7;
const RMAX = 0.29 * PITCH;

const fig = document.getElementById("fig-0");
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

const sx = (x, w) => ((x - EXT[0]) / (EXT[1] - EXT[0])) * w;
const sy = (y, h) => ((EXT[3] - y) / (EXT[3] - EXT[2])) * h;

if (fig) init();

function init() {
  const plot = fig.querySelector("[data-hero-plot]");
  const canvas = plot.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const hud = Object.fromEntries([...fig.querySelectorAll("[data-hud]")].map((el) => [el.dataset.hud, el]));
  const pauseBtn = fig.querySelector("[data-hero-pause]");
  const speed = fig.querySelector("[data-hero-speed]");
  const speedVal = fig.querySelector("[data-hero-speed-val]");
  const setButtons = [...fig.querySelectorAll("[data-set]")];
  const classButtons = [...fig.querySelectorAll("[data-class]")];

  const grid = new Float32Array(COLS * ROWS);
  let screen = null;

  const s = {
    name: "spiral", seed: 8, X: [], y: [], net: null,
    step: 0, since: 0, loss: NaN, acc: NaN, hist: [],
    status: "training", addClass: 0, speed: 1,
    onScreen: true, budget: 0, last: 0, lastDraw: 0, raf: 0, settle: 0,
  };
  let colors = null;

  for (const v of [-1, 0, 1]) {
    const label = v < 0 ? `−${-v}` : String(v);
    plot.insertAdjacentHTML(
      "beforeend",
      `<span class="tick tick-x" aria-hidden="true" style="left:${(sx(v, 100)).toFixed(3)}%">${label}</span>` +
        `<span class="tick tick-y" aria-hidden="true" style="top:${(sy(v, 100)).toFixed(3)}%">${label}</span>`
    );
  }
  plot.insertAdjacentHTML("beforeend", '<span class="axis-name axis-x" aria-hidden="true">x<sub>1</sub></span><span class="axis-name axis-y" aria-hidden="true">x<sub>2</sub></span>');

  const probe = createNet();
  fig.querySelector("[data-hero-cap]").textContent =
    `Not a video. A 2→${probe.H}→${probe.H}→1 tanh network (${paramCount(probe).toLocaleString("en-US")} parameters) learning a decision boundary ` +
    "by gradient descent with Adam, in your browser. Hand-written JavaScript, no libraries. Click the plot to add data.";

  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    const v = (n) => cs.getPropertyValue(n).trim();
    colors = { paper: v("--paper"), blue: v("--blue"), ox: v("--ox"), ink: v("--ink") };
  }

  function buildScreen(W, H, dpr) {
    const a = (PITCH * dpr) / Math.SQRT2;
    const xs = [], ys = [];
    for (let j = 0, y = a / 2; y < H + a; j++, y += a) {
      for (let x = (j % 2) * a; x < W + a; x += 2 * a) {
        xs.push(x);
        ys.push(y);
      }
    }
    const n = xs.length;
    const idx = new Int32Array(n), fx = new Float32Array(n), fy = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      const gx = Math.min(Math.max((xs[k] / W) * (COLS - 1), 0), COLS - 1.001);
      const gy = Math.min(Math.max((ys[k] / H) * (ROWS - 1), 0), ROWS - 1.001);
      const c = Math.floor(gx), r = Math.floor(gy);
      idx[k] = r * COLS + c;
      fx[k] = gx - c;
      fy[k] = gy - r;
    }
    screen = { W, H, dpr, n, x: Float32Array.from(xs), y: Float32Array.from(ys), idx, fx, fy };
  }

  function paintScreen() {
    predictGrid(s.net, COLS, ROWS, grid, EXT);
    const { n, x, y, idx, fx, fy, dpr } = screen;
    const rmax = RMAX * dpr, rmin = 0.3 * dpr;
    const lo = new Path2D(), hi = new Path2D();
    for (let k = 0; k < n; k++) {
      const i = idx[k], u = fx[k], v = fy[k];
      const p = (grid[i] * (1 - u) + grid[i + 1] * u) * (1 - v) + (grid[i + COLS] * (1 - u) + grid[i + COLS + 1] * u) * v;
      const r = rmax * Math.sqrt(Math.abs(2 * p - 1));
      if (r < rmin) continue;
      if (p > 0.5) {
        const h = r * 0.886;
        hi.rect(x[k] - h, y[k] - h, 2 * h, 2 * h);
      } else {
        lo.moveTo(x[k] + r, y[k]);
        lo.arc(x[k], y[k], r, 0, 2 * Math.PI);
      }
    }
    ctx.fillStyle = colors.blue;
    ctx.fill(lo);
    ctx.fillStyle = colors.ox;
    ctx.fill(hi);
  }

  function contour(cw, ch) {
    const f = (u, v) => (0.5 - u) / (v - u);
    ctx.beginPath();
    for (let r = 0; r < ROWS - 1; r++) {
      for (let c = 0; c < COLS - 1; c++) {
        const i = r * COLS + c;
        const tl = grid[i], tr = grid[i + 1], br = grid[i + COLS + 1], bl = grid[i + COLS];
        const code = ((tl > 0.5) << 3) | ((tr > 0.5) << 2) | ((br > 0.5) << 1) | (bl > 0.5);
        if (code === 0 || code === 15) continue;
        const x0 = c * cw, y0 = r * ch;
        const T = () => [x0 + cw * f(tl, tr), y0];
        const R = () => [x0 + cw, y0 + ch * f(tr, br)];
        const B = () => [x0 + cw * f(bl, br), y0 + ch];
        const L = () => [x0, y0 + ch * f(tl, bl)];
        const seg = (p, q) => {
          ctx.moveTo(p[0], p[1]);
          ctx.lineTo(q[0], q[1]);
        };
        const mid = (tl + tr + br + bl) / 4 > 0.5;
        switch (code) {
          case 1: case 14: seg(L(), B()); break;
          case 2: case 13: seg(B(), R()); break;
          case 3: case 12: seg(L(), R()); break;
          case 4: case 11: seg(T(), R()); break;
          case 6: case 9: seg(T(), B()); break;
          case 7: case 8: seg(L(), T()); break;
          case 5:
            if (mid) { seg(L(), T()); seg(B(), R()); } else { seg(T(), R()); seg(L(), B()); }
            break;
          case 10:
            if (mid) { seg(T(), R()); seg(L(), B()); } else { seg(L(), T()); seg(B(), R()); }
            break;
        }
      }
    }
    ctx.stroke();
  }

  function points(dpr) {
    const W = canvas.width, H = canvas.height;
    const rad = (W / dpr < 420 ? 3.7 : 4.5) * dpr;
    const half = rad * 0.89;
    const groups = [[], [], [], []];
    for (let i = 0; i < s.y.length; i++) {
      const x = s.X[2 * i], y = s.X[2 * i + 1], t = s.y[i];
      const wrong = predict(s.net, x, y) > 0.5 !== (t === 1);
      groups[t + 2 * wrong].push(sx(x, W), sy(y, H));
    }
    const path = (g, square) => {
      ctx.beginPath();
      for (let i = 0; i < g.length; i += 2) {
        if (square) ctx.rect(g[i] - half, g[i + 1] - half, 2 * half, 2 * half);
        else {
          ctx.moveTo(g[i] + rad, g[i + 1]);
          ctx.arc(g[i], g[i + 1], rad, 0, 2 * Math.PI);
        }
      }
    };
    ctx.lineJoin = "miter";
    for (const t of [0, 1]) {
      const col = t ? colors.ox : colors.blue;
      path(groups[t], t === 1);
      ctx.lineWidth = 7 * dpr;
      ctx.strokeStyle = colors.paper;
      ctx.stroke();
      ctx.fillStyle = col;
      ctx.fill();
      path(groups[t + 2], t === 1);
      ctx.lineWidth = 8.5 * dpr;
      ctx.stroke();
      ctx.fillStyle = colors.paper;
      ctx.fill();
      ctx.lineWidth = 1.5 * dpr;
      ctx.strokeStyle = col;
      ctx.stroke();
    }
  }

  function draw() {
    const W = canvas.width, H = canvas.height;
    if (!W || !H || !s.net) return;
    const dpr = W / Math.max(1, canvas.clientWidth);
    s.lastDraw = performance.now();
    if (!screen || screen.W !== W || screen.H !== H) buildScreen(W, H, dpr);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);
    paintScreen();
    const cw = W / (COLS - 1), ch = H / (ROWS - 1);
    ctx.lineWidth = 1.25 * dpr;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = colors.ink;
    contour(cw, ch);
    points(dpr);
    updateHud();
  }

  function updateHud() {
    hud.step.textContent = s.step.toLocaleString("en-US");
    hud.loss.textContent = Number.isFinite(s.loss) ? s.loss.toFixed(4) : "–";
    hud.acc.textContent = Number.isFinite(s.acc) ? (s.acc * 100).toFixed(1) : "–";
    const n = s.hist.length;
    if (n < 2) {
      hud.spark.setAttribute("d", "");
      return;
    }
    const m = Math.min(n, 100);
    const vals = [];
    for (let j = 0; j < m; j++) vals.push(Math.log10(s.hist[Math.round((j * (n - 1)) / (m - 1))]));
    const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
    hud.spark.setAttribute("d", vals.map((v, j) => `${j ? "L" : "M"}${((j / (m - 1)) * 100).toFixed(2)} ${(23 - ((v - lo) / span) * 22).toFixed(2)}`).join(""));
  }

  function setStatus(st) {
    s.status = st;
    hud.status.textContent = st;
    hud.status.dataset.state = st;
    const paused = st === "paused";
    pauseBtn.toggleAttribute("data-paused", paused);
    pauseBtn.setAttribute("aria-label", paused ? "Resume training" : "Pause training");
    pauseBtn.disabled = st === "converged" || st === "stopped";
    const acc = Number.isFinite(s.acc) ? `, training accuracy ${(s.acc * 100).toFixed(1)}%` : "";
    canvas.setAttribute(
      "aria-label",
      `Scatter plot of the ${s.name} dataset: ${s.y.length} points in two classes, blue circles for class 0 and red squares for class 1, ` +
        `over a dot screen of the network's predicted class with its decision boundary drawn as a line. Status ${st}${acc}.`
    );
  }

  function step() {
    const r = trainStep(s.net, s.X, s.y, HERO.lr);
    s.step++;
    s.since++;
    s.loss = r.loss;
    s.acc = r.acc;
    s.hist.push(r.loss);
    if (r.acc >= 0.99 && r.loss < 0.05) return "converged";
    if (s.since >= HERO.maxSteps) return "stopped";
    return null;
  }

  const animating = () => s.status === "training" && s.onScreen && !document.hidden && !reduceMotion.matches;

  function frame(t) {
    s.raf = 0;
    if (!animating()) return;
    const dt = s.last ? Math.min(t - s.last, 100) / 1000 : 0;
    s.last = t;
    s.budget = Math.min(s.budget + RATE * SPEEDS[s.speed] * (1 + s.since / RAMP) * dt, MAX_STEPS_PER_FRAME);
    let n = Math.floor(s.budget);
    s.budget -= n;
    let end = null;
    const stepped = n > 0;
    while (n-- > 0 && !end) end = step();
    if (end) {
      s.budget = 0;
      setStatus(end);
      draw();
      return;
    }
    if (stepped && t - s.lastDraw >= DRAW_EVERY_MS) draw();
    s.raf = requestAnimationFrame(frame);
  }

  function settle() {
    const token = ++s.settle;
    const chunk = () => {
      if (token !== s.settle) return;
      const t0 = performance.now();
      let end = null;
      while (!end && performance.now() - t0 < 25) end = step();
      if (end) {
        setStatus(end);
        draw();
      } else setTimeout(chunk, 0);
    };
    chunk();
  }

  function run() {
    if (reduceMotion.matches) {
      if (s.status === "training") return settle();
      return draw();
    }
    draw();
    if (animating() && !s.raf) {
      s.last = 0;
      s.raf = requestAnimationFrame(frame);
    }
  }

  function reset() {
    ({ X: s.X, y: s.y } = makeData(s.name, s.seed));
    s.net = createNet({ seed: s.seed });
    Object.assign(s, { step: 0, since: 0, loss: NaN, acc: NaN, hist: [], budget: 0 });
    s.settle++;
    setStatus("training");
    run();
  }

  function applyMotionMode() {
    const reduce = reduceMotion.matches;
    pauseBtn.hidden = reduce;
    fig.querySelector("[data-hero-speed-wrap]").hidden = reduce;
    if (reduce && s.status === "paused") setStatus("training");
  }

  const press = (buttons, active) => buttons.forEach((b) => b.setAttribute("aria-pressed", String(b === active)));

  setButtons.forEach((b) =>
    b.addEventListener("click", () => {
      press(setButtons, b);
      s.name = b.dataset.set;
      reset();
    })
  );
  classButtons.forEach((b) =>
    b.addEventListener("click", () => {
      press(classButtons, b);
      s.addClass = +b.dataset.class;
    })
  );
  fig.querySelector("[data-hero-retrain]").addEventListener("click", () => {
    s.seed = (Math.random() * 2 ** 31) >>> 0;
    reset();
  });
  pauseBtn.addEventListener("click", () => {
    if (s.status === "training") setStatus("paused");
    else if (s.status === "paused") setStatus("training");
    run();
  });
  speed.addEventListener("input", () => {
    s.speed = +speed.value;
    speedVal.textContent = `${SPEEDS[s.speed]}×`;
  });
  canvas.addEventListener("click", (e) => {
    const r = canvas.getBoundingClientRect();
    const x = EXT[0] + ((e.clientX - r.left) / r.width) * (EXT[1] - EXT[0]);
    const y = EXT[3] - ((e.clientY - r.top) / r.height) * (EXT[3] - EXT[2]);
    s.X.push(x, y);
    s.y.push(s.addClass);
    if (s.status === "converged" || s.status === "stopped") {
      s.since = 0;
      setStatus("training");
    }
    run();
  });

  new ResizeObserver(() => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const w = Math.round(plot.clientWidth * dpr), h = Math.round(plot.clientHeight * dpr);
    if (w === canvas.width && h === canvas.height) return;
    canvas.width = w;
    canvas.height = h;
    draw();
  }).observe(plot);
  new IntersectionObserver(([e]) => {
    s.onScreen = e.isIntersecting;
    run();
  }).observe(fig);
  document.addEventListener("visibilitychange", run);
  new MutationObserver(() => {
    readColors();
    draw();
  }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  reduceMotion.addEventListener("change", () => {
    applyMotionMode();
    run();
  });

  readColors();
  syncRange(speed);
  fig.querySelector("[data-hero-hud]").hidden = false;
  fig.querySelector("[data-hero-controls]").hidden = false;
  applyMotionMode();
  reset();
}
