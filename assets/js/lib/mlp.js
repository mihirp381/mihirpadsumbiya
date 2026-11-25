export const HERO = { hidden: 24, lr: 0.05, maxSteps: 4000, spiralTurns: 1.75 };
export const DATASETS = ["spiral", "moons", "rings", "xor"];

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rand) {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

const clamp = (v) => Math.max(-1, Math.min(1, v));

const GEN = {
  spiral(rand, i, n, c) {
    const t = (i + rand()) / n;
    const r = 0.12 + 0.83 * t;
    const a = 2 * Math.PI * HERO.spiralTurns * t + c * Math.PI;
    return [r * Math.cos(a) + 0.035 * gauss(rand), r * Math.sin(a) + 0.035 * gauss(rand)];
  },
  moons(rand, i, n, c) {
    const a = Math.PI * rand();
    const x = c ? 1 - Math.cos(a) : Math.cos(a);
    const y = c ? 0.5 - Math.sin(a) : Math.sin(a);
    return [(x - 0.5) * 0.64 + 0.06 * gauss(rand), (y - 0.25) * 0.64 + 0.06 * gauss(rand)];
  },
  rings(rand, i, n, c) {
    const r = c ? 0.68 + 0.26 * rand() : 0.42 * Math.sqrt(rand());
    const a = 2 * Math.PI * rand();
    return [r * Math.cos(a) + 0.025 * gauss(rand), r * Math.sin(a) + 0.025 * gauss(rand)];
  },
  xor(rand, i, n, c) {
    const sx = rand() < 0.5 ? -1 : 1;
    const sy = c ? -sx : sx;
    return [0.5 * sx + 0.16 * gauss(rand), 0.5 * sy + 0.16 * gauss(rand)];
  },
};

export function makeData(name, seed, n = 100) {
  const rand = mulberry32((seed ^ 0x9e3779b9) >>> 0);
  const X = [];
  const y = [];
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < n; i++) {
      const [a, b] = GEN[name](rand, i, n, c);
      X.push(clamp(a), clamp(b));
      y.push(c);
    }
  }
  return { X, y };
}

export function createNet({ hidden = HERO.hidden, seed = 1 } = {}) {
  const H = hidden;
  const o = { W1: 0, b1: 2 * H, W2: 3 * H, b2: 3 * H + H * H, W3: 4 * H + H * H, b3: 5 * H + H * H };
  const n = o.b3 + 1;
  const p = new Float64Array(n);
  const rand = mulberry32(seed);
  const init = (off, len, fanIn, fanOut) => {
    const lim = Math.sqrt(6 / (fanIn + fanOut));
    for (let i = 0; i < len; i++) p[off + i] = (2 * rand() - 1) * lim;
  };
  init(o.W1, 2 * H, 2, H);
  init(o.W2, H * H, H, H);
  init(o.W3, H, H, 1);
  return {
    H, o, p,
    g: new Float64Array(n), m: new Float64Array(n), v: new Float64Array(n), t: 0,
    a1: new Float64Array(H), a2: new Float64Array(H), d2: new Float64Array(H),
  };
}

export const paramCount = (net) => net.p.length;
export const sigmoid = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

function logit(net, x1, x2) {
  const { H, o, p, a1, a2 } = net;
  for (let i = 0; i < H; i++) a1[i] = Math.tanh(p[o.W1 + 2 * i] * x1 + p[o.W1 + 2 * i + 1] * x2 + p[o.b1 + i]);
  let z = p[o.b3];
  for (let i = 0; i < H; i++) {
    let s = p[o.b2 + i];
    const r = o.W2 + i * H;
    for (let j = 0; j < H; j++) s += p[r + j] * a1[j];
    a2[i] = Math.tanh(s);
    z += p[o.W3 + i] * a2[i];
  }
  return z;
}

export function lossAndGrad(net, X, y) {
  const { H, o, p, g, a1, a2, d2 } = net;
  const N = y.length;
  g.fill(0);
  let loss = 0;
  let correct = 0;
  for (let s = 0; s < N; s++) {
    const x1 = X[2 * s], x2 = X[2 * s + 1], t = y[s];
    const z = logit(net, x1, x2);
    loss += Math.max(z, 0) - t * z + Math.log1p(Math.exp(-Math.abs(z)));
    if (z > 0 === (t === 1)) correct++;
    const dz = (sigmoid(z) - t) / N;
    g[o.b3] += dz;
    for (let i = 0; i < H; i++) {
      g[o.W3 + i] += dz * a2[i];
      d2[i] = dz * p[o.W3 + i] * (1 - a2[i] * a2[i]);
      g[o.b2 + i] += d2[i];
      const r = o.W2 + i * H;
      for (let j = 0; j < H; j++) g[r + j] += d2[i] * a1[j];
    }
    for (let j = 0; j < H; j++) {
      let d = 0;
      for (let i = 0; i < H; i++) d += p[o.W2 + i * H + j] * d2[i];
      d *= 1 - a1[j] * a1[j];
      g[o.W1 + 2 * j] += d * x1;
      g[o.W1 + 2 * j + 1] += d * x2;
      g[o.b1 + j] += d;
    }
  }
  return { loss: loss / N, acc: correct / N };
}

export function adam(net, lr, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
  const { p, g, m, v } = net;
  net.t++;
  const c1 = 1 - b1 ** net.t;
  const c2 = 1 - b2 ** net.t;
  for (let i = 0; i < p.length; i++) {
    m[i] = b1 * m[i] + (1 - b1) * g[i];
    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
    p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
  }
}

export function trainStep(net, X, y, lr = HERO.lr) {
  const r = lossAndGrad(net, X, y);
  adam(net, lr);
  return r;
}

export const predict = (net, x1, x2) => sigmoid(logit(net, x1, x2));

export function predictGrid(net, cols, rows, out, [x0, x1, y0, y1] = [-1, 1, -1, 1]) {
  for (let r = 0; r < rows; r++) {
    const yy = y1 - ((y1 - y0) * r) / (rows - 1);
    for (let c = 0; c < cols; c++) out[r * cols + c] = predict(net, x0 + ((x1 - x0) * c) / (cols - 1), yy);
  }
  return out;
}
