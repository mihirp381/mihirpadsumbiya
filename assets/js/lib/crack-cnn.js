// Crack-detection CNN (Padsumbiya et al., 2022) in plain JS. All activations are HWC.
const S = 128, C1 = 126, P1 = 31, C2 = 29, P2 = 7, CH = 128, HID = 128;

const now = () => (globalThis.performance ? performance.now() : Date.now());

function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

function readTensor(t, buf) {
  const n = t.shape.reduce((a, b) => a * b, 1);
  const out = new Float32Array(n);
  if (t.dtype === "float32") {
    out.set(new Float32Array(buf, t.offset, n));
  } else if (t.dtype === "int8") {
    const q = new Int8Array(buf, t.offset, n);
    const scales = (Array.isArray(t.scale) ? t.scale : [t.scale]).map(Math.fround);
    const per = n / scales.length;
    for (let i = 0; i < n; i++) out[i] = q[i] * scales[(i / per) | 0];
  } else if (t.dtype === "float16") {
    const h = new Uint16Array(buf, t.offset, n);
    for (let i = 0; i < n; i++) out[i] = halfToFloat(h[i]);
  } else {
    throw new Error("unsupported dtype " + t.dtype);
  }
  return out;
}

export function createModel(metaJson, arrayBuffer) {
  const w = {};
  for (const t of metaJson.tensors) w[t.name] = readTensor(t, arrayBuffer);
  return {
    w,
    x: new Float32Array(S * S * 3),
    a1: new Float32Array(C1 * C1 * CH),
    p1: new Float32Array(P1 * P1 * CH),
    patch: new Float32Array(9 * CH),
  };
}

function maxPool4(src, n, out, m) {
  for (let y = 0; y < m; y++)
    for (let x = 0; x < m; x++) {
      const o = (y * m + x) * CH;
      for (let c = 0; c < CH; c++) out[o + c] = -Infinity;
      for (let dy = 0; dy < 4; dy++)
        for (let dx = 0; dx < 4; dx++) {
          const s = ((y * 4 + dy) * n + x * 4 + dx) * CH;
          for (let c = 0; c < CH; c++) if (src[s + c] > out[o + c]) out[o + c] = src[s + c];
        }
    }
}

function conv3x3(src, n, cin, wt, bias, out, patch) {
  const m = n - 2, row = 3 * cin, k = 3 * row;
  for (let y = 0; y < m; y++)
    for (let x = 0; x < m; x++) {
      for (let ky = 0; ky < 3; ky++) {
        const s = ((y + ky) * n + x) * cin;
        for (let j = 0; j < row; j++) patch[ky * row + j] = src[s + j];
      }
      const o = (y * m + x) * CH;
      for (let c = 0; c < CH; c++) {
        const wo = c * k;
        let acc = bias[c];
        for (let j = 0; j < k; j++) acc += patch[j] * wt[wo + j];
        out[o + c] = acc > 0 ? acc : 0;
      }
    }
}

export function forward(model, rgb, onStage) {
  const t0 = now(), w = model.w, x = model.x;
  if (rgb.length !== S * S * 3) throw new Error("expected 128*128*3 RGB bytes");
  for (let i = 0; i < x.length; i++) x[i] = rgb[i] / 255;

  if (onStage) onStage("conv1");
  conv3x3(x, S, 3, w["conv1.weight"], w["conv1.bias"], model.a1, model.patch);
  const a1 = model.a1, mean = new Float64Array(CH);
  for (let i = 0; i < a1.length; i += CH) for (let c = 0; c < CH; c++) mean[c] += a1[i + c];
  const channels = Array.from(mean.keys()).sort((p, q) => mean[q] - mean[p] || p - q).slice(0, 8);
  const maps = channels.map((c) => {
    const m = new Float32Array(C1 * C1);
    for (let i = 0; i < m.length; i++) m[i] = a1[i * CH + c];
    return m;
  });
  maxPool4(a1, C1, model.p1, P1);

  if (onStage) onStage("conv2");
  const conv2 = new Float32Array(C2 * C2 * CH);
  conv3x3(model.p1, P1, CH, w["conv2.weight"], w["conv2.bias"], conv2, model.patch);
  const p2 = new Float32Array(P2 * P2 * CH);
  maxPool4(conv2, C2, p2, P2);

  if (onStage) onStage("dense");
  const W1 = w["fc1.weight"], b1 = w["fc1.bias"], W2 = w["fc2.weight"], n = p2.length;
  let z = w["fc2.bias"][0];
  for (let h = 0; h < HID; h++) {
    let acc = b1[h];
    const o = h * n;
    for (let j = 0; j < n; j++) acc += p2[j] * W1[o + j];
    acc = Math.fround(acc);
    if (acc > 0) z += acc * W2[h];
  }
  const p = 1 / (1 + Math.exp(-z));
  return { p, ms: now() - t0, conv2, conv1Top: { channels, maps } };
}
