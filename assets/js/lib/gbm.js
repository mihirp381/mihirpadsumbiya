// sklearn GradientBoosting evaluator. Inputs are cast to float32 like sklearn; go left when x[f] <= t.
const fround = Math.fround;

export function predictRaw(model, x) {
  const x32 = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) x32[i] = fround(x[i]);
  const lr = model.lr;
  let s = model.init;
  for (const tr of model.trees) {
    const f = tr.f, t = tr.t, l = tr.l, r = tr.r;
    let n = 0;
    while (f[n] !== -1) n = x32[f[n]] <= t[n] ? l[n] : r[n];
    s += lr * tr.v[n];
  }
  return s;
}

export function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}
