import { HERO, DATASETS, makeData, createNet, lossAndGrad, trainStep, paramCount, mulberry32 } from "../assets/js/lib/mlp.js";

let failed = false;

{
  const net = createNet({ hidden: 5, seed: 7 });
  const rand = mulberry32(11);
  const X = [], y = [];
  for (let i = 0; i < 9; i++) {
    X.push(2 * rand() - 1, 2 * rand() - 1);
    y.push(i % 2);
  }
  for (let i = 0; i < net.p.length; i++) net.p[i] += 0.3 * (2 * rand() - 1);
  lossAndGrad(net, X, y);
  const analytic = Float64Array.from(net.g);
  const h = 1e-5;
  let worst = 0;
  for (let i = 0; i < net.p.length; i++) {
    const p0 = net.p[i];
    net.p[i] = p0 + h;
    const lp = lossAndGrad(net, X, y).loss;
    net.p[i] = p0 - h;
    const lm = lossAndGrad(net, X, y).loss;
    net.p[i] = p0;
    const num = (lp - lm) / (2 * h);
    const rel = Math.abs(num - analytic[i]) / Math.max(Math.abs(num) + Math.abs(analytic[i]), 1e-8);
    worst = Math.max(worst, rel);
  }
  const ok = worst < 1e-5;
  if (!ok) failed = true;
  console.log(`${ok ? "PASS" : "FAIL"} gradient check: ${net.p.length} params, max relative error ${worst.toExponential(2)} (tol 1e-5)`);
}

const SEEDS = 20;
const median = (a) => {
  const s = [...a].sort((p, q) => p - q);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};
console.log(`H=${HERO.hidden} (${paramCount(createNet())} params), lr=${HERO.lr}, MAX_STEPS=${HERO.maxSteps}, spiral turns=${HERO.spiralTurns}`);
for (const name of DATASETS) {
  const to97 = [], toDone = [];
  let misses = 0, worstAcc = 1;
  const t0 = performance.now();
  for (let seed = 1; seed <= SEEDS; seed++) {
    const { X, y } = makeData(name, seed);
    const net = createNet({ seed });
    let hit = 0, done = 0, acc = 0;
    for (let step = 1; step <= HERO.maxSteps; step++) {
      const r = trainStep(net, X, y, HERO.lr);
      acc = r.acc;
      if (!hit && r.acc >= 0.97) hit = step;
      if (r.acc >= 0.99 && r.loss < 0.05) { done = step; break; }
    }
    if (!hit) { misses++; failed = true; }
    worstAcc = Math.min(worstAcc, acc);
    to97.push(hit || Infinity);
    if (done) toDone.push(done);
  }
  const ms = performance.now() - t0;
  console.log(
    `${misses ? "FAIL" : "PASS"} ${name.padEnd(6)} steps to 97%: median ${median(to97)}, worst ${Math.max(...to97)}; ` +
      `converged (acc>=99%, loss<0.05) ${toDone.length}/${SEEDS}, median ${toDone.length ? median(toDone) : "-"} steps, worst final acc ${(worstAcc * 100).toFixed(1)}%; ${ms.toFixed(0)} ms`
  );
}
process.exit(failed ? 1 : 0);
