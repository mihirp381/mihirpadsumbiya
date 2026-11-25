export function scaleLinear(d0, d1, r0, r1) {
  const k = (r1 - r0) / (d1 - d0 || 1);
  const f = (v) => r0 + (v - d0) * k;
  f.invert = (y) => d0 + (y - r0) / k;
  return f;
}

export function scaleLog(d0, d1, r0, r1) {
  const l = scaleLinear(Math.log(d0), Math.log(d1), r0, r1);
  const f = (v) => l(Math.log(v));
  f.invert = (y) => Math.exp(l.invert(y));
  return f;
}

export function scaleAsinh(s, d0, d1, r0, r1) {
  const l = scaleLinear(Math.asinh(d0 / s), Math.asinh(d1 / s), r0, r1);
  const f = (v) => l(Math.asinh(v / s));
  f.invert = (y) => s * Math.sinh(l.invert(y));
  return f;
}

export function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}

export function ticks(lo, hi, count) {
  const step = niceStep(hi - lo, count);
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export function niceMax(v, count) {
  const step = niceStep(v, count);
  return Math.ceil(v / step - 1e-9) * step;
}

const r = (v) => Math.round(v * 10) / 10;

export function linePath(pts) {
  return pts.map(([x, y], i) => `${i ? "L" : "M"}${r(x)} ${r(y)}`).join("");
}

export function stepPath(xs, ys) {
  let d = `M${r(xs[0])} ${r(ys[0])}`;
  for (let i = 1; i < xs.length; i++) d += `H${r(xs[i])}V${r(ys[i])}`;
  return d;
}

export const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

export function text(x, y, s, cls = "", anchor = "start") {
  return `<text x="${r(x)}" y="${r(y)}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`;
}

export function line(x1, y1, x2, y2, cls) {
  return `<line x1="${r(x1)}" y1="${r(y1)}" x2="${r(x2)}" y2="${r(y2)}" class="${cls}"/>`;
}

export function axisX(x, values, y, label = String) {
  return line(x(values[0]), y, x(values.at(-1)), y, "ax") +
    values.map((v) => line(x(v), y, x(v), y + 5, "ax") + text(x(v), y + 18, label(v), "tk", "middle")).join("");
}

export function axisY(y, values, x, label = String) {
  return line(x, y(values[0]), x, y(values.at(-1)), "ax") +
    values.map((v) => line(x - 5, y(v), x, y(v), "ax") + text(x - 8, y(v) + 4, label(v), "tk", "end")).join("");
}

export function gridY(y, values, x0, x1) {
  return values.map((v) => line(x0, y(v), x1, y(v), "grid")).join("");
}

export function chart(host, height, draw, label) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "plot-svg");
  svg.setAttribute("role", "img");
  if (label) svg.setAttribute("aria-label", label);
  host.append(svg);
  let w = 0;
  const render = () => {
    if (!w) return;
    const h = typeof height === "function" ? height(w) : height;
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    svg.setAttribute("width", w);
    svg.setAttribute("height", h);
    svg.innerHTML = draw(w, h);
  };
  new ResizeObserver(([e]) => {
    const nw = Math.floor(e.contentRect.width);
    if (nw !== w) {
      w = nw;
      render();
    }
  }).observe(host);
  return { svg, render };
}
