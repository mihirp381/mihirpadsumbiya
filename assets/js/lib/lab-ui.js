import { esc } from "./plot.js";

export async function loadJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return res.json();
}

export function refs(root) {
  return Object.fromEntries([...root.querySelectorAll("[data-ref]")].map((el) => [el.dataset.ref, el]));
}

export function scheduler(fn) {
  let raf = 0;
  return () => {
    if (!raf) raf = requestAnimationFrame(() => {
      raf = 0;
      fn();
    });
  };
}

export function onTheme(fn) {
  new MutationObserver(fn).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function hexRgb(h) {
  h = h.replace("#", "");
  if (h.length === 3) h = [...h].map((c) => c + c).join("");
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function sliderHTML(ref, label, { min, max, step }) {
  return `<label class="range lab-range"><span class="range-label">${esc(label)}</span>` +
    `<output class="range-val" data-ref="${ref}-val"></output>` +
    `<input type="range" data-ref="${ref}" min="${min}" max="${max}" step="${step}"></label>`;
}

export function chipsHTML(ref, names) {
  return `<div class="lab-chips" data-ref="${ref}">${names.map((n, i) => `<button type="button" class="chip lab-chip" data-i="${i}" aria-pressed="false">${esc(n)}</button>`).join("")}</div>`;
}

export function pressChip(box, i) {
  box.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.i === i)));
}

export function sampleHook(hooks, count) {
  const n = parseInt(hooks.sample, 10);
  return n >= 1 && n <= count ? n - 1 : -1;
}

const at = (v) => `left:${(v * 100).toFixed(2)}%`;
export const edgeOf = (v) => (v < 0.12 ? "l" : v > 0.88 ? "r" : "");

export function place(el, v) {
  el.style.left = `${(v * 100).toFixed(2)}%`;
  el.dataset.edge = edgeOf(v);
}

export function scaleHTML(marks = []) {
  const tick = (i) => `<i class="scale-t${i % 5 ? "" : " scale-t-major"}" style="${at(i / 10)}"></i>`;
  const ref = (r) => (r ? ` data-ref="${r}"` : "");
  return `<div class="scale" aria-hidden="true">` +
    `<div class="scale-head"><span class="scale-ptr" data-ref="fill"><span class="scale-v" data-ref="fill-v"></span></span></div>` +
    `<div class="scale-rule">${Array.from({ length: 11 }, (_, i) => tick(i)).join("")}` +
    marks.map((m) => `<i class="scale-m ${m.cls}"${ref(m.tick)} style="${at(m.at ?? 0)}"></i>`).join("") + `</div>` +
    `<div class="scale-row">${[0, 0.5, 1].map((v) => `<span style="${at(v)}">${v}</span>`).join("")}</div>` +
    marks.map((m) => `<div class="scale-row"><span class="scale-lbl"${ref(m.lbl)} data-edge="${edgeOf(m.at ?? 0)}" style="${at(m.at ?? 0)}">${esc(m.label ?? "")}</span></div>`).join("") +
    `</div>`;
}

export function divergingRows(rows, max) {
  const w = (v) => `${(Math.min(Math.abs(v) / (max || 1), 1) * 38).toFixed(2)}%`;
  const bar = (v, cls, label) =>
    `<span class="dv-bar ${cls}" style="${v < 0 ? "right" : "left"}:50%;width:${w(v)}"></span>` +
    `<span class="dv-val" style="${v < 0 ? "right" : "left"}:calc(50% + ${w(v)} + 4px)">${esc(label)}</span>`;
  return rows
    .map((r) => `<li class="dv-row"><span class="dv-name">${esc(r.name)}${r.sub ? `<span class="dv-sub">${esc(r.sub)}</span>` : ""}</span>` +
      `<span class="dv-track">${r.bars.map((b) => `<span class="dv-lane">${bar(b.v, b.cls, b.label)}</span>`).join("")}</span></li>`)
    .join("");
}
