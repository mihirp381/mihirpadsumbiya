const root = document.documentElement;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

export function syncRange(input) {
  const min = input.min === "" ? 0 : +input.min;
  const max = input.max === "" ? 100 : +input.max;
  input.style.setProperty("--fill", `${((+input.value - min) / (max - min || 1)) * 100}%`);
}

function initTheme() {
  const btn = document.querySelector(".theme-toggle");
  const metas = document.querySelectorAll('meta[name="theme-color"]');
  const apply = (theme, save) => {
    root.dataset.theme = theme;
    btn.setAttribute("aria-pressed", String(theme === "dark"));
    const bg = getComputedStyle(root).getPropertyValue("--bg").trim();
    metas.forEach((m) => m.setAttribute("content", bg));
    if (save) {
      try { localStorage.setItem("theme", theme); } catch {}
    }
  };
  apply(root.dataset.theme === "light" ? "light" : "dark", false);
  btn.addEventListener("click", () => apply(root.dataset.theme === "dark" ? "light" : "dark", true));
}

function initNav() {
  const toggle = document.querySelector(".nav-toggle");
  const list = document.getElementById("nav-links");
  const isOpen = () => toggle.getAttribute("aria-expanded") === "true";
  const setOpen = (open) => {
    toggle.setAttribute("aria-expanded", String(open));
    list.classList.toggle("is-open", open);
  };
  toggle.addEventListener("click", () => setOpen(!isOpen()));
  list.addEventListener("click", (e) => {
    if (e.target.closest("a")) setOpen(false);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && isOpen()) {
      setOpen(false);
      toggle.focus();
    }
  });
  document.addEventListener("click", (e) => {
    if (isOpen() && !e.target.closest(".nav")) setOpen(false);
  });

  const links = new Map([...list.querySelectorAll("a")].map((a) => [a.hash.slice(1), a]));
  const alias = { recommendations: "about" };
  const mark = (id) => links.forEach((a, key) => (key === id ? a.setAttribute("aria-current", "true") : a.removeAttribute("aria-current")));
  const inBand = new Set();
  let atEnd = false;
  const update = () => {
    if (atEnd) return mark("contact");
    const ids = [...links.keys(), "recommendations"].filter((id) => inBand.has(id));
    mark(ids.length ? alias[ids[0]] || ids[0] : null);
  };
  const band = new IntersectionObserver(
    (entries) => {
      for (const e of entries) e.isIntersecting ? inBand.add(e.target.id) : inBand.delete(e.target.id);
      update();
    },
    { rootMargin: "-35% 0px -60% 0px" }
  );
  [...links.keys(), "recommendations"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) band.observe(el);
  });
  new IntersectionObserver(([e]) => {
    atEnd = e.isIntersecting;
    update();
  }).observe(document.querySelector(".footer"));
}

function initReveal() {
  if (reduceMotion.matches || !("IntersectionObserver" in window)) return;
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        e.target.classList.replace("is-pending", "is-in");
        io.unobserve(e.target);
      }
    },
    { rootMargin: "0px 0px -6% 0px" }
  );
  for (const el of document.querySelectorAll(".reveal")) {
    if (el.getBoundingClientRect().top < innerHeight) continue;
    el.classList.add("is-pending");
    io.observe(el);
  }
}

function initTabs() {
  const box = document.querySelector("[data-tabs]");
  if (!box) return;
  const lab = document.getElementById("lab");
  const tabs = [...box.querySelectorAll('[role="tab"]')];
  const panels = tabs.map((t) => document.getElementById(t.getAttribute("aria-controls")));
  const announced = new Set();
  let near = false;
  let current = 0;
  let tabbing = false;

  const announce = () => {
    const fig = panels[current].querySelector("[data-demo]");
    if (!near || !fig || announced.has(fig.dataset.demo)) return;
    announced.add(fig.dataset.demo);
    fig.dataset.shown = "";
    document.dispatchEvent(new CustomEvent("lab:show", { detail: { demo: fig.dataset.demo } }));
  };
  const select = (i, { focus = false, hash = false } = {}) => {
    current = i;
    tabs.forEach((t, j) => {
      t.setAttribute("aria-selected", String(j === i));
      t.tabIndex = j === i ? 0 : -1;
      panels[j].hidden = j !== i;
    });
    if (focus) tabs[i].focus();
    if (hash) history.replaceState(null, "", `#${panels[i].id}`);
    announce();
  };
  const indexOf = (hash) => panels.findIndex((p) => `#${p.id}` === hash);
  const show = (i) => {
    select(i);
    lab.scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "start" });
  };

  tabs.forEach((t, i) => t.addEventListener("click", () => select(i, { hash: true })));
  addEventListener("keydown", (e) => e.key === "Tab" && (tabbing = true), true);
  addEventListener("keyup", () => (tabbing = false), true);
  panels.forEach((p) => {
    p.addEventListener("focus", () => p.toggleAttribute("data-quiet", !tabbing));
    p.addEventListener("blur", () => p.removeAttribute("data-quiet"));
  });
  box.querySelector('[role="tablist"]').addEventListener("keydown", (e) => {
    const i = tabs.indexOf(document.activeElement);
    const n = tabs.length;
    const next = { ArrowRight: i + 1, ArrowLeft: i - 1 + n, Home: 0, End: n - 1 }[e.key];
    if (i < 0 || next === undefined) return;
    e.preventDefault();
    select(next % n, { focus: true, hash: true });
  });
  document.addEventListener("click", (e) => {
    const a = e.target.closest('a[href^="#lab-"]');
    const i = a ? indexOf(a.hash) : -1;
    if (i < 0) return;
    e.preventDefault();
    history.pushState(null, "", a.hash);
    show(i);
  });
  addEventListener("hashchange", () => {
    const i = indexOf(location.hash);
    if (i >= 0) show(i);
  });
  new IntersectionObserver(
    ([e]) => {
      near = e.isIntersecting;
      announce();
    },
    { rootMargin: "400px 0px" }
  ).observe(lab);

  const start = indexOf(location.hash);
  select(Math.max(start, 0));
  if (start >= 0) requestAnimationFrame(() => lab.scrollIntoView({ block: "start" }));
}

function initRanges() {
  document.querySelectorAll('.range input[type="range"]').forEach(syncRange);
  document.addEventListener("input", (e) => {
    if (e.target.matches?.('.range input[type="range"]')) syncRange(e.target);
  });
}

initTheme();
initNav();
initRanges();
initTabs();
initReveal();
