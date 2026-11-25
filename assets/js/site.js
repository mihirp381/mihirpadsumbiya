const root = document.documentElement;

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
    if (btn) btn.setAttribute("aria-pressed", String(theme === "dark"));
    const bg = getComputedStyle(root).getPropertyValue("--paper").trim();
    metas.forEach((m) => m.setAttribute("content", bg));
    if (save) {
      try { localStorage.setItem("theme", theme); } catch {}
    }
  };
  apply(root.dataset.theme === "dark" ? "dark" : "light", false);
  if (btn) btn.addEventListener("click", () => apply(root.dataset.theme === "dark" ? "light" : "dark", true));
}

function initRanges() {
  document.querySelectorAll('.range input[type="range"]').forEach(syncRange);
  document.addEventListener("input", (e) => {
    if (e.target.matches?.('.range input[type="range"]')) syncRange(e.target);
  });
}

function initTables() {
  document.querySelectorAll(".prose table").forEach((t) => {
    const heads = [...t.querySelectorAll("thead th")].map((th) => th.textContent.trim());
    t.querySelectorAll("tbody tr").forEach((tr) => [...tr.children].forEach((td, i) => heads[i] && td.setAttribute("data-label", heads[i])));
  });
}

if (!("siteReady" in root.dataset)) {
  root.dataset.siteReady = "";
  initTheme();
  initRanges();
  initTables();
}
