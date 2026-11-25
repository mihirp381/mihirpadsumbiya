import { refs, cssVar, onTheme } from "../lib/lab-ui.js";
import { esc } from "../lib/plot.js";
import { fmt, compoundIndex, searchCompounds } from "../lib/demo-math.js";

const EXAMPLES = ["mitoxantrone", "topotecan", "imatinib", "methotrexate", "Hoechst 33342"];
const FILTERS = [["all", "All"], ["inhibitor", "Inhibitors"], ["substrate", "Substrates"]];
const CODE = "https://github.com/mihirp381/Drug-Extraction-using-Natural-Language-Processing";
const MK = {
  inhibitor: '<svg class="mk mk-sq" viewBox="0 0 10 10" aria-hidden="true"><rect x="0.5" y="0.5" width="9" height="9"/></svg>',
  substrate: '<svg class="mk mk-ci" viewBox="0 0 10 10" aria-hidden="true"><circle cx="5" cy="5" r="5"/></svg>',
};

let drawer = null;
const loadDrawer = () =>
  (drawer ||= new Promise((resolve, reject) => {
    const el = document.createElement("script");
    el.src = new URL("../vendor/smiles-drawer.min.js", import.meta.url).href;
    el.onload = () => (window.SmilesDrawer ? resolve(window.SmilesDrawer) : reject(new Error("SmilesDrawer missing")));
    el.onerror = () => {
      drawer = null;
      reject(new Error("SmilesDrawer failed to load"));
    };
    document.head.append(el);
  }));

const pipeHTML = (total, inh, sub) => [
  ["Collect", "24,648", "PubMed Central papers pulled by API, plus 4,561 keyword matches from 293,996 bioRxiv papers"],
  ["Extract", "211,889", "drug mentions found near BCRP/ABCG2 by two NER models"],
  ["Classify", "81.03%", "test accuracy of a CNN + BiLSTM + attention relation classifier"],
  ["Deliver", total, `compounds: ${inh} inhibitors, ${sub} substrates, each with SMILES and sources`],
].map(([k, n, t], i) => `<li class="pipe-s"><span class="pipe-k"><span class="pipe-i">${i + 1}</span>${k}</span><span class="pipe-n">${n}</span><span class="pipe-t">${t}</span></li>`).join("");

export async function mount(fig, hooks = {}) {
  const body = fig.querySelector(".fig-body");
  body.innerHTML = `<div class="lab">
  <ol class="pipe" data-ref="pipe">${pipeHTML("&nbsp;", "", "")}</ol>
  <div class="lab-grid lab-grid-pick">
    <section class="lab-col lab-controls" aria-labelledby="bc-h">
      <div class="lab-h"><h3 id="bc-h">Compounds</h3><span class="lab-h-sub" data-ref="count">Loading compounds</span></div>
      <label class="lab-search"><span class="range-label">Search the mined compounds</span>
        <input type="search" data-ref="q" placeholder="e.g. mitoxantrone" autocomplete="off" spellcheck="false" disabled></label>
      <div class="lab-chips" data-ref="ex">${EXAMPLES.map((e) => `<button type="button" class="chip lab-chip">${esc(e)}</button>`).join("")}</div>
      <div class="seg" role="group" aria-label="Filter by predicted class" data-ref="filter">${FILTERS.map(([v, l], i) => `<button type="button" data-f="${v}" aria-pressed="${i === 0}">${MK[v] ?? ""}${l}</button>`).join("")}</div>
      <ul class="bc-list" data-ref="list" aria-label="Matching compounds"></ul>
    </section>
    <section class="lab-col lab-out" aria-label="Selected compound">
      <div class="bc-detail" data-ref="detail" aria-live="polite"><p class="note">Loading compounds</p></div>
      <div class="bc-mol"><svg data-ref="mol" role="img" aria-label="2D structure"></svg><p class="note bc-mol-note" data-ref="molnote" hidden>Structure could not be drawn from this SMILES.</p></div>
      <p class="note">This is the pipeline's raw output. Entity recognition over-extracts, so a few entries are not drugs at all (ATP, K, glucose), spellings that differ only by case appear twice, and roughly one label in five is wrong. I left it as the pipeline produced it.</p>
    </section>
  </div>
</div>`;
  const r = refs(body);

  const res = await fetch(new URL("../../models/bcrp.json", import.meta.url));
  if (!res.ok) throw new Error(`${res.status} bcrp.json`);
  const buf = await res.arrayBuffer();
  const data = JSON.parse(new TextDecoder().decode(buf));
  const { meta, rows } = data;
  const C = Object.fromEntries(data.cols.map((c, i) => [c, i]));
  const index = compoundIndex(rows);
  const s = { filter: "all", items: [], sel: -1, token: 0 };

  r.pipe.innerHTML = pipeHTML(fmt(meta.total), fmt(meta.inhibitors), fmt(meta.substrates));
  r.q.disabled = false;

  const drawMol = async (smiles) => {
    const token = ++s.token;
    r.molnote.hidden = true;
    let SD;
    try {
      SD = await loadDrawer();
    } catch (err) {
      console.error(err);
      if (token === s.token) r.molnote.hidden = false;
      return;
    }
    if (token !== s.token) return;
    const other = cssVar("--ink-2");
    const theme = { ...Object.fromEntries(["H", "S", "P", "F", "CL", "BR", "I", "B", "SI"].map((k) => [k, other])), C: cssVar("--ink"), O: cssVar("--ox"), N: cssVar("--blue"), BACKGROUND: cssVar("--paper") };
    const sd = new SD.SvgDrawer({ themes: { site: theme }, bondThickness: 1.1, fontFamily: '"Courier Prime", "Courier New", monospace', fontSizeLarge: 11, fontSizeSmall: 7, padding: 12 });
    SD.parse(
      smiles,
      (tree) => {
        if (token !== s.token) return;
        try {
          sd.draw(tree, r.mol, "site");
        } catch (err) {
          r.mol.replaceChildren();
          r.molnote.hidden = false;
        }
      },
      () => {
        if (token !== s.token) return;
        r.mol.replaceChildren();
        r.molnote.hidden = false;
      }
    );
  };

  const select = (i) => {
    s.sel = i;
    const row = rows[i];
    const inh = row[C.activity] === 1, n = row[C.nrefs], refsList = row[C.refs], cid = row[C.cid], smiles = row[C.smiles];
    const link = (href, label) => `<a href="${href}" target="_blank" rel="noopener">${esc(label)}</a>`;
    r.detail.innerHTML = `<h3 class="bc-name">${esc(row[C.name])}</h3>
      <dl class="kv">
        <div><dt>predicted</dt><dd class="bc-class">${MK[inh ? "inhibitor" : "substrate"]}BCRP ${inh ? "inhibitor" : "substrate"}</dd></div>
        <div><dt>mentioned in</dt><dd>${fmt(n)} ${n === 1 ? "paper" : "papers"}</dd></div>
        ${cid ? `<div><dt>PubChem</dt><dd>${link(`https://pubchem.ncbi.nlm.nih.gov/compound/${encodeURIComponent(cid)}`, `CID ${cid}`)}</dd></div>` : ""}
        <div class="kv-stack"><dt>sources</dt><dd class="bc-refs">${refsList.map((p) => link(`https://pmc.ncbi.nlm.nih.gov/articles/${encodeURIComponent(p)}/`, p)).join(" ")}${n > refsList.length ? ` <span class="bc-more">+${fmt(n - refsList.length)} more</span>` : ""}</dd></div>
        <div class="kv-stack"><dt>SMILES</dt><dd class="bc-smiles"><code>${esc(smiles)}</code><button type="button" class="btn btn-ghost btn-sm bc-copy">Copy</button></dd></div>
      </dl>`;
    r.detail.querySelector(".bc-copy").addEventListener("click", (e) => {
      navigator.clipboard?.writeText(smiles).then(() => (e.target.textContent = "Copied"), () => {});
    });
    r.list.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.i === i)));
    if (smiles) drawMol(smiles);
    else {
      r.mol.replaceChildren();
      r.molnote.hidden = false;
    }
  };

  const update = () => {
    const q = r.q.value;
    const { n, items } = searchCompounds(index, q, s.filter);
    s.items = items;
    r.count.textContent = q.trim() ? `${fmt(n)} ${n === 1 ? "match" : "matches"}${n > items.length ? `, first ${items.length} shown` : ""}` : `${items.length} most-referenced of ${fmt(n)}`;
    r.list.innerHTML = items.length
      ? items.map((c) => `<li><button type="button" class="bc-item" data-i="${c.i}" aria-pressed="${c.i === s.sel}">${MK[c.act ? "inhibitor" : "substrate"]}<span class="bc-item-name">${esc(c.name)}</span>` +
          `<span class="dots" aria-hidden="true"></span><span class="sr">${c.act ? "inhibitor" : "substrate"}, </span><span class="bc-n">${fmt(c.nrefs)} ${c.nrefs === 1 ? "paper" : "papers"}</span></button></li>`).join("")
      : `<li class="note bc-empty">No compound names contain “${esc(q.trim())}”.</li>`;
  };

  r.q.addEventListener("input", update);
  r.q.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      r.list.querySelector("button")?.focus();
    } else if (e.key === "Enter" && s.items.length) {
      e.preventDefault();
      select(s.items[0].i);
    }
  });
  r.list.addEventListener("keydown", (e) => {
    const k = { ArrowDown: 1, ArrowUp: -1 }[e.key];
    if (!k) return;
    e.preventDefault();
    const btns = [...r.list.querySelectorAll("button")];
    const j = btns.indexOf(document.activeElement) + k;
    (j < 0 ? r.q : btns[Math.min(j, btns.length - 1)]).focus();
  });
  r.list.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b) select(+b.dataset.i);
  });
  r.ex.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    r.q.value = b.textContent;
    update();
    if (s.items.length) select(s.items[0].i);
  });
  r.filter.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    s.filter = b.dataset.f;
    r.filter.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    update();
  });
  onTheme(() => s.sel >= 0 && rows[s.sel][C.smiles] && drawMol(rows[s.sel][C.smiles]));

  fig.querySelector(".fig-cap").innerHTML =
    `<span class="fig-num">Figure 4.</span>The dataset my Lantern Pharma Datathon pipeline produced (1st place): ${fmt(meta.total)} compounds mined from ${fmt(meta.unique_references)} papers and labelled as inhibitors or substrates of the BCRP/ABCG2 efflux transporter. ` +
    `Search runs locally over a ${fmt(buf.byteLength / 1e6, 1)} MB file. Structures are drawn from SMILES with SmilesDrawer. Code: <a href="${CODE}" target="_blank" rel="noopener">github.com/mihirp381/Drug-Extraction-using-Natural-Language-Processing</a>.`;

  if (hooks.q) r.q.value = hooks.q;
  update();
  if (s.items.length) select(s.items[0].i);
}
