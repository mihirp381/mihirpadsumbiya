const DEMOS = new Set(["concrete", "crack", "churn", "bcrp"]);
const params = new URLSearchParams(location.search);
let hooks = { sample: params.get("sample"), q: params.get("q") };

function start(fig) {
  if (!fig || !DEMOS.has(fig.dataset.demo) || "mounted" in fig.dataset) return;
  fig.dataset.mounted = "";
  const opts = hooks || {};
  hooks = null;
  load(fig, opts);
}

async function load(fig, opts) {
  const body = fig.querySelector(".fig-body");
  try {
    const { mount } = await import(`./demos/${fig.dataset.demo}.js`);
    await mount(fig, opts);
    fig.dataset.ready = "";
  } catch (err) {
    console.error(err);
    body.innerHTML = '<div class="fig-loading lab-error"><p>This figure could not load its model.</p><button type="button" class="btn btn-ghost btn-sm">Retry</button></div>';
    const btn = body.querySelector("button");
    btn.addEventListener("click", () => {
      body.innerHTML = '<p class="fig-loading">Loading model</p>';
      load(fig, {});
    });
  }
}

document.querySelectorAll(".fig[data-demo]").forEach(start);
