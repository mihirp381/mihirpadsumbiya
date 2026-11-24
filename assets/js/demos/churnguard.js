import { onTheme } from "../lib/lab-ui.js";

const APP = new URL("../../../lab/churnguard-app/app/", import.meta.url);
const IMG = new URL("../../img/churnguard/", import.meta.url);
const ALT = "The ChurnGuard app scoring Customer C: a customer information form on the left, a 63.1% churn probability with the decision to flag for retention, a radar chart of the customer's percentiles against the median churned and retained customers, and two donut charts of the held-out churn rate and the share flagged at a 0.30 threshold.";
const NOTE = "Start downloads about 31 MB (Python compiled to WebAssembly, Streamlit, pandas and the packages they need) from cdn.jsdelivr.net, pypi.org and files.pythonhosted.org. In my tests on a fast connection the app appeared after 7 to 11 seconds. These are plain file downloads; nothing you enter leaves this tab.";

const theme = () => (document.documentElement.dataset.theme === "dark" ? "dark" : "light");

export async function mount(fig) {
  const body = fig.querySelector(".fig-body");
  body.classList.add("cg-body");
  body.innerHTML = `<div class="cg-poster">
  <div class="cg-shot">
    <img class="cg-shot-light" src="${new URL("app-paper.webp", IMG)}" width="1280" height="800" alt="${ALT}">
    <img class="cg-shot-dark" src="${new URL("app-night.webp", IMG)}" width="1280" height="800" alt="${ALT}">
  </div>
  <div class="cg-start">
    <p class="note">${NOTE}</p>
    <button type="button" class="btn">Start the app</button>
  </div>
</div>`;

  let frame = null;
  const src = () => `${APP}?theme=${theme()}`;
  body.querySelector("button").addEventListener("click", () => {
    frame = document.createElement("iframe");
    frame.className = "cg-frame";
    frame.title = "ChurnGuard, a Streamlit app running in this tab";
    frame.src = src();
    body.replaceChildren(frame);
    frame.focus();
  });
  onTheme(() => {
    if (frame) frame.src = src();
  });
}
