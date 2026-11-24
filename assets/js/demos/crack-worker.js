import { createModel, forward } from "../lib/crack-cnn.js";

let model = null;

async function fetchOk(name, as) {
  const res = await fetch(new URL(`../../models/${name}`, import.meta.url));
  if (!res.ok) throw new Error(`${res.status} ${name}`);
  return res[as]();
}

export function load() {
  if (!model) {
    model = Promise.all([fetchOk("crack-cnn.json", "json"), fetchOk("crack-cnn.bin", "arrayBuffer")]).then(([meta, bin]) => createModel(meta, bin));
    model.catch(() => (model = null));
  }
  return model;
}

export async function run(rgb, onStage) {
  const out = forward(await load(), rgb, onStage);
  return { p: out.p, ms: out.ms, conv2: out.conv2, channels: out.conv1Top.channels, maps: out.conv1Top.maps };
}

if (typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope) {
  self.onmessage = async ({ data: { id, rgb } }) => {
    try {
      const out = await run(rgb, (stage) => self.postMessage({ type: "stage", id, stage }));
      self.postMessage({ type: "result", id, ...out }, [out.conv2.buffer, ...out.maps.map((m) => m.buffer)]);
    } catch (err) {
      self.postMessage({ type: "error", id, message: String(err?.message || err) });
    }
  };
}
