import argparse
import base64
import copy
import io
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crack_model import CrackCNN, predict  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
QUANT = ("conv2.weight", "fc1.weight")


def js_layout(name, t):
    if name in ("conv1.weight", "conv2.weight"):
        return t.permute(0, 2, 3, 1).contiguous()
    if name == "fc1.weight":
        return t.reshape(128, 128, 7, 7).permute(0, 2, 3, 1).reshape(128, 6272).contiguous()
    return t.contiguous()


def quantize(t, mode):
    if mode == "float16":
        return t.half().float(), None
    if mode == "per_tensor":
        s = (t.abs().max() / 127).float().reshape(1)
    else:
        s = (t.reshape(t.shape[0], -1).abs().amax(1) / 127).float()
    view = s.reshape(-1, *([1] * (t.dim() - 1)))
    q = torch.clamp(torch.round(t / view), -127, 127).to(torch.int8)
    return q.float() * view, (q, s)


def metrics(p, y):
    pred = (p > 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return {"accuracy": round((tp + tn) / len(y), 5), "precision": round(prec, 5), "recall": round(rec, 5),
            "f1": round(2 * prec * rec / (prec + rec), 5), "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "n": int(len(y)), "positive": "crack (label 1)", "decision": "p > 0.5"}


def p_cpu(model, u8):
    x = torch.from_numpy(np.ascontiguousarray(u8)).permute(0, 3, 1, 2).float().div(255.0)
    with torch.no_grad():
        return torch.sigmoid(model(x)).double().numpy()


def jpeg_roundtrip(u8):
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, format="JPEG", quality=92)
    data = buf.getvalue()
    return data, np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--models", default=os.path.join(HERE, "..", "assets", "models"))
    ap.add_argument("--img", default=os.path.join(HERE, "..", "assets", "img", "crack"))
    ap.add_argument("--fixture", default=os.path.join(HERE, "fixtures", "crack_parity.json"))
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    X = np.load(os.path.join(a.cache, "X128.npy"), mmap_mode="r")
    y = np.load(os.path.join(a.cache, "y.npy")).astype(int)
    files = json.load(open(os.path.join(a.cache, "files.json")))["files"]
    sp = np.load(os.path.join(a.run, "split.npz"))
    te = sp["test"]
    log = json.load(open(os.path.join(a.run, "train_log.json")))

    sd = torch.load(os.path.join(a.run, "crack_final.pt"), map_location="cpu")
    f32 = CrackCNN(); f32.load_state_dict(sd); f32.eval()
    p32 = predict(f32.to(dev), X, te, dev, bs=50)
    m32 = metrics(p32, y[te])

    tried = []
    for mode in ("per_tensor", "per_channel", "float16"):
        sdq = copy.deepcopy(sd)
        packed = {}
        for k in QUANT:
            sdq[k], packed[k] = quantize(sd[k], mode)
        mq = CrackCNN(); mq.load_state_dict(sdq); mq.eval()
        pq = predict(mq.to(dev), X, te, dev, bs=50)
        mm = metrics(pq, y[te])
        drop = round(100 * (m32["accuracy"] - mm["accuracy"]), 3)
        tried.append({"mode": mode, "accuracy": mm["accuracy"], "drop_pp": drop})
        print(tried[-1], flush=True)
        if drop <= 0.3:
            break
    mq = mq.cpu()

    blob, tensors = bytearray(), []
    for k, v in sd.items():
        lay = js_layout(k, v)
        entry = {"name": k, "shape": list(lay.shape)}
        if k in QUANT and mode != "float16":
            q, s = packed[k]
            data = js_layout(k, q.float()).to(torch.int8).numpy().tobytes()
            entry.update(dtype="int8", scale=[float(x) for x in s.numpy()] if mode == "per_channel" else float(s.numpy()[0]))
        elif k in QUANT:
            data = lay.half().numpy().astype("<f2").tobytes()
            entry["dtype"] = "float16"
        else:
            data = lay.float().numpy().astype("<f4").tobytes()
            entry["dtype"] = "float32"
        while len(blob) % 4:
            blob.append(0)
        entry.update(offset=len(blob), length=len(data))
        blob += data
        tensors.append(entry)
    with open(os.path.join(a.models, "crack-cnn.bin"), "wb") as fh:
        fh.write(blob)

    yt = y[te]
    wrong = np.flatnonzero((pq > 0.5).astype(int) != yt)
    pick_wrong = int(wrong[np.argmax(np.abs(pq[wrong] - 0.5))]) if len(wrong) else None
    samples, chosen = [], []
    os.makedirs(a.img, exist_ok=True)
    for cls in (1, 0):
        ok = np.flatnonzero((yt == cls) & ((pq > 0.5).astype(int) == cls))
        ok = ok[np.argsort(pq[ok], kind="stable")]
        need_wrong = pick_wrong is not None and yt[pick_wrong] == cls
        n_ok = 5 if need_wrong else 6
        ranks = list(np.linspace(0, len(ok) - 1, n_ok).round().astype(int))
        picks = [("ok", r) for r in ranks] + ([("wrong", pick_wrong)] if need_wrong else [])
        for kind, r in picks:
            cand = [pick_wrong] if kind == "wrong" else [ok[r + d] for d in (0, 1, -1, 2, -2, 3, -3) if 0 <= r + d < len(ok)]
            for i in cand:
                if i in chosen:
                    continue
                data, dec = jpeg_roundtrip(np.asarray(X[te[i]]))
                p = float(p_cpu(mq, dec[None])[0])
                if ((p > 0.5) != (cls == 1)) == (kind == "wrong"):
                    chosen.append(i)
                    samples.append({"data": data, "label": cls, "p": p, "misclassified": kind == "wrong",
                                    "source": files[te[i]]})
                    break
            else:
                raise RuntimeError("no stable candidate")
    samples.sort(key=lambda s: (-s["label"], -s["p"]))
    out_samples = []
    for n, s in enumerate(samples, 1):
        name = f"s{n:02d}.jpg"
        with open(os.path.join(a.img, name), "wb") as fh:
            fh.write(s["data"])
        out_samples.append({"file": f"assets/img/crack/{name}", "label": s["label"], "p": s["p"],
                            "misclassified": s["misclassified"], "source": s["source"]})

    rng = np.random.default_rng(42)
    fx = np.concatenate([rng.choice(np.flatnonzero(yt == c), 3, replace=False) for c in (1, 0)])
    imgs = np.asarray(X[te[fx]])
    fp = p_cpu(mq, imgs)
    fixture = {"note": "raw 128x128x3 HWC RGB bytes from the cached test set; p from PyTorch (CPU, float32) with shipped dequantised weights",
               "images": [{"source": files[te[i]], "label": int(yt[i]), "rgb": base64.b64encode(im.tobytes()).decode(), "p": float(p)}
                          for i, im, p in zip(fx, imgs, fp)]}
    with open(a.fixture, "w") as fh:
        json.dump(fixture, fh, separators=(",", ":"))

    meta = {
        "paper": "Padsumbiya, M., Brahmbhatt, V., Thakkar, S. (2022). Automatic Crack Detection Using Convolutional Neural Network. "
                 "Journal of Soft Computing in Civil Engineering 6(3). doi:10.22115/SCCE.2022.325596.1397",
        "dataset": "Özgenel, Ç.F. (2019), \"Concrete Crack Images for Classification\", Mendeley Data, V2, doi:10.17632/5y9wdsg2zt.2, CC BY 4.0",
        "architecture": "Conv2d(3->128,3x3,valid)+ReLU, MaxPool4, Conv2d(128->128,3x3,valid)+ReLU, MaxPool4, Flatten(6272), "
                        "Dense(128)+ReLU [L2 0.01 on kernel and bias], Dense(1)+sigmoid",
        "params": sum(v.numel() for v in sd.values()),
        "input": "128x128x3 RGB, 227x227 source resized with PIL bilinear, pixel/255",
        "labels": "1 = crack (Positive folder), 0 = no crack (Negative). The paper used the opposite polarity.",
        "split": {"train": int(len(sp["train"])), "val": int(len(sp["val"])), "test": int(len(te)), "per_class": [16200, 1800, 2000],
                  "method": "stratified, sklearn train_test_split twice (test 4000, then val 3600), seed 42", "train_subset": log["train_subset"]},
        "hyperparameters": {"loss": "binary cross-entropy (BCEWithLogits) + 0.01*(sum fc1.W^2 + sum fc1.b^2)",
                            "optimizer": "Adam", "lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8, "batch_size": 16, "epochs": len(log["epochs"]),
                            "init": "Glorot uniform weights, zero biases (Keras defaults)", "shuffle": "every epoch, numpy seed 42", "torch_seed": 42},
        "epochs": log["epochs"],
        "epoch_log_note": "train_acc_running = mean over the epoch while training (as Keras reports); train_acc_end_of_epoch = full training split in eval mode after the epoch; train_seconds excludes evaluation",
        "device": log["device"], "torch": log["torch"],
        "train_seconds_total": log["train_seconds_sum"], "wall_seconds_incl_eval": log["total_seconds"],
        "metrics_float32": m32,
        "metrics_shipped": mm,
        "quantization": {"tensors": list(QUANT), "chosen": mode, "tried": tried,
                         "rule": "int8 symmetric, scale = max|w|/127, q = round(w/scale) clipped to [-127,127]; w = q*scale; "
                                 "per-channel scale indexes the first (output) dimension"},
        "layout": {"activations": "HWC (row, column, channel)",
                   "conv": "[out][ky][kx][in]", "fc1": "[out][h][w][c] over the 7x7x128 HWC pooled map (flatten index (h*7+w)*128+c)",
                   "fc2": "[out=1][in=128]", "bin": "little-endian, each tensor 4-byte aligned; offset/length in bytes",
                   "pytorch_flatten": "PyTorch flattens NCHW as (c*7+h)*7+w; fc1 columns were permuted to HWC for export"},
        "sample_rule": "test images; per class spread evenly over the shipped model's sorted p among correct predictions; the misclassified "
                       "image with the most confident error is included once; p recomputed from the saved JPEG (q92) re-decoded with PIL",
        "bin_bytes": len(blob),
    }
    out = {"meta": meta, "tensors": tensors, "samples": out_samples}
    with open(os.path.join(a.models, "crack-cnn.json"), "w") as fh:
        json.dump(out, fh, separators=(",", ":"), ensure_ascii=False)
    print(json.dumps({k: meta[k] for k in ("metrics_float32", "metrics_shipped", "quantization")}, indent=1))
    print(json.dumps(out_samples, indent=1))
    print("wrong count", len(wrong))


if __name__ == "__main__":
    main()
