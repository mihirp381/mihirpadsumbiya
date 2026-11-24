import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crack_model import CrackCNN, predict, split_indices, to_input  # noqa: E402

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--train-subset", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    X = np.load(os.path.join(a.cache, "X128.npy"))
    y = np.load(os.path.join(a.cache, "y.npy")).astype(np.float32)
    tr, va, te = split_indices(y, SEED)
    if a.train_subset:
        from sklearn.model_selection import train_test_split
        tr, _ = train_test_split(tr, train_size=a.train_subset, stratify=y[tr], random_state=SEED)
        tr = np.sort(tr)
    np.savez(os.path.join(a.out, "split.npz"), train=tr, val=va, test=te)

    model = CrackCNN().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 954241, n_params
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    bce = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)
    log = []
    t_all = time.time()
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        perm = tr[rng.permutation(len(tr))]
        correct = torch.zeros((), device=dev)
        loss_sum = torch.zeros((), device=dev)
        steps = len(perm) // 16 + (len(perm) % 16 > 0)
        for s in range(steps):
            b = perm[s * 16:(s + 1) * 16]
            xb = to_input(X[b], dev)
            yb = torch.from_numpy(y[b]).to(dev)
            logit = model(xb)
            loss = bce(logit, yb) + model.penalty()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            correct += ((logit.detach() > 0).float() == yb).sum()
            loss_sum += loss.detach() * len(b)
            if s % 200 == 0:
                print(f"ep{ep + 1} step {s}/{steps} {time.time() - t0:.1f}s", flush=True)
            if a.max_steps and s + 1 >= a.max_steps:
                print(f"bench {a.max_steps} steps {time.time() - t0:.2f}s", flush=True)
                return
        train_sec = time.time() - t0
        pv = predict(model, X, va, dev)
        val_acc = float(((pv > 0.5) == (y[va] > 0.5)).mean())
        pt = predict(model, X, tr, dev)
        train_acc = float(((pt > 0.5) == (y[tr] > 0.5)).mean())
        rec = {"epoch": ep + 1, "train_acc_end_of_epoch": round(train_acc, 5),
               "train_acc_running": round(float(correct) / len(perm), 5), "train_loss_running": round(float(loss_sum) / len(perm), 5),
               "val_acc": round(val_acc, 5), "train_seconds": round(train_sec, 1), "epoch_seconds_incl_eval": round(time.time() - t0, 1)}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save(model.state_dict(), os.path.join(a.out, f"crack_ep{ep + 1}.pt"))
    summary = {"device": str(dev), "torch": torch.__version__, "params": n_params, "epochs": log,
               "train_size": int(len(tr)), "val_size": int(len(va)), "test_size": int(len(te)),
               "train_subset": a.train_subset or None, "total_seconds": round(time.time() - t_all, 1),
               "train_seconds_sum": round(sum(r["train_seconds"] for r in log), 1)}
    with open(os.path.join(a.out, "train_log.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    torch.save(model.state_dict(), os.path.join(a.out, "crack_final.pt"))
    print("done", summary["total_seconds"], flush=True)


if __name__ == "__main__":
    main()
