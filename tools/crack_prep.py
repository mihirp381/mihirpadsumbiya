import argparse
import json
import os

import numpy as np
from PIL import Image

SIZE = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files, labels = [], []
    for cls, lab in (("Negative", 0), ("Positive", 1)):
        names = sorted(n for n in os.listdir(os.path.join(a.src, cls)) if n.lower().endswith(".jpg"))
        files += [f"{cls}/{n}" for n in names]
        labels += [lab] * len(names)
    X = np.empty((len(files), SIZE, SIZE, 3), np.uint8)
    sizes, modes = {}, {}
    for i, f in enumerate(files):
        with Image.open(os.path.join(a.src, f)) as im:
            sizes[str(im.size)] = sizes.get(str(im.size), 0) + 1
            modes[im.mode] = modes.get(im.mode, 0) + 1
            X[i] = np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))
    np.save(os.path.join(a.out, "X128.npy"), X)
    np.save(os.path.join(a.out, "y.npy"), np.array(labels, np.uint8))
    with open(os.path.join(a.out, "files.json"), "w") as fh:
        json.dump({"files": files, "sizes": sizes, "modes": modes}, fh)
    print(len(files), sizes, modes)


if __name__ == "__main__":
    main()
