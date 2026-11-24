# Model tools

Scripts that train the in-browser demo models and export them to `assets/models/`. Raw data stays outside the repo.

## Setup

```
uv venv ../.venv --python 3.12
uv pip install --python ../.venv/bin/python -r tools/requirements.txt
```

Run every command from the repo root with `python -I` (isolated mode). The data lives outside the repo, so every script takes its data paths as flags.

## Scripts

| Script | Output |
| --- | --- |
| `export_gbm.py` | Library: sklearn gradient boosting -> compact JSON trees, plus Python checks and parity fixtures |
| `train_concrete.py` | `assets/models/concrete.json`, `fixtures/gbm_parity.json` (concrete) |
| `train_churn.py` | `assets/models/churn.json`, `fixtures/gbm_parity.json` (churn) |
| `crack_prep.py` | Decodes the crack images once to `crack-cache/X128.npy` (uint8, 128x128 RGB) |
| `crack_model.py` | Library: the CNN, split, batched prediction |
| `train_crack.py` | Trains the CNN on `mps`, writes checkpoints, split and `train_log.json` to `crack-run/` |
| `export_crack.py` | Quantises, evaluates, writes `assets/models/crack-cnn.{bin,json}`, `assets/img/crack/s01..s12.jpg`, `fixtures/crack_parity.json` |
| `build_bcrp.py` | `assets/models/bcrp.json` |
| `churnguard_stats.py` | `assets/apps/churnguard/population.json`: percentiles and box-plot statistics of the churn training split, for the Figure 6 app |
| `parity.mjs` | Node check that the JS libraries match Python; exits non-zero on failure |
| `hero_check.mjs` | Gradient check and convergence check for the hero network in `assets/js/lib/mlp.js` |
| `demo_check.mjs` | Checks the pure logic behind the Lab figures against the shipped model files |
| `churnguard_check.py` | Standard-library check that the Figure 6 app's Python tree walker matches `churn.json` and `gbm.js` to 1e-9 and writes five notes per customer |

## Reproduce

`DATA` is the folder holding the downloaded datasets (UCI concrete, the Özgenel crack images); `REFS` is the folder holding clones of the source repositories (ChurnGuard, Drug-Extraction-using-Natural-Language-Processing).

```
python -I tools/train_concrete.py --xls DATA/concrete/extracted/Concrete_Data.xls
python -I tools/train_churn.py --csv REFS/ChurnGuard/Cleaned_Modelling.csv
python -I tools/crack_prep.py --src DATA/crack --out DATA/crack-cache
python -I tools/train_crack.py --cache DATA/crack-cache --out DATA/crack-run
python -I tools/export_crack.py --cache DATA/crack-cache --run DATA/crack-run
python -I tools/build_bcrp.py --tsv REFS/Drug-Extraction-using-Natural-Language-Processing/training.tsv
python -I tools/churnguard_stats.py --csv REFS/ChurnGuard/Cleaned_Modelling.csv
node tools/parity.mjs
node tools/hero_check.mjs
node tools/demo_check.mjs
python3 -I tools/churnguard_check.py
```

GBM results are deterministic. CNN training on `mps` is not bit-reproducible run to run.

## Data

- Concrete: I-Cheng Yeh (1998), Cement and Concrete Research 28(12); UCI ML Repository "Concrete Compressive Strength", CC BY 4.0. Unzip `concrete.zip` into its own directory.
- Bank churn: `Cleaned_Modelling.csv` from the ChurnGuard repo (UTD FinHack 2024), a cleaned copy of a public bank customer churn dataset.
- Cracks: Özgenel, Ç.F. (2019), "Concrete Crack Images for Classification", Mendeley Data V2, doi:10.17632/5y9wdsg2zt.2, CC BY 4.0. Extract `crack.rar` with `bsdtar -xf` into its own directory.
- BCRP: `training.tsv`, the output of the owner's literature-mining pipeline (Drug-Extraction-using-Natural-Language-Processing repo).

## Formats

- GBM JSON `model`: `{init, lr, trees:[{f,t,l,r,v}]}`, one entry per node; `f=-1` marks a leaf. Cast inputs to float32, go left when `x[f] <= t`; raw = init + lr * sum of leaf values. Regression output = raw; classifier p = sigmoid(raw). Thresholds are the shortest decimals that give the same decision as sklearn's float64 threshold for every float32 input. JS: `assets/js/lib/gbm.js` (`predictRaw`, `sigmoid`).
- CNN: `crack-cnn.json` `tensors` gives name, dtype, shape, byte offset/length and int8 scale(s) inside `crack-cnn.bin`; `meta.layout` gives the memory layout (HWC activations, conv `[out][ky][kx][in]`, fc1 `[out][h][w][c]`). JS: `assets/js/lib/crack-cnn.js` (`createModel(json, arrayBuffer)`, `forward(model, rgb, onStage)`).
