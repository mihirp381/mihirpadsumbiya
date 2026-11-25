import argparse
import ast
import gzip
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "assets", "models", "bcrp.json"))
    a = ap.parse_args()
    d = pd.read_csv(a.tsv, sep="\t", keep_default_na=False, dtype=str)
    rows, allrefs = [], set()
    for _, r in d.iterrows():
        refs = ast.literal_eval(r["References"])
        allrefs.update(refs)
        cid = int(float(r["PubChem_CID"])) if r["PubChem_CID"].strip() else None
        act = {"BCRP_inhibitor": 1, "BCRP_substrate": 0}[r["Activity"]]
        rows.append([r["Drug"], act, cid, len(refs), refs[:3], r["SMILES"]])
    rows.sort(key=lambda x: (-x[3], x[0]))
    meta = {
        "source": "Final output of the BCRP/ABCG2 literature-mining pipeline (Lantern Pharma datathon), training.tsv",
        "activity": "1 = BCRP_inhibitor, 0 = BCRP_substrate",
        "refs": "first 3 PMC ids in the order the pipeline listed them; nrefs = length of the full list",
        "total": len(rows),
        "inhibitors": sum(r[1] == 1 for r in rows),
        "substrates": sum(r[1] == 0 for r in rows),
        "unique_references": len(allrefs),
        "rows_with_smiles": sum(bool(r[5].strip()) for r in rows),
        "rows_with_cid": sum(r[2] is not None for r in rows),
        "rows_with_repeated_ref_in_list": int(sum(len(ast.literal_eval(x)) != len(set(ast.literal_eval(x))) for x in d["References"])),
    }
    out = {"meta": meta, "cols": ["name", "activity", "cid", "nrefs", "refs", "smiles"], "rows": rows}
    s = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(s)
    print(json.dumps(meta, indent=1))
    print("size", len(s.encode()), "gzip", len(gzip.compress(s.encode(), 9)))
    for r in rows[:25]:
        print(r[3], r[0], "inhibitor" if r[1] else "substrate", r[2])


if __name__ == "__main__":
    main()
