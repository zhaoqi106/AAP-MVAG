#!/usr/bin/env python3
import argparse
import glob
import json
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for path in glob.glob(os.path.join(args.root, "**", "metrics_test.json"), recursive=True):
        with open(path) as f:
            m = json.load(f)
        m["metrics_path"] = path
        m["run_dir"] = os.path.dirname(path)
        rows.append(m)
    if not rows:
        raise SystemExit(f"No metrics_test.json found under {args.root}")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out.replace(".csv", "_per_run.csv"), index=False)
    group_cols = [c for c in ["model", "task_name", "split"] if c in df.columns]
    metric_cols = [c for c in ["AUC", "AUPRC", "BEDROC20", "EF@0.1%", "EF@0.5%", "EF@1%", "EF@5%", "EF@10%", "Precision@30", "Precision@50", "Precision@100", "MCC", "F1"] if c in df.columns]
    out_rows = []
    for key, g in df.groupby(group_cols):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        for mc in metric_cols:
            vals = pd.to_numeric(g[mc], errors="coerce")
            base[f"{mc}_mean"] = vals.mean()
            base[f"{mc}_std"] = vals.std(ddof=1)
        base["n_runs"] = len(g)
        out_rows.append(base)
    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    print(args.out)


if __name__ == "__main__":
    main()
