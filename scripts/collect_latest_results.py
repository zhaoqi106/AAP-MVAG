#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import yaml


def read_yaml(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def collect(root):
    rows = []
    for metrics_path in sorted(root.glob("*/metrics_test.yaml")):
        run_dir = metrics_path.parent
        metrics = read_yaml(metrics_path)
        config = read_yaml(run_dir / "config.yaml")
        summary = read_yaml(run_dir / "run_summary.yaml")
        gates = read_yaml(run_dir / "test_gate_weights.yaml")
        dataset_cfg = config.get("dataset", {})
        training_cfg = config.get("training", {})
        global_gate = gates.get("global", {})
        fingerprint_gate = gates.get("fingerprint", {})
        splitting = dataset_cfg.get("splitting")
        split = {
            "stratified": "balanced_random",
            "stratified_scaffold": "balanced_scaffold",
        }.get(splitting, splitting)
        valid_fraction = dataset_cfg.get("valid_size")
        test_fraction = dataset_cfg.get("test_size")
        train_fraction = (
            1.0 - float(valid_fraction) - float(test_fraction)
            if valid_fraction is not None and test_fraction is not None
            else None
        )
        rows.append(
            {
                "task": config.get("task_name"),
                "split": split,
                "train_fraction": train_fraction,
                "valid_fraction": valid_fraction,
                "test_fraction": test_fraction,
                "run": config.get("run_idx"),
                "split_seed": dataset_cfg.get("seed"),
                "model_seed": training_cfg.get("model_seed"),
                "best_epoch": summary.get("best_epoch"),
                "best_val_AUC": summary.get("best_val_auc"),
                "best_val_AUPRC": summary.get("best_val_auprc"),
                "AUC": metrics.get("AUC"),
                "AUPRC": metrics.get("AUPRC"),
                "BEDROC20": metrics.get("BEDROC20"),
                "EF@0.5%": metrics.get("EF@0.5%"),
                "EF@1%": metrics.get("EF@1%"),
                "Precision@50": metrics.get("Precision@50"),
                "Precision@100": metrics.get("Precision@100"),
                "N": metrics.get("N"),
                "N_POS": metrics.get("N_POS"),
                "graph_AUPRC": metrics.get("graph_auprc"),
                "fingerprint_AUPRC": metrics.get("fp_auprc"),
                "descriptor_AUPRC": metrics.get("desc_auprc"),
                "joint_AUPRC": metrics.get("joint_auprc"),
                "gate_graph": global_gate.get("graph"),
                "gate_fingerprint": global_gate.get("fp"),
                "gate_descriptor": global_gate.get("desc"),
                "gate_joint": global_gate.get("joint"),
                "fp_cmf_r2": fingerprint_gate.get("cmf_r2"),
                "fp_cmf_r3": fingerprint_gate.get("cmf_r3"),
                "fp_rdk": fingerprint_gate.get("rdk"),
                "fp_atom_pair": fingerprint_gate.get("atom_pair"),
                "fp_maccs": fingerprint_gate.get("maccs"),
                "training_time_seconds": summary.get(
                    "training_time_seconds"
                ),
                "trainable_parameter_count": summary.get(
                    "trainable_parameter_count"
                ),
                "run_dir": str(run_dir),
                "_completed_at_ns": metrics_path.stat().st_mtime_ns,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    latest_keys = [
        "task",
        "split",
        "split_seed",
        "model_seed",
    ]
    return (
        frame.sort_values("_completed_at_ns")
        .drop_duplicates(subset=latest_keys, keep="last")
        .drop(columns="_completed_at_ns")
        .sort_values(["task", "split", "run"])
        .reset_index(drop=True)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiment_records")
    parser.add_argument("--out", default="results_summary.csv")
    args = parser.parse_args()
    frame = collect(Path(args.root))
    frame.to_csv(args.out, index=False)
    grouped_path = Path(args.out).with_name(
        f"{Path(args.out).stem}_grouped{Path(args.out).suffix}"
    )
    if frame.empty:
        pd.DataFrame().to_csv(grouped_path, index=False)
    else:
        metric_columns = [
            "AUC",
            "AUPRC",
            "BEDROC20",
            "EF@0.5%",
            "EF@1%",
            "Precision@50",
            "Precision@100",
            "best_epoch",
            "training_time_seconds",
            "graph_AUPRC",
            "fingerprint_AUPRC",
            "descriptor_AUPRC",
            "joint_AUPRC",
        ]
        grouped = (
            frame.groupby(
                [
                    "task",
                    "split",
                    "train_fraction",
                    "valid_fraction",
                    "test_fraction",
                ],
                dropna=False,
            )[metric_columns]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        grouped.to_csv(grouped_path, index=False)
    print(f"saved {args.out} rows={len(frame)}")
    print(f"saved {grouped_path}")


if __name__ == "__main__":
    main()
