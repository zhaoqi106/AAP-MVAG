import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception as e:
    print("XGBoost not available:", e)
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception as e:
    print("LightGBM not available:", e)
    HAS_LGBM = False


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def smiles_to_ecfp(smiles_list, radius=2, n_bits=2048):
    fps = []
    valid_idx = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius,
            nBits=n_bits,
            useChirality=True,
        )
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        valid_idx.append(i)

    if not fps:
        raise ValueError("No valid molecules found.")

    return np.asarray(fps, dtype=np.float32), np.asarray(valid_idx, dtype=int)


def bedroc_from_ranked_labels(ranked_labels, alpha=20.0):
    ranked_labels = np.asarray(ranked_labels).astype(int)
    n = int(len(ranked_labels))
    n_pos = int(np.sum(ranked_labels))

    if n <= 1 or n_pos <= 0 or n_pos >= n:
        return float("nan")

    active_ranks = np.flatnonzero(ranked_labels > 0) + 1
    active_fraction = n_pos / n

    observed = np.exp(-alpha * active_ranks / n).sum()
    random_sum = (
        active_fraction
        * (-np.expm1(-alpha))
        / np.expm1(alpha / n)
    )

    if random_sum <= 0:
        return float("nan")

    rie = observed / random_sum
    common = (n / n_pos) * (-np.expm1(-alpha * active_fraction)) / (-np.expm1(-alpha))
    rie_max = common
    rie_min = common * np.exp(-alpha * (1.0 - active_fraction))
    denom = rie_max - rie_min

    if denom <= 0:
        return float("nan")

    return float(np.clip((rie - rie_min) / denom, 0.0, 1.0))


def compute_metrics(y_true, y_score):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)

    finite = np.isfinite(y_score)
    y_true = y_true[finite]
    y_score = y_score[finite]

    y_pred = (y_score >= 0.5).astype(int)

    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = int(n - n_pos)
    pos_rate = float(n_pos / n) if n > 0 else float("nan")

    order = np.argsort(-y_score, kind="stable")
    ranked_labels = y_true[order]

    def enrichment_factor(top_fraction):
        if n == 0 or n_pos == 0:
            return float("nan")
        top_n = int(np.ceil(n * top_fraction))
        top_n = max(top_n, 1)
        hits = float(ranked_labels[:top_n].sum())
        expected = n_pos * (top_n / n)
        return hits / expected if expected > 0 else float("nan")

    def precision_at_k(k):
        if n == 0:
            return float("nan")
        top_n = min(int(k), n)
        return float(ranked_labels[:top_n].sum()) / top_n

    metrics = {
        "AUC": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "AUPRC": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "ACC": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "BEDROC20": float(bedroc_from_ranked_labels(ranked_labels, alpha=20.0)),
        "EF@0.5%": float(enrichment_factor(0.005)),
        "EF@1%": float(enrichment_factor(0.01)),
        "Precision@50": float(precision_at_k(50)),
        "Precision@100": float(precision_at_k(100)),
        "Precision@500": float(precision_at_k(500)),
        "Precision@1000": float(precision_at_k(1000)),
        "N": int(n),
        "N_POS": int(n_pos),
        "N_NEG": int(n_neg),
        "POS_RATE": float(pos_rate),
    }

    if np.isfinite(metrics["AUPRC"]) and pos_rate > 0:
        metrics["AUPRC_LIFT"] = float(metrics["AUPRC"] / pos_rate)
    else:
        metrics["AUPRC_LIFT"] = float("nan")

    if np.isfinite(metrics["AUPRC"]) and 0.0 < pos_rate < 1.0:
        metrics["NORM_AUPRC"] = float((metrics["AUPRC"] - pos_rate) / (1.0 - pos_rate))
    else:
        metrics["NORM_AUPRC"] = float("nan")

    return metrics


def load_task_csv(path, smiles_col, target_col):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    if smiles_col not in df.columns:
        if "SMILES" in df.columns:
            smiles_col = "SMILES"
        elif "smiles" in df.columns:
            smiles_col = "smiles"
        else:
            raise ValueError(f"No SMILES/smiles column found in {path}")

    if target_col not in df.columns:
        raise ValueError(f"Cannot find target column {target_col} in {path}. Existing columns: {list(df.columns)}")

    labels = pd.to_numeric(df[target_col], errors="coerce")
    smiles = df[smiles_col]

    valid = smiles.notna() & labels.notna()
    smiles = smiles[valid].astype(str).tolist()
    labels = labels[valid].astype(int).to_numpy()

    # Convert possible {-1, 1} labels to {0, 1}
    labels = np.where(labels == -1, 0, labels)
    labels = np.where(labels > 0, 1, 0)

    return smiles, labels


def stratified_scaffold_split(smiles, y, valid_size=0.1, test_size=0.1, seed=42):
    """GNEprop-style non-chiral scaffold-stratified split.

    Scaffold groups are kept intact. Groups with many positives are assigned
    greedily to the split with the lowest target-normalized positive count,
    while respecting the requested split sizes where possible.
    """
    rng = random.Random(seed)
    scaffolds = {}

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Cannot compute scaffold for molecule index {i}")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False,
        )
        scaffolds.setdefault(scaffold, set()).add(i)

    train_fraction = 1.0 - float(valid_size) - float(test_size)
    train_target_size = int(train_fraction * len(y))
    valid_target_size = int(float(valid_size) * len(y))
    test_target_size = int(float(test_size) * len(y))

    big_index_sets = []
    small_index_sets = []
    for index_set in scaffolds.values():
        if (
            len(index_set) > valid_target_size / 2
            and len(index_set) > test_target_size / 2
        ):
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)

    rng.shuffle(big_index_sets)
    rng.shuffle(small_index_sets)

    train_idx = list(
        index for index_set in big_index_sets for index in index_set
    )
    valid_idx = []
    test_idx = []
    groups = [
        {
            "indices": train_idx,
            "target_size": train_target_size,
            "positive_count": int(np.sum(y[train_idx] == 1)),
        },
        {
            "indices": test_idx,
            "target_size": test_target_size,
            "positive_count": 0,
        },
        {
            "indices": valid_idx,
            "target_size": valid_target_size,
            "positive_count": 0,
        },
    ]

    small_index_lists = [list(index_set) for index_set in small_index_sets]
    # The shuffle above supplies the seed-dependent tie break.
    small_index_lists.sort(
        key=lambda index_set: int(np.sum(y[index_set] == 1)),
        reverse=True,
    )

    for index_set in small_index_lists:
        ratios = [
            group["positive_count"] / (group["target_size"] + 1e-3)
            for group in groups
        ]
        positive_count = int(np.sum(y[index_set] == 1))
        group_order = list(range(len(groups)))
        if (
            math.isclose(ratios[0], ratios[1], rel_tol=1e-3)
            and math.isclose(ratios[1], ratios[2], rel_tol=1e-3)
        ):
            rng.shuffle(group_order)
        else:
            group_order.sort(key=lambda group_index: ratios[group_index])

        priority_groups = [groups[group_index] for group_index in group_order]
        if (
            len(priority_groups[0]["indices"]) + len(index_set)
            <= priority_groups[0]["target_size"]
        ):
            destination = priority_groups[0]
        elif (
            len(priority_groups[1]["indices"]) + len(index_set)
            <= priority_groups[1]["target_size"]
        ):
            destination = priority_groups[1]
        else:
            destination = priority_groups[2]
        destination["indices"].extend(index_set)
        destination["positive_count"] += positive_count

    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    rng.shuffle(test_idx)

    return (
        np.asarray(train_idx, dtype=int),
        np.asarray(valid_idx, dtype=int),
        np.asarray(test_idx, dtype=int),
    )


def make_models(seed, n_pos, n_neg, n_jobs, xgb_device, lgbm_device, lgbm_gpu_device_id):
    scale_pos_weight = float(n_neg / max(n_pos, 1))

    models = {
        "RF_ECFP": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=seed,
        )
    }

    if HAS_XGB:
        xgb_kwargs = {
            "n_estimators": 800,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "scale_pos_weight": scale_pos_weight,
            "n_jobs": n_jobs,
            "random_state": seed,
        }
        if xgb_device:
            xgb_kwargs["device"] = xgb_device
        models["XGBoost_ECFP"] = XGBClassifier(**xgb_kwargs)

    if HAS_LGBM:
        lgbm_kwargs = {
            "n_estimators": 800,
            "learning_rate": 0.03,
            "num_leaves": 64,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary",
            "scale_pos_weight": scale_pos_weight,
            "n_jobs": n_jobs,
            "random_state": seed,
            "verbose": -1,
        }
        if lgbm_device:
            lgbm_kwargs["device_type"] = lgbm_device
        if lgbm_gpu_device_id:
            lgbm_kwargs["gpu_device_id"] = int(lgbm_gpu_device_id)
        models["LightGBM_ECFP"] = LGBMClassifier(**lgbm_kwargs)

    return models


def load_fixed_split(path):
    import torch

    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or not {"train_idx", "valid_idx", "test_idx"} <= set(obj.keys()):
        raise ValueError(f"split_indices.pt must contain train_idx, valid_idx, test_idx: {path}")
    return (
        np.asarray(obj["train_idx"], dtype=np.int64),
        np.asarray(obj["valid_idx"], dtype=np.int64),
        np.asarray(obj["test_idx"], dtype=np.int64),
    )


def run_one_task(
    task,
    seeds,
    out_dir,
    radius,
    n_bits,
    n_jobs,
    xgb_device,
    lgbm_device,
    lgbm_gpu_device_id,
    valid_size,
    test_size,
    split_method,
    fixed_split_json="",
):
    task_name = task["name"]
    path = task["path"]
    smiles_col = task.get("smiles_col", "SMILES")
    target_col = task["target"]

    print("=" * 100)
    print(f"Task: {task_name}")
    print(f"CSV: {path}")
    print(f"Target: {target_col}")

    smiles, y = load_task_csv(path, smiles_col, target_col)
    fixed_map = {}
    if fixed_split_json:
        with open(fixed_split_json, "r", encoding="utf-8") as handle:
            fixed_map = json.load(handle)

    print(f"Raw valid rows: {len(y)} pos={int(y.sum())} pos_rate={float(y.mean()):.6f}")

    X, valid_idx = smiles_to_ecfp(smiles, radius=radius, n_bits=n_bits)
    if fixed_map and not np.array_equal(np.asarray(valid_idx), np.arange(len(y))):
        raise RuntimeError(
            f"{task_name}: RDKit dropped {len(y) - len(valid_idx)} rows, so "
            "split_indices.pt would not refer to the same molecules as test11."
        )
    y = y[valid_idx]
    smiles = [smiles[i] for i in valid_idx]

    print(f"Valid molecules after RDKit: {len(y)} pos={int(y.sum())} pos_rate={float(y.mean()):.6f}")

    if len(np.unique(y)) < 2:
        raise ValueError(f"Task {task_name} has only one class after filtering.")

    all_rows = []

    for seed in seeds:
        print("-" * 100)
        print(f"Task={task_name} seed={seed}")

        idx_all = np.arange(len(y))

        temp_size = float(valid_size) + float(test_size)
        if not 0.0 < temp_size < 1.0:
            raise ValueError(f"valid_size + test_size must be in (0, 1), got {temp_size}")

        split_method = str(split_method).lower()
        split_source = ""
        if fixed_map:
            if str(seed) not in fixed_map:
                raise KeyError(f"No fixed split for seed {seed}")
            split_source = fixed_map[str(seed)]
            train_idx, valid_idx2, test_idx = load_fixed_split(split_source)
            for name, idx in (("train", train_idx), ("valid", valid_idx2), ("test", test_idx)):
                if len(idx) == 0 or idx.min() < 0 or idx.max() >= len(y):
                    raise RuntimeError(f"{task_name} seed {seed} {name} index out of range")
            if set(train_idx) & set(valid_idx2) or set(train_idx) & set(test_idx) or set(valid_idx2) & set(test_idx):
                raise RuntimeError(f"{task_name} seed {seed} fixed split has overlapping indices")
            y_train = y[train_idx]
            y_test = y[test_idx]
            split_prefix = "fixed_test11"
        elif split_method in {"stratified_random", "stratified", "random"}:
            split_prefix = "stratified_random"
            train_idx, temp_idx, y_train, y_temp = train_test_split(
                idx_all,
                y,
                test_size=temp_size,
                random_state=seed,
                stratify=y,
            )

            valid_idx2, test_idx, y_valid, y_test = train_test_split(
                temp_idx,
                y_temp,
                test_size=float(test_size) / temp_size,
                random_state=seed,
                stratify=y_temp,
            )
        elif split_method in {"stratified_scaffold", "scaffold_stratified", "balanced_scaffold"}:
            split_prefix = "stratified_scaffold"
            train_idx, valid_idx2, test_idx = stratified_scaffold_split(
                smiles,
                y,
                valid_size=valid_size,
                test_size=test_size,
                seed=seed,
            )
            y_train = y[train_idx]
            y_test = y[test_idx]
        else:
            raise ValueError(f"Unsupported split_method: {split_method}")

        X_train = X[train_idx]
        X_valid = X[valid_idx2]
        X_test = X[test_idx]

        print(
            f"Split seed={seed}: "
            f"train={len(train_idx)} pos={int(y[train_idx].sum())} rate={float(y[train_idx].mean()):.5f} | "
            f"valid={len(valid_idx2)} pos={int(y[valid_idx2].sum())} rate={float(y[valid_idx2].mean()):.5f} | "
            f"test={len(test_idx)} pos={int(y[test_idx].sum())} rate={float(y[test_idx].mean()):.5f}"
        )

        models = make_models(
            seed=seed,
            n_pos=int(y_train.sum()),
            n_neg=int(len(y_train) - y_train.sum()),
            n_jobs=n_jobs,
            xgb_device=xgb_device,
            lgbm_device=lgbm_device,
            lgbm_gpu_device_id=lgbm_gpu_device_id,
        )

        for model_name, model in models.items():
            print(f"Training {model_name} ...")

            model.fit(X_train, y_train)

            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                y_score = model.decision_function(X_test)

            metrics = compute_metrics(y_test, y_score)

            run_dir = out_dir / task_name / model_name / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            pred_df = pd.DataFrame({
                "row_id": np.asarray(test_idx, dtype=int),
                "SMILES": [smiles[i] for i in test_idx],
                "label": y_test,
                "prob": y_score,
                "pred": (y_score >= 0.5).astype(int),
            })
            pred_df.to_csv(run_dir / "test_predictions.csv", index=False)
            if split_source:
                (run_dir / "split_indices_source.txt").write_text(split_source + "\n", encoding="utf-8")

            with open(run_dir / "metrics_test.yaml", "w") as f:
                yaml.safe_dump(metrics, f, sort_keys=False)

            row = {
                "task": task_name,
                "model": model_name,
                "split": (
                    f"fixed_test11_{split_method}"
                    if fixed_map
                    else f"{split_prefix}_{int(round((1.0 - temp_size) * 100))}_{int(round(float(valid_size) * 100))}_{int(round(float(test_size) * 100))}"
                ),
                "seed": seed,
                "run_dir": str(run_dir),
                **metrics,
            }
            all_rows.append(row)

            print(
                f"{task_name} {model_name} seed={seed}: "
                f"AUC={metrics['AUC']:.4f} "
                f"AUPRC={metrics['AUPRC']:.4f} "
                f"BEDROC20={metrics['BEDROC20']:.4f} "
                f"EF@1%={metrics['EF@1%']:.2f} "
                f"P@50={metrics['Precision@50']:.3f} "
                f"P@100={metrics['Precision@100']:.3f}"
            )

    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="ml_ecfp_stratified_random_results")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n_bits", type=int, default=2048)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--tasks_json", default="")
    parser.add_argument("--xgb_device", default=os.environ.get("XGB_DEVICE", ""))
    parser.add_argument("--lgbm_device", default=os.environ.get("LGBM_DEVICE", ""))
    parser.add_argument("--lgbm_gpu_device_id", default=os.environ.get("LGBM_GPU_DEVICE_ID", ""))
    parser.add_argument("--valid_size", type=float, default=0.05)
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--split_method", default="stratified_random")
    parser.add_argument("--fixed_split_json", default="", help="JSON mapping seed to test11 split_indices.pt")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    if args.tasks_json:
        tasks = json.load(open(args.tasks_json, "r"))
    else:
        tasks = [
            {
                "name": "GNEtolC",
                "path": "data/GNEtolC.csv",
                "smiles_col": "SMILES",
                "target": "Activity_tolC",
            },
            {
                "name": "HepG2",
                "path": "data/datasets - HepG2 cytotoxicity.csv",
                "smiles_col": "smiles",
                "target": "label",
            },
            {
                "name": "HSkMC",
                "path": "data/datasets - HSkMC cytotoxicity.csv",
                "smiles_col": "smiles",
                "target": "label",
            },
            {
                "name": "IMR90",
                "path": "data/datasets - IMR-90 viability.csv",
                "smiles_col": "smiles",
                "target": "label",
            },
        ]

    all_rows = []
    for task in tasks:
        rows = run_one_task(
            task=task,
            seeds=seeds,
            out_dir=out_dir,
            radius=args.radius,
            n_bits=args.n_bits,
            n_jobs=args.n_jobs,
            xgb_device=args.xgb_device,
            lgbm_device=args.lgbm_device,
            lgbm_gpu_device_id=args.lgbm_gpu_device_id,
            valid_size=args.valid_size,
            test_size=args.test_size,
            split_method=args.split_method,
            fixed_split_json=args.fixed_split_json,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "ml_ecfp_run_level_metrics.csv", index=False)

    metric_cols = [
        "AUC",
        "AUPRC",
        "AUPRC_LIFT",
        "NORM_AUPRC",
        "BEDROC20",
        "EF@0.5%",
        "EF@1%",
        "Precision@50",
        "Precision@100",
        "Precision@500",
        "Precision@1000",
        "N",
        "N_POS",
        "POS_RATE",
    ]

    grouped = (
        df.groupby(["task", "model", "split"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.to_csv(out_dir / "ml_ecfp_grouped_metrics_mean_std.csv", index=False)

    print("\nSaved:")
    print(out_dir / "ml_ecfp_run_level_metrics.csv")
    print(out_dir / "ml_ecfp_grouped_metrics_mean_std.csv")

    print("\nGrouped summary:")
    print(grouped)


if __name__ == "__main__":
    main()