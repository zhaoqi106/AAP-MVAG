#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight GINE baseline for molecular binary classification.

Features:
- RF/ML-style 8:1:1 balanced random split or scaffold split when no split file is provided.
- Optional external split file: CSV with a split column, or pickle with train/valid/test indices.
- Metrics: AUROC, AUPRC, BEDROC20 when RDKit scorer is available, EF@0.1/0.5/1/5/10, Precision@K, Recall@K.
- Saves best checkpoint by validation AUPRC and writes test_predictions.csv + metrics_test.json.

Example:
python run_gine_baseline.py \
  --data-csv /path/to/GNEtolC.csv \
  --smiles-col smiles --label-col label \
  --task-name GNEtolC --split balanced_random --seed 42 \
  --out-dir /home/zhao/pzj/gine-duibi/results/GNEtolC/balanced_random/seed42
"""

import argparse
import json
import math
import os
import pickle
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_recall_fscore_support, matthews_corrcoef, cohen_kappa_score
from sklearn.model_selection import train_test_split

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception as exc:
    raise RuntimeError("RDKit is required. Install with: conda install -c conda-forge rdkit") from exc

try:
    from torch_geometric.data import Data, Dataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, global_max_pool
except Exception as exc:
    raise RuntimeError("PyTorch Geometric is required. Install torch-geometric matching your torch/cuda version.") from exc


BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def one_hot(value, choices: Sequence) -> List[float]:
    return [float(value == c) for c in choices]


def atom_features(atom: Chem.Atom) -> List[float]:
    atomic_num = atom.GetAtomicNum()
    degree = atom.GetTotalDegree()
    formal_charge = atom.GetFormalCharge()
    total_h = atom.GetTotalNumHs()
    implicit_valence = atom.GetImplicitValence()
    hybrid = atom.GetHybridization()
    feats = []
    feats += [atomic_num / 100.0]
    feats += [degree / 6.0]
    feats += [formal_charge / 5.0]
    feats += [total_h / 8.0]
    feats += [implicit_valence / 8.0]
    feats += one_hot(hybrid, HYBRIDIZATIONS)
    feats += [float(atom.GetIsAromatic())]
    feats += [float(atom.IsInRing())]
    feats += [float(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED)]
    return feats


def bond_features(bond: Chem.Bond) -> List[float]:
    feats = []
    feats += one_hot(bond.GetBondType(), BOND_TYPES)
    feats += [float(bond.GetIsConjugated())]
    feats += [float(bond.IsInRing())]
    stereo = int(bond.GetStereo())
    feats += [float(stereo) / 6.0]
    return feats


def smiles_to_data(smiles: str, y: float, row_id: int) -> Optional[Data]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index.append([i, j])
        edge_attr.append(bf)
        edge_index.append([j, i])
        edge_attr.append(bf)
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([float(y)], dtype=torch.float))
    data.smiles = str(smiles)
    data.row_id = int(row_id)
    return data


class MolDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, smiles_col: str, label_col: str):
        super().__init__()
        self.items = []
        bad = 0
        for idx, row in frame.iterrows():
            data = smiles_to_data(row[smiles_col], row[label_col], int(idx))
            if data is None:
                bad += 1
            else:
                self.items.append(data)
        if bad:
            print(f"[WARN] skipped invalid molecules: {bad}")

    def len(self):
        return len(self.items)

    def get(self, idx):
        return self.items[idx]


def scaffold_of(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "INVALID"
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return "INVALID"


def balanced_random_split(labels: np.ndarray, seed: int, train_frac: float = 0.8, valid_frac: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_idx, tmp_idx = train_test_split(indices, test_size=1.0 - train_frac, random_state=seed, stratify=labels)
    tmp_labels = labels[tmp_idx]
    valid_rel, test_rel = train_test_split(np.arange(len(tmp_idx)), test_size=0.5, random_state=seed, stratify=tmp_labels)
    valid_idx = tmp_idx[valid_rel]
    test_idx = tmp_idx[test_rel]
    return np.array(train_idx), np.array(valid_idx), np.array(test_idx)


def balanced_scaffold_split(smiles: Sequence[str], labels: np.ndarray, seed: int, train_frac: float = 0.8, valid_frac: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GNEprop-style non-chiral scaffold-stratified split."""
    rng = random.Random(seed)
    scaffold_to_indices: Dict[str, set] = {}
    for i, smi in enumerate(smiles):
        scaffold_to_indices.setdefault(scaffold_of(smi), set()).add(i)

    n = len(labels)
    test_frac = 1.0 - float(train_frac) - float(valid_frac)
    train_target_size = int(float(train_frac) * n)
    valid_target_size = int(float(valid_frac) * n)
    test_target_size = int(test_frac * n)

    big_index_sets = []
    small_index_sets = []
    for index_set in scaffold_to_indices.values():
        if (
            len(index_set) > valid_target_size / 2
            and len(index_set) > test_target_size / 2
        ):
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)

    rng.shuffle(big_index_sets)
    rng.shuffle(small_index_sets)

    train_idx = [
        index for index_set in big_index_sets for index in index_set
    ]
    valid_idx = []
    test_idx = []
    groups = [
        {
            "indices": train_idx,
            "target_size": train_target_size,
            "positive_count": int(np.sum(labels[train_idx] == 1)),
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
    small_index_lists.sort(
        key=lambda index_set: int(np.sum(labels[index_set] == 1)),
        reverse=True,
    )

    for index_set in small_index_lists:
        ratios = [
            group["positive_count"] / (group["target_size"] + 1e-3)
            for group in groups
        ]
        positive_count = int(np.sum(labels[index_set] == 1))
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

    return np.array(train_idx), np.array(valid_idx), np.array(test_idx)


def load_external_split(path: str, df: pd.DataFrame, smiles_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.endswith(".pt"):
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict) or not {"train_idx", "valid_idx", "test_idx"} <= set(obj.keys()):
            raise ValueError(f"split_indices.pt must contain train_idx, valid_idx, test_idx: {path}")
        return (
            np.asarray(obj["train_idx"], dtype=np.int64),
            np.asarray(obj["valid_idx"], dtype=np.int64),
            np.asarray(obj["test_idx"], dtype=np.int64),
        )
    if path.endswith(".pkl") or path.endswith(".pickle"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        # Common formats: dict with train/valid/test, train_idx/val_idx/test_idx, or tuple/list.
        if isinstance(obj, dict):
            keys = {k.lower(): k for k in obj.keys()}
            train = obj.get(keys.get("train", "train"), obj.get(keys.get("train_idx", "train_idx")))
            valid = obj.get(keys.get("valid", "valid"), obj.get(keys.get("val", "val"), obj.get(keys.get("valid_idx", "valid_idx"))))
            test = obj.get(keys.get("test", "test"), obj.get(keys.get("test_idx", "test_idx")))
            if train is None or valid is None or test is None:
                raise ValueError(f"Cannot parse split pickle keys: {list(obj.keys())}")
            return np.array(train), np.array(valid), np.array(test)
        if isinstance(obj, (list, tuple)) and len(obj) >= 3:
            return np.array(obj[0]), np.array(obj[1]), np.array(obj[2])
        raise ValueError("Unsupported pickle split format.")

    split_df = pd.read_csv(path)
    split_col = None
    for c in ["split", "subset", "set"]:
        if c in split_df.columns:
            split_col = c
            break
    if split_col is None:
        raise ValueError("Split CSV must contain one of: split, subset, set")

    if "index" in split_df.columns:
        idx = split_df["index"].astype(int).values
        split_values = split_df[split_col].astype(str).str.lower().values
        train = idx[np.isin(split_values, ["train", "training"])]
        valid = idx[np.isin(split_values, ["valid", "val", "validation"])]
        test = idx[np.isin(split_values, ["test"])]
        return train, valid, test

    if smiles_col in split_df.columns:
        smi_to_idx = {str(s): i for i, s in enumerate(df[smiles_col].astype(str).values)}
        train, valid, test = [], [], []
        for _, row in split_df.iterrows():
            idx = smi_to_idx.get(str(row[smiles_col]))
            if idx is None:
                continue
            sp = str(row[split_col]).lower()
            if sp in ["train", "training"]:
                train.append(idx)
            elif sp in ["valid", "val", "validation"]:
                valid.append(idx)
            elif sp == "test":
                test.append(idx)
        return np.array(train), np.array(valid), np.array(test)

    raise ValueError("Split CSV must contain either index column or smiles column.")


class GINEModel(nn.Module):
    def __init__(self, atom_dim: int, edge_dim: int, hidden_dim: int = 256, num_layers: int = 5, dropout: float = 0.2, pool: str = "mean"):
        super().__init__()
        self.dropout = dropout
        self.pool = pool
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=edge_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        readout_dim = hidden_dim if pool in ["mean", "sum", "max"] else hidden_dim * 3
        self.head = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.atom_encoder(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_attr)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h
        if self.pool == "mean":
            g = global_mean_pool(x, batch)
        elif self.pool == "sum":
            g = global_add_pool(x, batch)
        elif self.pool == "max":
            g = global_max_pool(x, batch)
        else:
            g = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        return self.head(g).view(-1)


@dataclass
class EvalResult:
    y_true: np.ndarray
    y_score: np.ndarray
    y_pred: np.ndarray
    metrics: Dict[str, float]


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return float("nan")
    k = min(k, len(y_true))
    order = np.argsort(-y_score)[:k]
    return float(np.mean(y_true[order]))


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if y_true.sum() <= 0:
        return float("nan")
    k = min(k, len(y_true))
    order = np.argsort(-y_score)[:k]
    return float(y_true[order].sum() / y_true.sum())


def ef_at_percent(y_true: np.ndarray, y_score: np.ndarray, percent: float) -> float:
    if y_true.sum() <= 0:
        return float("nan")
    k = max(1, int(math.ceil(len(y_true) * percent / 100.0)))
    return precision_at_k(y_true, y_score, k) / max(float(y_true.mean()), 1e-12)


def bedroc20(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from rdkit.ML.Scoring import Scoring
        rows = [[float(s), int(y)] for s, y in zip(y_score, y_true)]
        rows.sort(key=lambda r: r[0], reverse=True)
        return float(Scoring.CalcBEDROC(rows, 1, 20.0))
    except Exception:
        return float("nan")


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_score = y_score.astype(float)
    y_pred = (y_score >= threshold).astype(int)
    m = {}
    try:
        m["AUC"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        m["AUC"] = float("nan")
    try:
        m["AUPRC"] = float(average_precision_score(y_true, y_score))
    except Exception:
        m["AUPRC"] = float("nan")
    m["BEDROC20"] = bedroc20(y_true, y_score)
    m["Accuracy"] = float(accuracy_score(y_true, y_pred))
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    m["Precision"] = float(pr)
    m["Recall"] = float(rc)
    m["F1"] = float(f1)
    try:
        m["MCC"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        m["MCC"] = float("nan")
    try:
        m["Kappa"] = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        m["Kappa"] = float("nan")
    for pct in [0.1, 0.5, 1.0, 5.0, 10.0]:
        name = f"EF@{pct:g}%"
        m[name] = ef_at_percent(y_true, y_score, pct)
    for k in [30, 50, 100, 300, 500, 1000]:
        m[f"Precision@{k}"] = precision_at_k(y_true, y_score, k)
        m[f"Recall@{k}"] = recall_at_k(y_true, y_score, k)
    m["N"] = int(len(y_true))
    m["N_POS"] = int(y_true.sum())
    m["N_NEG"] = int(len(y_true) - y_true.sum())
    m["POS_RATE"] = float(y_true.mean()) if len(y_true) else float("nan")
    return m


def evaluate(model, loader, device, threshold: float = 0.5) -> EvalResult:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            y = batch.y.view(-1).detach().cpu().numpy()
            ys.append(y)
            ps.append(prob)
    y_true = np.concatenate(ys) if ys else np.array([])
    y_score = np.concatenate(ps) if ps else np.array([])
    y_pred = (y_score >= threshold).astype(int)
    return EvalResult(y_true=y_true, y_score=y_score, y_pred=y_pred, metrics=compute_metrics(y_true, y_score, threshold))


def train_one_epoch(model, loader, optimizer, device, pos_weight: float, grad_clip: float) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float, device=device))
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        y = batch.y.view(-1)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def subset_dataset(dataset: MolDataset, indices: Sequence[int]) -> List[Data]:
    allowed = set(int(i) for i in indices)
    return [d for d in dataset.items if int(d.row_id) in allowed]


def infer_feature_dims(dataset: MolDataset) -> Tuple[int, int]:
    for d in dataset.items:
        atom_dim = int(d.x.shape[1])
        edge_dim = int(d.edge_attr.shape[1]) if d.edge_attr.numel() else 7
        return atom_dim, edge_dim
    raise ValueError("empty dataset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--task-name", default="task")
    parser.add_argument("--split", default="balanced_random", choices=["balanced_random", "balanced_scaffold"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--external-split", default=None, help="Optional .csv/.pkl split file. If given, --split only names output.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pool", default="mean", choices=["mean", "sum", "max", "mixed"])
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    df = pd.read_csv(args.data_csv)
    if args.smiles_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(f"CSV columns not found. Available columns: {list(df.columns)}")
    df = df[[args.smiles_col, args.label_col]].dropna().copy()
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
    df = df.dropna(subset=[args.label_col]).copy()
    df[args.label_col] = df[args.label_col].astype(int)
    df[args.label_col] = np.where(df[args.label_col] == -1, 0, df[args.label_col])
    df[args.label_col] = np.where(df[args.label_col] > 0, 1, 0)
    df.reset_index(drop=True, inplace=True)

    labels = df[args.label_col].values.astype(int)
    if args.external_split:
        train_idx, valid_idx, test_idx = load_external_split(args.external_split, df, args.smiles_col)
    elif args.split == "balanced_random":
        train_idx, valid_idx, test_idx = balanced_random_split(labels, args.seed)
    else:
        train_idx, valid_idx, test_idx = balanced_scaffold_split(df[args.smiles_col].astype(str).values, labels, args.seed)

    split_manifest = []
    for name, idxs in [("train", train_idx), ("valid", valid_idx), ("test", test_idx)]:
        y = labels[idxs]
        split_manifest.append({"split": name, "N": int(len(idxs)), "N_POS": int(y.sum()), "POS_RATE": float(y.mean()) if len(y) else None})
    with open(os.path.join(args.out_dir, "split_manifest.json"), "w") as f:
        json.dump(split_manifest, f, indent=2)
    print("[split]", split_manifest)

    dataset = MolDataset(df, args.smiles_col, args.label_col)
    train_data = subset_dataset(dataset, train_idx)
    valid_data = subset_dataset(dataset, valid_idx)
    test_data = subset_dataset(dataset, test_idx)

    atom_dim, edge_dim = infer_feature_dims(dataset)
    device = torch.device(args.device)
    model = GINEModel(atom_dim=atom_dim, edge_dim=edge_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout, pool=args.pool).to(device)

    y_train = labels[train_idx]
    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, int(len(y_train) - y_train.sum()))
    pos_weight = min(50.0, max(1.0, n_neg / n_pos))
    print(f"[train] pos_weight={pos_weight:.4f}, device={device}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_metric = -1.0
    best_epoch = -1
    bad_epochs = 0
    best_path = os.path.join(args.out_dir, "best_model.pt")
    history = []

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, pos_weight=pos_weight, grad_clip=args.grad_clip)
        scheduler.step()
        valid_res = evaluate(model, valid_loader, device, threshold=args.threshold)
        valid_auprc = valid_res.metrics.get("AUPRC", float("nan"))
        valid_bedroc = valid_res.metrics.get("BEDROC20", float("nan"))
        composite = valid_auprc if math.isnan(valid_bedroc) else 0.75 * valid_auprc + 0.25 * valid_bedroc
        history.append({"epoch": epoch, "loss": loss, "valid_AUPRC": valid_auprc, "valid_BEDROC20": valid_bedroc, "valid_composite": composite})
        print(f"epoch={epoch:03d} loss={loss:.4f} valid_auprc={valid_auprc:.4f} valid_bedroc20={valid_bedroc:.4f} composite={composite:.4f}")
        if composite > best_metric:
            best_metric = composite
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "atom_dim": atom_dim, "edge_dim": edge_dim}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[early stop] best_epoch={best_epoch}, best_valid_composite={best_metric:.4f}")
                break

    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "history.csv"), index=False)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_res = evaluate(model, test_loader, device, threshold=args.threshold)

    metrics = dict(test_res.metrics)
    metrics.update({"model": "GINE", "task_name": args.task_name, "split": args.split, "seed": args.seed, "best_epoch": best_epoch, "best_valid_composite": best_metric})
    with open(os.path.join(args.out_dir, "metrics_test.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pred_rows = []
    # Re-run with no shuffle and keep smiles order in test_data.
    for data, score, pred in zip(test_data, test_res.y_score, test_res.y_pred):
        pred_rows.append({"row_id": int(data.row_id), "smiles": data.smiles, "y_true": int(data.y.item()), "y_score": float(score), "y_pred": int(pred)})
    pd.DataFrame(pred_rows).to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)
    print("[test metrics]")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
