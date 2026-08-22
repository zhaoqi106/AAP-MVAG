import multiprocessing as mp
import sys
import torch
import pandas as pd
import numpy as np
import hashlib
import itertools
import json
import math
import random
import re
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    FilterCatalog,
    GraphDescriptors,
    Lipinski,
    MACCSkeys,
    QED,
    rdMolDescriptors,
    rdchem,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os
from torch.utils.data import Subset, SubsetRandomSampler

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    rdchem.ChiralType.CHI_UNSPECIFIED,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]
BONDDIR_LIST = [
    rdchem.BondDir.NONE,
    rdchem.BondDir.ENDUPRIGHT,
    rdchem.BondDir.ENDDOWNRIGHT,
]

# ---- Enhanced atom / bond feature lookup lists (kept as categorical indices) ----
DEGREE_LIST = list(range(0, 7))
FORMAL_CHARGE_LIST = [-2, -1, 0, 1, 2]
HYBRIDIZATION_LIST = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
NUM_H_LIST = list(range(0, 5))
BONDSTEREO_LIST = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOCIS,
    rdchem.BondStereo.STEREOTRANS,
]

# Stable 2D descriptor groups:
# size/topology / charge/composition / polarity/surface /
# lipophilicity/flexibility/complexity / electronic / druglike-alerts
DESC_GROUP_DIMS = [10, 14, 7, 8, 2, 5]
DESC_TOTAL_DIM = int(sum(DESC_GROUP_DIMS))
DESC_NAMES = [
    "ExactMolWt",
    "HeavyAtomCount",
    "NumAtoms",
    "NumHeteroAtoms",
    "NumRings",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumRingSystems",
    "LargestRingSize",
    "FormalCharge",
    "PositiveAtomCount",
    "NegativeAtomCount",
    "NitrogenCount",
    "OxygenCount",
    "SulfurCount",
    "HalogenCount",
    "PrimaryAmineCount",
    "SecondaryAmineCount",
    "TertiaryAmineCount",
    "AmideCount",
    "AcidicGroupCount",
    "BasicGroupCount",
    "ZwitterionFlag",
    "TPSA",
    "HBD",
    "HBA",
    "NHOHCount",
    "NOCount",
    "LabuteASA",
    "FractionPolarAtoms",
    "MolLogP",
    "MolMR",
    "AromaticAtomFraction",
    "FractionCSP3",
    "RotatableBondCount",
    "BertzCT",
    "BridgeheadAtomCount",
    "SpiroAtomCount",
    "MaxPartialCharge",
    "MinPartialCharge",
    "ChiralCenterCount",
    "QED",
    "LipinskiViolationCount",
    "PAINSAlertCount",
    "StructuralAlertCount",
]
BOND_FEATURE_DIM = 6

FEATURE_VERSION = "mapvis_c_v1_graph_multifp_desc46_robust"
DEFAULT_FINGERPRINT_TYPE = "count_morgan"
DEFAULT_FINGERPRINT_RADIUS = 2
DEFAULT_FINGERPRINT_NUM_BITS = 2048
DEFAULT_FINGERPRINT_MAX_COUNT = 8

_PRIMARY_AMINE = Chem.MolFromSmarts("[NX3;H2;!$(NC=O)]")
_SECONDARY_AMINE = Chem.MolFromSmarts("[NX3;H1;!$(NC=O)]")
_TERTIARY_AMINE = Chem.MolFromSmarts("[NX3;H0;!$(NC=O);!$(N-[!#6])]")
_AMIDE = Chem.MolFromSmarts("[CX3](=[OX1])[NX3]")
_ACIDIC_GROUP = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1,O-]")
_BASIC_GROUP = Chem.MolFromSmarts(
    "[N;!$(N-C=O);!$(N-S(=O)=O);!$(N-[N,O,S]);!$(N#*)]"
)
_FILTER_CATALOGS = {}


def _resolve_column_name(columns, requested, aliases=()):
    if requested in columns:
        return requested
    lowered = {str(col).lower(): col for col in columns}
    for candidate in (requested, *aliases):
        match = lowered.get(str(candidate).lower())
        if match is not None:
            return match
    raise KeyError(
        f"Column '{requested}' not found. Available columns: {list(columns)}"
    )


def data_path_hash(data_path):
    normalized_path = os.path.abspath(os.path.expanduser(str(data_path)))
    return hashlib.sha1(normalized_path.encode("utf-8")).hexdigest()[:12]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_feature_version(
    use_cmf=False,
    fingerprint_configs=None,
    fingerprint_type=None,
    fingerprint_num_bits=None,
    fingerprint_max_count=None,
):
    """Build a cache feature version that changes only when the feature schema changes."""
    suffixes = [FEATURE_VERSION]
    if use_cmf:
        if fingerprint_configs is None:
            fingerprint_configs = [
                {
                    "name": "cmf",
                    "type": fingerprint_type or DEFAULT_FINGERPRINT_TYPE,
                    "num_bits": fingerprint_num_bits or DEFAULT_FINGERPRINT_NUM_BITS,
                    "max_count": fingerprint_max_count or DEFAULT_FINGERPRINT_MAX_COUNT,
                }
            ]
        canonical = json.dumps(fingerprint_configs, sort_keys=True, separators=(",", ":"))
        suffixes.extend(["fingerprints", hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]])
    return "__".join(suffixes)


def safe_index(items, value, fallback):
    try:
        return items.index(value)
    except ValueError:
        return fallback


def _safe_float(x, default=0.0):
    try:
        value = float(x)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return value


def _substructure_count(mol, pattern):
    if pattern is None:
        return 0.0
    return float(len(mol.GetSubstructMatches(pattern, uniquify=True)))


def _ring_system_stats(mol):
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    largest_ring = max((len(ring) for ring in rings), default=0)
    systems = []
    for ring in rings:
        overlapping = [idx for idx, system in enumerate(systems) if system & ring]
        if not overlapping:
            systems.append(set(ring))
            continue
        merged = set(ring)
        for idx in reversed(overlapping):
            merged.update(systems.pop(idx))
        systems.append(merged)
    return float(len(systems)), float(largest_ring)


def _partial_charge_extrema(mol):
    try:
        charged = Chem.Mol(mol)
        AllChem.ComputeGasteigerCharges(charged)
        charges = [
            _safe_float(atom.GetProp("_GasteigerCharge"), default=0.0)
            for atom in charged.GetAtoms()
        ]
        if charges:
            return max(charges), min(charges)
    except Exception:
        pass
    return 0.0, 0.0


def _filter_alert_count(mol, catalog_name):
    catalog = _FILTER_CATALOGS.get(catalog_name)
    if catalog is None:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, catalog_name))
        catalog = FilterCatalog.FilterCatalog(params)
        _FILTER_CATALOGS[catalog_name] = catalog
    try:
        return float(len(catalog.GetMatches(mol)))
    except Exception:
        return 0.0


def _compute_descriptor_groups(mol):
    """Return 46 stable 2D descriptors; no conformer generation is performed."""
    atoms = list(mol.GetAtoms())
    heavy = max(int(mol.GetNumHeavyAtoms()), 1)
    atomic_numbers = [atom.GetAtomicNum() for atom in atoms]
    formal_charge = float(sum(atom.GetFormalCharge() for atom in atoms))
    positive_atoms = float(sum(atom.GetFormalCharge() > 0 for atom in atoms))
    negative_atoms = float(sum(atom.GetFormalCharge() < 0 for atom in atoms))
    aromatic_atoms = float(sum(atom.GetIsAromatic() for atom in atoms))
    polar_atoms = float(sum(number in {7, 8} for number in atomic_numbers))
    ring_systems, largest_ring = _ring_system_stats(mol)
    primary_amines = _substructure_count(mol, _PRIMARY_AMINE)
    secondary_amines = _substructure_count(mol, _SECONDARY_AMINE)
    tertiary_amines = _substructure_count(mol, _TERTIARY_AMINE)
    acidic_groups = _substructure_count(mol, _ACIDIC_GROUP)
    basic_groups = _substructure_count(mol, _BASIC_GROUP)
    max_charge, min_charge = _partial_charge_extrema(mol)

    size_topology = [
        _safe_float(Descriptors.ExactMolWt(mol)),
        float(mol.GetNumHeavyAtoms()),
        float(mol.GetNumAtoms()),
        _safe_float(rdMolDescriptors.CalcNumHeteroatoms(mol)),
        _safe_float(rdMolDescriptors.CalcNumRings(mol)),
        _safe_float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        _safe_float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        _safe_float(rdMolDescriptors.CalcNumSaturatedRings(mol)),
        ring_systems,
        largest_ring,
    ]
    charge_composition = [
        formal_charge,
        positive_atoms,
        negative_atoms,
        float(atomic_numbers.count(7)),
        float(atomic_numbers.count(8)),
        float(atomic_numbers.count(16)),
        float(sum(number in {9, 17, 35, 53} for number in atomic_numbers)),
        primary_amines,
        secondary_amines,
        tertiary_amines,
        _substructure_count(mol, _AMIDE),
        acidic_groups,
        basic_groups,
        float(positive_atoms > 0 and negative_atoms > 0),
    ]
    polarity_surface = [
        _safe_float(rdMolDescriptors.CalcTPSA(mol)),
        _safe_float(rdMolDescriptors.CalcNumHBD(mol)),
        _safe_float(rdMolDescriptors.CalcNumHBA(mol)),
        _safe_float(Lipinski.NHOHCount(mol)),
        _safe_float(Lipinski.NOCount(mol)),
        _safe_float(rdMolDescriptors.CalcLabuteASA(mol)),
        polar_atoms / float(heavy),
    ]
    lipo_flex_complexity = [
        _safe_float(Crippen.MolLogP(mol)),
        _safe_float(Crippen.MolMR(mol)),
        aromatic_atoms / float(heavy),
        _safe_float(Lipinski.FractionCSP3(mol)),
        _safe_float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        _safe_float(GraphDescriptors.BertzCT(mol)),
        _safe_float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
        _safe_float(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
    ]
    electronic = [max_charge, min_charge]
    lipinski_violations = float(
        (_safe_float(Descriptors.ExactMolWt(mol)) > 500.0)
        + (_safe_float(Crippen.MolLogP(mol)) > 5.0)
        + (_safe_float(rdMolDescriptors.CalcNumHBD(mol)) > 5.0)
        + (_safe_float(rdMolDescriptors.CalcNumHBA(mol)) > 10.0)
    )
    try:
        qed = _safe_float(QED.qed(mol), default=0.0)
    except Exception:
        qed = 0.0
    druglike_alerts = [
        float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))),
        qed,
        lipinski_violations,
        _filter_alert_count(mol, "PAINS"),
        _filter_alert_count(mol, "BRENK"),
    ]

    desc = (
        size_topology
        + charge_composition
        + polarity_surface
        + lipo_flex_complexity
        + electronic
        + druglike_alerts
    )
    desc = [_safe_float(value, default=0.0) for value in desc]
    assert len(desc) == DESC_TOTAL_DIM, f"descriptor length {len(desc)} != {DESC_TOTAL_DIM}"
    return desc


def _fingerprint_output_dim(config):
    fp_type = str(config.get("type", DEFAULT_FINGERPRINT_TYPE)).lower()
    if fp_type == "maccs":
        return 167
    num_bits = int(config.get("num_bits", DEFAULT_FINGERPRINT_NUM_BITS))
    if fp_type in {"count_binary_morgan", "count+binary", "count_binary", "morgan_count_binary"}:
        return num_bits * 2
    return num_bits


def normalize_fingerprint_configs(dataset_config, enabled=True):
    if not enabled:
        return []
    raw_configs = dataset_config.get("fingerprints")
    if not raw_configs:
        raw_configs = [
            {
                "name": "cmf_r2",
                "type": dataset_config.get("fingerprint_type", DEFAULT_FINGERPRINT_TYPE),
                "radius": dataset_config.get("fingerprint_radius", DEFAULT_FINGERPRINT_RADIUS),
                "num_bits": dataset_config.get("fingerprint_num_bits", DEFAULT_FINGERPRINT_NUM_BITS),
                "max_count": dataset_config.get("fingerprint_max_count", DEFAULT_FINGERPRINT_MAX_COUNT),
                "log1p": dataset_config.get("fingerprint_log1p", True),
            }
        ]
    normalized = []
    seen = set()
    for index, raw in enumerate(raw_configs):
        cfg = dict(raw)
        name = str(cfg.get("name", f"fp_{index}"))
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError(f"Fingerprint name must contain only letters, digits, underscores: {name}")
        if name in seen:
            raise ValueError(f"Duplicate fingerprint name: {name}")
        seen.add(name)
        cfg["name"] = name
        cfg["type"] = str(cfg.get("type", DEFAULT_FINGERPRINT_TYPE)).lower()
        cfg["radius"] = int(cfg.get("radius", DEFAULT_FINGERPRINT_RADIUS))
        cfg["num_bits"] = int(cfg.get("num_bits", 167 if cfg["type"] == "maccs" else DEFAULT_FINGERPRINT_NUM_BITS))
        cfg["max_count"] = int(cfg.get("max_count", DEFAULT_FINGERPRINT_MAX_COUNT))
        cfg["log1p"] = bool(cfg.get("log1p", cfg["type"] in {"count_morgan", "count", "morgan_count", "cmf"}))
        cfg["dim"] = int(_fingerprint_output_dim(cfg))
        normalized.append(cfg)
    return normalized


def _compute_fingerprint(mol, config):
    fingerprint_type = str(config.get("type", DEFAULT_FINGERPRINT_TYPE)).lower()
    num_bits = int(config.get("num_bits", DEFAULT_FINGERPRINT_NUM_BITS))
    radius = int(config.get("radius", DEFAULT_FINGERPRINT_RADIUS))
    max_count = int(config.get("max_count", DEFAULT_FINGERPRINT_MAX_COUNT))

    if fingerprint_type in {"count_morgan", "count", "morgan_count", "cmf"}:
        fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
        values = np.zeros((num_bits,), dtype=np.uint8)
        for bit_idx, count in fp.GetNonzeroElements().items():
            values[int(bit_idx) % num_bits] = min(int(count), max_count)
        return values

    if fingerprint_type in {"binary_morgan", "binary", "bit", "01"}:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
        values = np.zeros((num_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, values)
        return values

    if fingerprint_type in {"count_binary_morgan", "count+binary", "count_binary", "morgan_count_binary"}:
        count_values = _compute_fingerprint(
            mol, {"type": "count_morgan", "radius": radius, "num_bits": num_bits, "max_count": max_count}
        )
        binary_values = _compute_fingerprint(
            mol, {"type": "binary_morgan", "radius": radius, "num_bits": num_bits}
        )
        return np.concatenate([count_values, binary_values], axis=0).astype(np.uint8, copy=False)

    if fingerprint_type in {"rdkit", "rdk", "rdkfingerprint"}:
        fp = Chem.RDKFingerprint(mol, fpSize=num_bits)
        values = np.zeros((num_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, values)
        return values

    if fingerprint_type in {"atom_pair", "atom_pair_hashed", "hashed_atom_pair"}:
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=num_bits)
        values = np.zeros((num_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, values)
        return values

    if fingerprint_type == "maccs":
        fp = MACCSkeys.GenMACCSKeys(mol)
        values = np.zeros((167,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, values)
        return values

    raise ValueError(f"Unsupported fingerprint_type: {fingerprint_type}")


def _compute_fingerprint_bundle(mol, fingerprint_configs):
    return {
        config["name"]: _compute_fingerprint(mol, config)
        for config in fingerprint_configs
    }


def _preprocess_pool_initializer():
    """子进程内收敛日志与线程数，避免 worker 互相抢资源。"""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    try:
        from rdkit import RDLogger, rdBase

        RDLogger.DisableLog("rdApp.*")
        rdBase.DisableLog("rdApp.*")
    except Exception:
        pass


def _effective_preprocess_workers(requested: int, n_items: int) -> int:
    """不超过样本数与 CPU 核数，避免 oversubscribe 导致内存与调度问题。"""
    cpu = int(os.cpu_count() or 1)
    req = max(1, int(requested))
    effective = min(req, max(1, n_items), max(1, cpu))
    if effective < req:
        print(
            f"⚠️ preprocess_num_processes={req} 已限幅为 {effective}（cpu_count={cpu}，样本数={n_items}）",
            flush=True,
        )
    return effective


def featurize_worker(args):
    idx, row, smiles_col, label_col, label_cols, fingerprint_config = args
    try:
        result = featurize_one(row, smiles_col, label_col, label_cols, fingerprint_config)
        if result is not None:
            result["source_idx"] = int(idx)
        return idx, result, None
    except Exception as exc:
        smiles = row.get(smiles_col, "") if isinstance(row, dict) else ""
        smiles_preview = str(smiles).replace("\n", " ")[:160]
        error = f"{type(exc).__name__}: {exc} | smiles={smiles_preview}"
        return idx, None, error

def featurize_one(row, smiles_col, label_col, label_cols=None, fingerprint_config=None):
    smiles = row[smiles_col]
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ---- Enhanced atom features (8 categorical / int columns) ----
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append([
            safe_index(ATOM_LIST, atom.GetAtomicNum(), len(ATOM_LIST)),
            safe_index(CHIRALITY_LIST, atom.GetChiralTag(), 0),
            safe_index(DEGREE_LIST, atom.GetTotalDegree(), len(DEGREE_LIST)),
            safe_index(FORMAL_CHARGE_LIST, atom.GetFormalCharge(), FORMAL_CHARGE_LIST.index(0)),
            safe_index(HYBRIDIZATION_LIST, atom.GetHybridization(), len(HYBRIDIZATION_LIST)),
            int(atom.GetIsAromatic()),
            safe_index(NUM_H_LIST, atom.GetTotalNumHs(), len(NUM_H_LIST)),
            int(atom.IsInRing()),
        ])

    # ---- Enhanced bond features (6 columns) ----
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
        bond_feat = [
            safe_index(BOND_LIST, bond.GetBondType(), len(BOND_LIST)),
            safe_index(BONDDIR_LIST, bond.GetBondDir(), 0),
            safe_index(BONDSTEREO_LIST, bond.GetStereo(), 0),
            int(bond.GetIsConjugated()),
            int(bond.GetIsAromatic()),
            int(bond.IsInRing()),
        ]
        edge_attr.extend([bond_feat, bond_feat])

    # ---- Stable 46-dimensional 2D descriptors ----
    prop = _compute_descriptor_groups(mol)

    fingerprint_config = fingerprint_config or {}
    fingerprints = None
    if fingerprint_config.get("enabled", False):
        fingerprints = _compute_fingerprint_bundle(
            mol,
            fingerprint_config.get("fingerprints", []),
        )

    if label_cols:
        label = {}
        for task_col in label_cols:
            if task_col in row and pd.notna(row[task_col]) and row[task_col] != -1:
                label[task_col] = float(row[task_col])
            else:
                label[task_col] = -1.0
    else:
        label = float(row[label_col])

    return {
        "atom_feats": atom_feats,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "prop": prop,
        "fingerprints": fingerprints,
        "label": label,
        "smiles": smiles
    }

def preprocess(
    data_path,
    cache_path,
    smiles_col,
    label_col,
    num_processes=12,
    chunksize=None,
    maxtasksperchild=None,
    fingerprint_config=None,
):
    return preprocess_with_labels(
        data_path,
        cache_path,
        smiles_col,
        label_col,
        None,
        num_processes,
        chunksize=chunksize,
        maxtasksperchild=maxtasksperchild,
        fingerprint_config=fingerprint_config,
    )

def preprocess_with_labels(
    data_path,
    cache_path,
    smiles_col,
    label_col,
    label_cols=None,
    num_processes=12,
    chunksize=None,
    maxtasksperchild=None,
    fingerprint_config=None,
):
    normalized_label_cols = list(label_cols or [])
    source_path_hash = data_path_hash(data_path)
    fingerprint_config = dict(fingerprint_config or {})
    use_cmf = bool(fingerprint_config.get("enabled", False))
    cache_feature_version = build_feature_version(
        use_cmf=use_cmf,
        fingerprint_configs=fingerprint_config.get("fingerprints", []),
    )
    if os.path.exists(cache_path):
        print(f"✅ 加载缓存 {cache_path}")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            isinstance(cached, dict)
            and cached.get("feature_version") == cache_feature_version
            and cached.get("label_col") == label_col
            and cached.get("label_cols", []) == normalized_label_cols
            and cached.get("data_path_hash") == source_path_hash
        ):
            return cached["data"]
        if isinstance(cached, list):
            print("⚠️ 发现旧版缓存，将重新构建")

    df = pd.read_csv(data_path)
    df.columns = [col.lstrip("\ufeff") for col in df.columns]
    smiles_col = _resolve_column_name(df.columns, smiles_col, aliases=("SMILES", "smiles"))
    label_col = _resolve_column_name(df.columns, label_col)
    normalized_label_cols = [
        _resolve_column_name(df.columns, task_col)
        for task_col in normalized_label_cols
    ]
    records = df.to_dict("records")
    n_records = len(records)
    num_workers = _effective_preprocess_workers(num_processes, n_records)
    # 主进程若已初始化 CUDA，fork 子进程会不稳定；spawn 独立解释器更安全（略慢启动）。
    use_fork = os.environ.get("DATASET_PREPROCESS_FORK", "").strip().lower() in ("1", "true", "yes")
    pool_ctx = mp if use_fork else mp.get_context("spawn")
    if chunksize is None:
        chunk = max(1, min(512, n_records // max(8, num_workers * 4) or 1))
    else:
        chunk = max(1, int(chunksize))
    pool_options = {"initializer": _preprocess_pool_initializer}
    if maxtasksperchild is not None:
        pool_options["maxtasksperchild"] = max(1, int(maxtasksperchild))
    print(
        f"预处理 {n_records} 条数据...  {num_workers} 进程"
        f"（Pool 后端={'fork' if use_fork else 'spawn'}，chunksize={chunk}，"
        f"maxtasksperchild={pool_options.get('maxtasksperchild')}）",
        flush=True,
    )
    worker_args = (
        (idx, row, smiles_col, label_col, normalized_label_cols, fingerprint_config)
        for idx, row in enumerate(records)
    )
    results = [None] * n_records
    failed = []
    with pool_ctx.Pool(
        num_workers,
        **pool_options,
    ) as p:
        for idx, result, error in tqdm(
            p.imap_unordered(featurize_worker, worker_args, chunksize=chunk),
            total=n_records,
            file=sys.stdout,
            mininterval=0.5,
            dynamic_ncols=False,
            ascii=True,
        ):
            if error is not None:
                failed.append((idx, error))
                continue
            results[idx] = result

    if failed:
        print(f"⚠️ 预处理失败 {len(failed)} 条，已跳过。前 5 条错误：", flush=True)
        for idx, error in failed[:5]:
            print(f"  idx={idx}: {error}", flush=True)

    data = [r for r in results if r is not None]
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_cache_path = f"{cache_path}.tmp.{os.getpid()}"
    try:
        torch.save(
            {
                "data": data,
                "feature_version": cache_feature_version,
                "smiles_col": smiles_col,
                "label_col": label_col,
                "label_cols": normalized_label_cols,
                "data_path": os.path.abspath(os.path.expanduser(str(data_path))),
                "data_path_hash": source_path_hash,
                "fingerprint_config": fingerprint_config,
            },
            tmp_cache_path,
        )
        os.replace(tmp_cache_path, cache_path)
    except Exception:
        if os.path.exists(tmp_cache_path):
            os.remove(tmp_cache_path)
        raise
    print(f"✅ 缓存保存至 {cache_path}")
    return data

def stratified_split(
    dataset,
    valid_size=0.1,
    test_size=0.1,
    seed=42,
    primary_task="label_tolC",
    bucket_size=10,
):
    """GNEprop's SingleTaskStratifiedSplitter algorithm."""
    train_size = 1.0 - float(valid_size) - float(test_size)
    sizes = (train_size, float(valid_size), float(test_size))
    if not math.isclose(sum(sizes), 1.0, abs_tol=1e-3):
        raise ValueError(f"Split fractions must sum to one, got {sizes}")

    labels = np.asarray(
        [
            (
                np.nan
                if _primary_label_value(item["label"], primary_task) is None
                else _primary_label_value(item["label"], primary_task)
            )
            for item in dataset.data
        ],
        dtype=float,
    )
    sorted_indices = np.argsort(labels)
    np.random.seed(seed)

    train_bucket_cutoff = int(np.round(train_size * bucket_size))
    valid_bucket_cutoff = (
        int(np.round(float(valid_size) * bucket_size)) + train_bucket_cutoff
    )
    train_idx, valid_idx, test_idx = [], [], []

    while sorted_indices.shape[0] >= bucket_size:
        current_batch, sorted_indices = np.split(sorted_indices, [bucket_size])
        shuffled = np.random.permutation(range(bucket_size))
        train_idx.extend(
            current_batch[shuffled[:train_bucket_cutoff]].tolist()
        )
        valid_idx.extend(
            current_batch[
                shuffled[train_bucket_cutoff:valid_bucket_cutoff]
            ].tolist()
        )
        test_idx.extend(
            current_batch[shuffled[valid_bucket_cutoff:]].tolist()
        )

    # GNEprop places any final incomplete bucket in the training split.
    train_idx.extend(sorted_indices.tolist())
    return train_idx, valid_idx, test_idx


def _primary_label_value(label, primary_task):
    if isinstance(label, dict):
        value = label.get(primary_task, -1.0)
    else:
        value = label
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value == -1.0 or not np.isfinite(value):
        return None
    return value


def stratified_scaffold_split(
    dataset,
    valid_size=0.1,
    test_size=0.1,
    seed=42,
    primary_task="label_tolC",
):
    """GNEprop's non-chiral scaffold_stratified split algorithm."""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    train_fraction = 1.0 - float(valid_size) - float(test_size)
    sizes = (train_fraction, float(valid_size), float(test_size))
    if not math.isclose(sum(sizes), 1.0, abs_tol=1e-3):
        raise ValueError(f"Split fractions must sum to one, got {sizes}")

    train_size = int(sizes[0] * len(dataset))
    valid_target_size = int(sizes[1] * len(dataset))
    test_target_size = int(sizes[2] * len(dataset))
    if train_size == valid_target_size == 0:
        return [], [], list(range(len(dataset)))

    scaffolds = {}
    for index, item in enumerate(dataset.data):
        mol = Chem.MolFromSmiles(item["smiles"])
        if mol is None:
            raise ValueError(
                f"Cannot compute GNEprop scaffold for dataset index {index}"
            )
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False,
        )
        scaffolds.setdefault(scaffold, set()).add(index)

    labels = np.asarray(
        [
            (
                np.nan
                if _primary_label_value(item["label"], primary_task) is None
                else _primary_label_value(item["label"], primary_task)
            )
            for item in dataset.data
        ],
        dtype=float,
    )

    rng = random.Random(seed)
    big_index_sets, small_index_sets = [], []
    for index_set in scaffolds.values():
        if (
            len(index_set) > valid_target_size / 2
            and len(index_set) > test_target_size / 2
        ):
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    rng.seed(seed)
    rng.shuffle(big_index_sets)
    rng.shuffle(small_index_sets)

    train_idx = list(itertools.chain.from_iterable(big_index_sets))
    test_idx, valid_idx = [], []
    groups = [
        {
            "set": train_idx,
            "size": train_size,
            "positive_count": int(np.sum(labels[train_idx] == 1)),
        },
        {"set": test_idx, "size": test_target_size, "positive_count": 0},
        {"set": valid_idx, "size": valid_target_size, "positive_count": 0},
    ]

    small_index_lists = [list(index_set) for index_set in small_index_sets]
    sorted_group_indices = np.argsort(
        [np.sum(labels[index_set] == 1) for index_set in small_index_lists]
    )
    sorted_index_lists = [
        small_index_lists[index] for index in sorted_group_indices
    ]
    sorted_index_lists.reverse()

    for index_set in sorted_index_lists:
        ratios = [
            group["positive_count"] / (group["size"] + 1e-3)
            for group in groups
        ]
        positive_count = int(np.sum(labels[index_set] == 1))
        sort_index = np.arange(len(ratios))
        if (
            math.isclose(ratios[0], ratios[1], rel_tol=1e-3)
            and math.isclose(ratios[1], ratios[2], rel_tol=1e-3)
        ):
            rng.shuffle(sort_index)
        else:
            sort_index = np.argsort(ratios)

        priority_groups = [groups[index] for index in sort_index]
        if (
            len(priority_groups[0]["set"]) + len(index_set)
            <= priority_groups[0]["size"]
        ):
            destination = priority_groups[0]
        elif (
            len(priority_groups[1]["set"]) + len(index_set)
            <= priority_groups[1]["size"]
        ):
            destination = priority_groups[1]
        else:
            destination = priority_groups[2]
        destination["set"] += index_set
        destination["positive_count"] += positive_count

    if valid_target_size == 0:
        return groups[0]["set"], [], groups[1]["set"]
    if test_target_size == 0:
        return groups[0]["set"], groups[2]["set"], []
    return groups[0]["set"], groups[2]["set"], groups[1]["set"]


def _collect_task_label_stats(data, indices, task_cols):
    """对给定 indices 切片，按任务统计 N / pos / pos_rate（忽略 -1 缺失）。

    返回:
        { task: {"N": int, "pos": int, "pos_rate": float, "neg": int, "missing": int} }
    """
    stats = {}
    for task in task_cols:
        n_valid = 0
        n_pos = 0
        n_missing = 0
        for idx in indices:
            if idx >= len(data):
                continue
            label = data[idx]["label"]
            if isinstance(label, dict):
                value = label.get(task, -1.0)
            else:
                value = label
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                n_missing += 1
                continue
            if value_f == -1.0 or not np.isfinite(value_f):
                n_missing += 1
                continue
            n_valid += 1
            if value_f >= 0.5:
                n_pos += 1
        n_neg = n_valid - n_pos
        pos_rate = (n_pos / n_valid) if n_valid > 0 else float("nan")
        stats[task] = {
            "N": int(n_valid),
            "pos": int(n_pos),
            "neg": int(n_neg),
            "missing": int(n_missing),
            "pos_rate": float(pos_rate),
        }
    return stats


def report_split_label_stats(
    dataset,
    train_idx,
    valid_idx,
    test_idx,
    task_cols=None,
    primary_task=None,
    save_dir=None,
    split_method="scaffold",
    seed=None,
):
    """scaffold split 完成后立即打印并保存各 split × task 的样本数 / 正例率。"""
    if task_cols is None or len(task_cols) == 0:
        task_cols = getattr(dataset, "task_columns", []) or ["label"]

    splits = {
        "train": list(train_idx),
        "valid": list(valid_idx),
        "test": list(test_idx),
    }

    data = dataset.data
    header = "=" * 72
    print(header, flush=True)
    print(
        f"[Split Report] split_method={split_method} | seed={seed} | "
        f"total={len(data)} | train={len(splits['train'])} | "
        f"valid={len(splits['valid'])} | test={len(splits['test'])}",
        flush=True,
    )
    print(header, flush=True)

    aggregated = {}
    for split_name, indices in splits.items():
        task_stats = _collect_task_label_stats(data, indices, task_cols)
        aggregated[split_name] = task_stats
        for task, st in task_stats.items():
            # 格式：train label_tolC: N=..., pos=..., pos_rate=0.0253
            print(
                f"{split_name:>5} {task}: N={st['N']}, pos={st['pos']}, "
                f"neg={st['neg']}, missing={st['missing']}, "
                f"pos_rate={st['pos_rate']:.4f}",
                flush=True,
            )
        print("-" * 72, flush=True)

    # 保存到 experiment_records 目录
    if save_dir:
        try:
            import yaml as _yaml

            os.makedirs(save_dir, exist_ok=True)
            payload = {
                "split_method": split_method,
                "seed": None if seed is None else int(seed),
                "totals": {
                    "dataset": int(len(data)),
                    "train": int(len(splits["train"])),
                    "valid": int(len(splits["valid"])),
                    "test": int(len(splits["test"])),
                },
                "primary_task": primary_task,
                "tasks": list(task_cols),
                "per_split": {
                    split_name: {
                        task: {
                            "N": int(st["N"]),
                            "pos": int(st["pos"]),
                            "neg": int(st["neg"]),
                            "missing": int(st["missing"]),
                            "pos_rate": float(st["pos_rate"]),
                        }
                        for task, st in task_stats.items()
                    }
                    for split_name, task_stats in aggregated.items()
                },
            }
            with open(os.path.join(save_dir, "split_stats.yaml"), "w", encoding="utf-8") as f:
                _yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

            # split_indices.pt
            torch.save(
                {
                    "train_idx": torch.as_tensor(splits["train"], dtype=torch.long),
                    "valid_idx": torch.as_tensor(splits["valid"], dtype=torch.long),
                    "test_idx": torch.as_tensor(splits["test"], dtype=torch.long),
                    "split_method": split_method,
                    "seed": None if seed is None else int(seed),
                    "tasks": list(task_cols),
                    "primary_task": primary_task,
                },
                os.path.join(save_dir, "split_indices.pt"),
            )
            print(
                f"[Split Report] saved -> {os.path.join(save_dir, 'split_stats.yaml')} "
                f"and split_indices.pt",
                flush=True,
            )
        except Exception as exc:  # 不因记录失败影响训练
            print(f"[Split Report] 保存失败: {exc}", flush=True)

    return aggregated

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split="train"):
        data_path = config["dataset"]["data_path"]
        self.data_path = os.path.abspath(os.path.expanduser(str(data_path)))
        target = config["dataset"]["target"]
        smiles_col = config["dataset"].get("smiles_col", "SMILES")
        num_processes = int(config["dataset"].get("preprocess_num_processes", 12))
        chunksize = config["dataset"].get("preprocess_chunksize")
        maxtasksperchild = config["dataset"].get("preprocess_maxtasksperchild")
        self.use_cmf = True
        self.fingerprint_configs = normalize_fingerprint_configs(
            config["dataset"],
            enabled=True,
        )
        self.fingerprint_specs = [
            {
                "name": fp_config["name"],
                "type": fp_config["type"],
                "dim": int(fp_config["dim"]),
            }
            for fp_config in self.fingerprint_configs
        ]
        self.fingerprint_dim = int(sum(spec["dim"] for spec in self.fingerprint_specs))
        if self.use_cmf:
            model_config = config.setdefault("model", {})
            model_config["fingerprint_dim"] = int(self.fingerprint_dim)
            model_config["fingerprint_specs"] = list(self.fingerprint_specs)
        self.fingerprint_config = {
            "enabled": self.use_cmf,
            "fingerprints": self.fingerprint_configs,
        }
        cache_feature_version = build_feature_version(
            use_cmf=self.use_cmf,
            fingerprint_configs=self.fingerprint_configs,
        )
        cache_key = data_path_hash(data_path)
        if self.use_cmf:
            fp_names = "-".join(config_item["name"] for config_item in self.fingerprint_configs)
            cache_key = f"{cache_key}_{fp_names}"
        cache_key = f"{cache_key}_{hashlib.sha1(cache_feature_version.encode('utf-8')).hexdigest()[:8]}"
        cache_path = f"data/processed/cached_{config['task_name']}_{target}_{cache_key}.pt"
        self.descriptor_names = list(DESC_NAMES)
        self.descriptor_group_dims = list(DESC_GROUP_DIMS)
        self.descriptor_scaling = str(config["dataset"].get("descriptor_scaling", "robust_train")).lower()
        self.descriptor_clip = float(config["dataset"].get("descriptor_clip", 10.0))
        self.descriptor_center = np.zeros(DESC_TOTAL_DIM, dtype=np.float32)
        self.descriptor_scale = np.ones(DESC_TOTAL_DIM, dtype=np.float32)
        self.descriptor_scaler_state = None
        model_config = config.setdefault("model", {})
        model_config["use_desc"] = bool(model_config.get("use_desc", model_config.get("use_3d", True)))
        model_config["prop_dim"] = DESC_TOTAL_DIM
        model_config["desc_group_dims"] = list(DESC_GROUP_DIMS)
        model_config["descriptor_names"] = list(DESC_NAMES)
        self.task_columns = [target]
        self.splitting = config["dataset"]["splitting"]
        if self.splitting not in {
            "stratified",
            "stratified_scaffold",
            "fixed_indices",
        }:
            raise ValueError(
                "Only stratified, stratified_scaffold, and fixed_indices are supported"
            )
        self.fixed_split_manifest = config["dataset"].get("fixed_split_manifest")
        if self.splitting == "fixed_indices" and not self.fixed_split_manifest:
            raise ValueError(
                "dataset.fixed_split_manifest is required when splitting=fixed_indices"
            )
        self.data = preprocess_with_labels(
            data_path,
            cache_path,
            smiles_col,
            target,
            self.task_columns,
            num_processes,
            chunksize=chunksize,
            maxtasksperchild=maxtasksperchild,
            fingerprint_config=self.fingerprint_config,
        )
        self.valid_size = config["dataset"]["valid_size"]
        self.test_size = config["dataset"]["test_size"]
        self.global_batch_size = int(config["dataset"]["batch_size"])
        self.batch_size = self.global_batch_size
        self.num_workers = config["dataset"]["num_workers"]
        self.seed = config["dataset"].get("seed", 42)
        self.primary_task = config["dataset"].get(
            "primary_task",
            target,
        )

    def __len__(self):
        return len(self.data)

    def _normalize_fingerprint_tensor(self, values, config):
        tensor = torch.as_tensor(values, dtype=torch.float)
        fp_type = str(config["type"]).lower()
        max_count = max(int(config.get("max_count", DEFAULT_FINGERPRINT_MAX_COUNT)), 1)
        if fp_type in {"count_binary_morgan", "count+binary", "count_binary", "morgan_count_binary"}:
            num_bits = int(config["num_bits"])
            count_part = tensor[:num_bits]
            binary_part = tensor[num_bits:].clamp(min=0.0, max=1.0)
            if config.get("log1p", True):
                count_part = torch.log1p(count_part.clamp(min=0.0, max=float(max_count)))
                count_part = count_part / float(np.log1p(max_count))
            return torch.cat([count_part, binary_part], dim=0)
        if fp_type in {"count_morgan", "count", "morgan_count", "cmf"} and config.get("log1p", True):
            tensor = torch.log1p(tensor.clamp(min=0.0, max=float(max_count)))
            tensor = tensor / float(np.log1p(max_count))
        return tensor

    def fit_descriptor_scaler(self, train_indices, save_dir=None):
        train_indices = list(train_indices)
        if not train_indices:
            raise ValueError("Cannot fit descriptor scaler without training samples")
        matrix = np.asarray([self.data[index]["prop"] for index in train_indices], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != DESC_TOTAL_DIM:
            raise ValueError(
                f"Descriptor matrix must have shape [N, {DESC_TOTAL_DIM}], got {matrix.shape}"
            )
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        if self.descriptor_scaling in {"none", "off", "false"}:
            center = np.zeros(DESC_TOTAL_DIM, dtype=np.float64)
            scale = np.ones(DESC_TOTAL_DIM, dtype=np.float64)
            method = "none"
        elif self.descriptor_scaling in {"robust", "robust_train", "robust_scaler"}:
            center = np.median(matrix, axis=0)
            q1 = np.percentile(matrix, 25.0, axis=0)
            q3 = np.percentile(matrix, 75.0, axis=0)
            scale = q3 - q1
            scale[~np.isfinite(scale) | (np.abs(scale) < 1e-8)] = 1.0
            method = "median_iqr_train_only"
        else:
            raise ValueError(f"Unsupported dataset.descriptor_scaling: {self.descriptor_scaling}")
        self.descriptor_center = center.astype(np.float32)
        self.descriptor_scale = scale.astype(np.float32)
        self.descriptor_scaler_state = {
            "method": method,
            "fit_split": "train",
            "train_count": len(train_indices),
            "feature_names": list(self.descriptor_names),
            "center": self.descriptor_center.tolist(),
            "scale": self.descriptor_scale.tolist(),
            "clip": float(self.descriptor_clip),
        }
        if save_dir:
            import yaml as _yaml

            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "descriptor_scaler.yaml"), "w", encoding="utf-8") as handle:
                _yaml.safe_dump(
                    self.descriptor_scaler_state,
                    handle,
                    allow_unicode=True,
                    sort_keys=False,
                )
        return self.descriptor_scaler_state

    def __getitem__(self, idx):
        d = self.data[idx]
        atom_feats = torch.tensor(d["atom_feats"], dtype=torch.long)
        edge_index = torch.tensor(d["edge_index"], dtype=torch.long).t().contiguous() if d["edge_index"] else torch.empty((2,0), dtype=torch.long)
        if d["edge_attr"]:
            edge_attr = torch.tensor(d["edge_attr"], dtype=torch.long)
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(1, -1)
            if edge_attr.size(1) != BOND_FEATURE_DIM:
                fixed_edge_attr = torch.zeros((edge_attr.size(0), BOND_FEATURE_DIM), dtype=torch.long)
                width = min(edge_attr.size(1), BOND_FEATURE_DIM)
                fixed_edge_attr[:, :width] = edge_attr[:, :width]
                edge_attr = fixed_edge_attr
        else:
            edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.long)
        raw_prop = torch.tensor(d["prop"], dtype=torch.float)
        graph = Data(x=atom_feats, edge_index=edge_index, edge_attr=edge_attr)
        graph.raw_prop = raw_prop.view(1, -1)
        graph.sample_idx = torch.tensor([int(idx)], dtype=torch.long)
        raw_bundle = d.get("fingerprints")
        if not isinstance(raw_bundle, dict):
            raise ValueError("Cache is missing the fingerprint bundle")
        parts = []
        for fp_config in self.fingerprint_configs:
            values = raw_bundle.get(fp_config["name"])
            if values is None:
                raise ValueError(
                    f"Cache is missing fingerprint {fp_config['name']}"
                )
            parts.append(
                self._normalize_fingerprint_tensor(values, fp_config)
            )
        cmf = torch.cat(parts, dim=0)
        label = d["label"]
        if not isinstance(label, dict) or self.primary_task not in label:
            raise ValueError("Cache is missing the configured single-task label")
        label = {
            self.primary_task: torch.tensor(
                label[self.primary_task],
                dtype=torch.float,
            )
        }
        center = torch.from_numpy(self.descriptor_center)
        scale = torch.from_numpy(self.descriptor_scale)
        prop = (raw_prop - center) / scale
        prop = torch.nan_to_num(prop, nan=0.0, posinf=self.descriptor_clip, neginf=-self.descriptor_clip)
        prop = prop.clamp(min=-self.descriptor_clip, max=self.descriptor_clip)
        smiles = d["smiles"]
        return graph, prop, cmf, label, smiles

    def _load_fixed_split_indices(self, seed):
        manifest_path = os.path.abspath(
            os.path.expanduser(str(self.fixed_split_manifest))
        )
        payload = torch.load(
            manifest_path,
            map_location="cpu",
            weights_only=False,
        )
        expected_size = int(payload.get("dataset_size", -1))
        if expected_size != len(self):
            raise ValueError(
                f"Fixed split dataset size mismatch: manifest={expected_size}, "
                f"loaded={len(self)}"
            )
        expected_sha256 = payload.get("dataset_sha256")
        if expected_sha256:
            actual_sha256 = file_sha256(self.data_path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    "Fixed split dataset SHA256 mismatch: "
                    f"manifest={expected_sha256}, loaded={actual_sha256}"
                )

        seed_splits = payload.get("seeds", {})
        split_payload = seed_splits.get(int(seed))
        if split_payload is None:
            split_payload = seed_splits.get(str(int(seed)))
        if split_payload is None:
            raise KeyError(
                f"Seed {seed} is absent from fixed split manifest {manifest_path}"
            )

        split_indices = {}
        for split_name in ("train", "valid", "test"):
            key = f"{split_name}_idx"
            if key not in split_payload:
                raise KeyError(f"Fixed split manifest is missing {key}")
            values = torch.as_tensor(
                split_payload[key],
                dtype=torch.long,
            ).view(-1).tolist()
            if len(values) != len(set(values)):
                raise ValueError(
                    f"Fixed split {split_name} contains duplicate indices"
                )
            if values and (min(values) < 0 or max(values) >= len(self)):
                raise IndexError(
                    f"Fixed split {split_name} contains out-of-range indices"
                )
            split_indices[split_name] = values

        train_set = set(split_indices["train"])
        valid_set = set(split_indices["valid"])
        test_set = set(split_indices["test"])
        if train_set & valid_set or train_set & test_set or valid_set & test_set:
            raise ValueError("Fixed train/valid/test splits overlap")
        if train_set | valid_set | test_set != set(range(len(self))):
            raise ValueError("Fixed train/valid/test splits do not cover the dataset")

        source = str(payload.get("split_source", "external"))
        split_kind = str(payload.get("split_kind", "fixed"))
        method = f"fixed_{source}_{split_kind}"
        print(
            f"[Fixed Split] method={method} | seed={int(seed)} | "
            f"manifest={manifest_path}",
            flush=True,
        )
        return (
            split_indices["train"],
            split_indices["valid"],
            split_indices["test"],
            method,
        )

    def get_train_validation_data_loaders(
        self,
        seed=None,
        num_workers=None,
        pin_memory=False,
        split_report_dir=None,
        primary_task=None,
    ):
        effective_workers = (
            self.num_workers if num_workers is None else int(num_workers)
        )
        loader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": effective_workers,
            "drop_last": False,
            "pin_memory": bool(pin_memory),
        }
        if effective_workers > 0:
            loader_kwargs["persistent_workers"] = True

        split_seed = self.seed if seed is None else int(seed)
        split_task = primary_task or self.primary_task
        if self.splitting == "fixed_indices":
            train_idx, valid_idx, test_idx, split_method = (
                self._load_fixed_split_indices(split_seed)
            )
        elif self.splitting == "stratified":
            split_method = "stratified"
            train_idx, valid_idx, test_idx = stratified_split(
                self, self.valid_size, self.test_size,
                seed=split_seed, primary_task=split_task,
            )
        else:
            split_method = "scaffold_stratified"
            train_idx, valid_idx, test_idx = stratified_scaffold_split(
                self, self.valid_size, self.test_size,
                seed=split_seed, primary_task=split_task,
            )

        self.fit_descriptor_scaler(train_idx, save_dir=split_report_dir)
        report_split_label_stats(
            self, train_idx, valid_idx, test_idx,
            task_cols=[self.primary_task],
            primary_task=self.primary_task,
            save_dir=split_report_dir,
            split_method=split_method,
            seed=split_seed,
        )
        train_loader = DataLoader(
            self, sampler=SubsetRandomSampler(train_idx), **loader_kwargs
        )
        valid_loader = DataLoader(
            Subset(self, valid_idx), shuffle=False, **loader_kwargs
        )
        test_loader = DataLoader(
            Subset(self, test_idx), shuffle=False, **loader_kwargs
        )
        return train_loader, valid_loader, test_loader
