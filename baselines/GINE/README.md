# GINE supervised molecular-classification baseline

This supplement contains the unchanged GINE training and aggregation scripts
from gine_baseline_package-duibi. GINE is a comparison baseline, NOT a
contrastive-learning method: training uses supervised weighted binary cross
entropy, without contrastive pretraining, paired views or a contrastive loss.
The reference experiment family is `log-shengao`. Confirm that its results
match the manuscript tables before submission.

## Contents and provenance

- `scripts/`: original training and summary scripts, copied byte-for-byte.
- `configs/<task>/<split>/seed*.json`: all 56 actual argument dictionaries
  extracted from historical best_model.pt checkpoints; only paths are rewritten.
  Checkpoint SHA256 values and feature dimensions are recorded as provenance.
- `configs/observed_training_parameters.json`: unique observed argument values.
- `data/`: four original input CSVs in their original row order.
- `splits/`: original fixed index files for 4 tasks x 2 split types x 7 seeds.
- `reference_results/`: historical predictions, metrics, training histories and
  split manifests; these are not results from a new run.
- `verification/audit.json`: split, prediction, checkpoint and metric checks.
- `requirements.txt`, `environment_snapshot.json`, `MANIFEST.sha256`:
  pinned direct dependencies, packaging environment and file hashes.

Trained weight files are not included in this source/configuration supplement.
Training creates new best_model.pt files. Existing checkpoints were loaded on
CPU to extract settings and verify compatibility, without changing the originals.
Dataset redistribution rights and original dataset citations must be confirmed
by the authors before public release. No new data license is implied.

## Installation and execution

Use Python 3.9 (packaging environment: 3.9.20). From the extracted folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_baseline.py --task GNEtolC --split balanced_random --seed 42 --output results/GNEtolC/balanced_random/seed42 --check-only
python run_baseline.py --task GNEtolC --split balanced_random --seed 42 --output results/GNEtolC/balanced_random/seed42
```

The launcher defaults to CPU and refuses an existing output directory. It uses
the saved run-specific hyperparameters, with device and output path overridden.
`--num-workers 0` can be used on systems with multiprocessing restrictions.
`--check-only` checks imports and input-file presence; it does not train.
CPU training can be slow. To use an available GPU, select it according to the
local cluster allocation rules using CUDA_VISIBLE_DEVICES, then add
`--device cuda`. Do not copy another user's GPU assignment.

The original environment uses torch 2.2.0+cu118 and torch-geometric 2.6.1.
requirements.txt pins the public torch version 2.2.0. For CUDA execution,
install the matching PyTorch/PyG build for your hardware. Optional PyG compiled
extensions, if installed, must match that build. A fresh installation has not
been tested; the CPU forward checks used the existing environment.

Run all configurations sequentially (56 training runs; not a smoke test):

```bash
for task in GNEtolC HepG2 HSkMC IMR90; do
  for split in balanced_random balanced_scaffold; do
    for seed in 42 43 44 45 46 47 48; do
      python run_baseline.py --task "$task" --split "$split" --seed "$seed" --output "results/$task/$split/seed$seed"
    done
  done
done
python scripts/summarize_gine_results.py --root results --out results/gine_grouped_metrics.csv
```

## Graph features and architecture

RDKit parses each SMILES into a molecular graph. There is no ECFP or descriptor
branch. Each bond is represented by two directed edges. Atom features (13):
atomic number/100, total degree/6, formal charge/5, total H count/8, implicit
valence/8, five hybridization indicators (SP, SP2, SP3, SP3D, SP3D2), aromaticity,
ring membership, and a binary indicator of specified chirality. The latter does
not encode full stereochemical identity. Bond features (7): four bond-type
indicators (single/double/triple/aromatic), conjugation, ring membership, and
the integer RDKit stereo enum/6. No explicit hydrogen addition is performed.

Atom encoder: Linear(13,256), BatchNorm, ReLU, dropout 0.2. Four residual GINE
blocks each use a 256->512->256 ReLU MLP, learned edge projection from 7 features,
BatchNorm, ReLU and dropout 0.2, followed by residual addition. PyG GINEConv
defaults apply (add aggregation, eps=0, train_eps=False). Graph readout is global
mean pooling. Classification head: Linear(256,256), ReLU, dropout 0.2,
Linear(256,1). Sigmoid converts logits to probabilities for evaluation.

## Training configuration

The per-run checkpoint-derived JSON files are authoritative. The expected
settings, verified against the extracted configurations, are:

| Setting                     | Value                                                        |
| --------------------------- | ------------------------------------------------------------ |
| Maximum epochs              | 120                                                          |
| Batch size                  | 128                                                          |
| Optimizer                   | AdamW, learning rate 0.0003, weight decay 0.00001            |
| AdamW other defaults        | betas=(0.9,0.999), eps=1e-8, amsgrad=False                   |
| Scheduler                   | CosineAnnealingLR, T_max=120, eta_min=0; stepped after each training epoch |
| Loss                        | BCEWithLogitsLoss, default mean reduction                    |
| Positive-class weight       | clip(n_negative/n_positive, 1, 50); each count is first clamped to at least 1 |
| Gradient clipping           | global norm 5.0                                              |
| Hidden dimension / layers   | 256 / 4                                                      |
| Dropout / pooling           | 0.2 / mean                                                   |
| Early-stopping patience     | 25 consecutive non-improving validation epochs               |
| Classification threshold    | 0.5                                                          |
| Seeds                       | 42,43,44,45,46,47,48                                         |
| Historical device / workers | CUDA / 2                                                     |

Checkpoint selection and early stopping maximize validation
`0.75 * AUPRC + 0.25 * BEDROC20`, falling back to AUPRC when BEDROC20 is NaN.
Strict improvement is required; ties retain the earlier checkpoint. The original
script's opening docstring says AUPRC, but the executable implementation uses
this composite. The best checkpoint is reloaded before test evaluation. Test
metrics do not drive selection. There is no hyperparameter search implemented;
this alone does not establish how the authors originally chose these settings.

Python, NumPy and PyTorch RNGs are seeded. However, cuDNN deterministic=False
and benchmark=True; strict deterministic GPU reproduction is not guaranteed.

## Inputs, partitions and metrics

GNEtolC uses SMILES/Activity_tolC; HepG2, HSkMC and IMR90 use smiles/label.
Missing SMILES/labels and nonnumeric labels are removed; labels are cast to
integer, -1 becomes 0, then positive values become 1. The row index is reset.
No SMILES canonicalization or deduplication is performed. Invalid/empty graphs
are skipped by the original dataset loader; the historical prediction checks
verify that all expected test rows survived. Do not change data row order.

Fixed external partitions override newly generated splits. Nominal fractions
are 80/10/10; exact sizes and class counts are in the audit. The balanced_random
and balanced_scaffold names do not imply 1:1 class resampling. Indices and input
data were matched against the RF/XGBoost/LightGBM log-shengao family, including
test molecule/label alignment. GINE outputs test rows in input-row order;
the ECFP runner outputs them in split-index order, so compare by row_id.

AUPRC is sklearn average precision, not trapezoidal PR integration. BEDROC20
uses RDKit CalcBEDROC(alpha=20). EF percentages are 0.1/0.5/1/5/10 with ceil
rounding; precision/recall at k use k=30/50/100/300/500/1000, capped at test size.
Other metrics include ROC-AUC, accuracy, precision, recall, F1, MCC and Kappa.
Top-k ordering uses NumPy argsort defaults; BEDROC uses Python's stable sort.
The summarizer reports arithmetic means and sample SD (ddof=1).

## Verification scope and limitations

Packaging verifies all fixed indices are unique, disjoint and cover the filtered
input. Historical test row IDs, SMILES and labels are checked against input data
and all three ECFP baselines. AUC, AP and BEDROC are recomputed from saved
predictions. Selected epochs are checked against history.csv. Every historical
checkpoint is loaded strictly into the current architecture and evaluated on
two synthetic molecules on CPU, without training or optimizer updates.
Raw-SMILES intersections are reported; this is not a canonical-molecule or
scaffold leakage audit. These checks do not prove equivalence to every other
model in the manuscript or prove the source code never changed historically.

No full training rerun, full checkpoint-to-test prediction replay, fresh
environment install or GPU test was performed. Environment versions were
captured at packaging time, not from a historical lockfile. Numerical differences
across devices, versions and nondeterministic kernels remain possible. Match
the supplied historical results to manuscript tables before claiming reproduction.
Upload the archive and specify its actual location in the reviewer response;
do not claim publication before deposit.
