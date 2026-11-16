# Executable ECFP baselines

This supplement packages RF_ECFP, XGBoost_ECFP and LightGBM_ECFP from
RF-duibi/scripts/run_ml_ecfp_stratified_random.py, copied without modification.
The reference experiment family is `log-shengao`, not older `log-over` runs.
Confirm that this family matches the manuscript tables before submission.

## Contents

- `scripts/`: unchanged original implementation (all three models).
- `run_baselines.py`: portable CPU launcher; missing model libraries are fatal.
- `configs/`: eight task/split configurations and resolved estimator parameters.
- `data/`: four original task CSVs, retaining their row order.
- `splits/`: 56 original fixed train/validation/test index files (4 tasks x 2 splits x 7 seeds).
- `reference_results/`: historical per-run and mean/sample-standard-deviation tables.
- `verification/split_audit.json`: sizes, positive counts, raw-SMILES overlap and
  alignment checks against all three historical prediction files.
- `environment_snapshot.json`, `requirements.txt`, `MANIFEST.sha256`: environment
  provenance, pinned direct dependencies and file integrity hashes.

The data are included for local preparation of the supplement. The authors must
confirm redistribution rights and add original dataset citations before public
release. No new license or redistribution permission is implied by this package.

## Installation and execution

Use Python 3.9 (packaging environment: Python 3.9.20). From the extracted folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_baselines.py --task GNEtolC --split balanced_random --output results/GNEtolC_random --check-only
python run_baselines.py --task GNEtolC --split balanced_random --output results/GNEtolC_random
python run_baselines.py --task GNEtolC --split balanced_scaffold --output results/GNEtolC_scaffold
```

Task choices: GNEtolC, HepG2, HSkMC, IMR90. Split choices: balanced_random,
balanced_scaffold. Each invocation runs all three models and defaults to seeds
42,43,44,45,46,47,48. Use `--seeds 42` for one seed and `--n-jobs 8` to control
CPU threads. Output directories must not already exist. Training is not a quick
installation test; `--check-only` checks imports and file presence without fitting.
All experiments can be executed sequentially:

```bash
for task in GNEtolC HepG2 HSkMC IMR90; do
  for split in balanced_random balanced_scaffold; do
    python run_baselines.py --task "$task" --split "$split" --output "results/${task}_${split}"
  done
done
```

Linux may require the system OpenMP runtime (e.g. libgomp1). The original torch
installation was 2.2.0+cu118; requirements pins its public version 2.2.0 because
only CPU deserialization of trusted bundled split files is needed. Do not load
untrusted `.pt` files. A clean environment install has not been verified.

## Complete experimental configuration

### Inputs and fingerprints

GNEtolC: SMILES column `SMILES`, target `Activity_tolC`. Other tasks: SMILES
column `smiles`, target `label`. Rows with missing SMILES or nonnumeric labels
are removed. Labels are cast to integers, -1 becomes 0, and positive values
become 1. Invalid RDKit molecules are skipped, but fixed-split mode refuses to
proceed if fingerprint generation drops rows. Inputs are not canonicalized or
deduplicated by the training runner; its canonicalization helper is unused.
Split indices address the filtered row sequence, not arbitrary reordered CSVs.

Fingerprints: RDKit Morgan bit vectors, radius 2 (ECFP4), 2048 bits,
`useChirality=True`, with remaining RDKit defaults (bond types included; no
feature invariants). Bits are converted to float32, with no scaling or descriptor
concatenation. The same features and split indices are used by all three models.

### Models

| Model         | Explicit constructor settings                                |
| ------------- | ------------------------------------------------------------ |
| RF_ECFP       | n_estimators=500; max_features=sqrt; min_samples_leaf=1; class_weight=balanced_subsample; random_state=seed; n_jobs=8 in portable launcher |
| XGBoost_ECFP  | n_estimators=800; max_depth=5; learning_rate=0.03; subsample=0.8; colsample_bytree=0.8; objective=binary:logistic; eval_metric=aucpr; tree_method=hist; scale_pos_weight=n_negative/n_positive on training set; random_state=seed; n_jobs=8; device=cpu in portable launcher |
| LightGBM_ECFP | n_estimators=800; learning_rate=0.03; num_leaves=64; subsample=0.8; colsample_bytree=0.8; objective=binary; scale_pos_weight=n_negative/n_positive on training set; random_state=seed; n_jobs=8; verbose=-1; device_type=cpu in portable launcher |

Additional constructor defaults are exported in `configs/model_parameters.json`
and determined by the pinned library versions. Null/None settings retain the
library's internal defaults. The exported seed=42 and scale_pos_weight=1 are
templates: the runner substitutes each seed and the actual training-set ratio.
In particular, LightGBM retains `subsample_freq=0`, so setting subsample=0.8
does NOT enable row bagging. RF retains unlimited depth and bootstrap sampling.

### Splitting, fitting and evaluation

The supplied fixed indices override split generation. They come from the
corresponding test11 records; nominal proportions are 80/10/10, with actual
sizes and class counts in the audit. The supplied scaffold partitions should
not be replaced by newly generated partitions. The original runner's defaults
are different (90/5/5 and seeds 42-44); use the portable launcher for this family.
Balanced split names do not mean a 1:1 resampled class distribution.

Each estimator is fit only on the training partition using `model.fit(X_train,
y_train)`. Validation features are constructed but not used for hyperparameter
selection, early stopping, threshold optimization or refitting. The script
contains no hyperparameter search; this does not establish how the original
authors selected these fixed settings. No early-stopping callback or eval_set
is supplied. XGBoost's eval_metric setting alone does not enable early stopping.

Predicted positive-class probabilities are evaluated on the fixed test partition.
Classification threshold: 0.5. AUPRC is sklearn average precision (not trapezoidal
PR integration). Other outputs include ROC-AUC, accuracy, precision, recall, F1,
MCC, BEDROC(alpha=20), EF at 0.5% and 1%, and precision at 50/100/500/1000.
Ties use stable descending-score ordering; enrichment top counts use ceil, and
precision at k uses min(k, test size). Summaries use pandas sample SD (ddof=1).
Per-seed probabilities, labels and row IDs are written to test_predictions.csv;
metrics go to metrics_test.yaml. The original runner does not save fitted models.

## Verification and limits

Packaging checks require each fixed partition to be in range, duplicate-free,
mutually disjoint and collectively cover the filtered input. Historical test
row IDs, SMILES and labels are checked against every model/seed. Raw-SMILES
intersection counts are reported; this is not a canonical-structure or scaffold
leakage audit, and does not establish matching splits for other manuscript models.

No training was rerun to build this package. Reference tables are historical,
not new outputs. The environment snapshot was collected at packaging time, not
from a training-time lockfile. Historical device/thread arguments for the
log-shengao family were not recovered; the portable launcher explicitly uses
CPU and 8 threads. CPU/GPU and library differences can change numerical results;
bitwise reproduction of manuscript scores has not been demonstrated. Neither
a full end-to-end training run nor a fresh dependency installation was performed.

Before submission: match the reference tables to the manuscript, confirm data
redistribution rights, and upload the archive to the chosen supplementary-code
location. Do not claim a public release until it has actually been deposited.
