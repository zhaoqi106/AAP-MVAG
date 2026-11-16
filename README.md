# AAP-MVAG

**A Multi-View Adaptive Gating Framework for Antibacterial Activity Prediction and Cytotoxicity Assessment**

Official PyTorch implementation of AAP-MVAG. The framework integrates molecular graphs, 2D molecular descriptors, and multiple molecular fingerprints through local fingerprint gating, joint representation learning, and global adaptive fusion.

The same architecture is trained **independently** for four binary classification tasks: antibacterial activity against *E. coli* ΔtolC (GNEtolC), and cytotoxicity-related endpoints in HepG2, HSkMC, and IMR-90 cells. The antibacterial ranking and cytotoxicity predictions are not jointly optimized.

## Repository structure

```text
AAP-MVAG/
|-- baselines/               # Four comparison models and reproduction materials
|   |-- README.md            # Baseline package guide
|   |-- RF_XGBoost_LightGBM/  # RF_ECFP, XGBoost_ECFP, LightGBM_ECFP
|   `-- GINE/                # Supervised GINE baseline
|-- configs/                 # Task-specific random/scaffold YAML configurations
|-- data/12/                 # Input CSV files
|-- dataset/                 # Data handling and molecular preprocessing
|-- models/                  # AAP-MVAG model components
|-- scripts/                 # Data-cache preparation and result collection
|-- main.py                  # Single-task training and evaluation entry point
|-- run_all.sh               # Batch runner for all task/split configurations
|-- utils.py                 # Shared utilities
|-- requirements.txt         # AAP-MVAG Python dependencies
|-- README.md                # Original brief README
`-- README (1).md            # This detailed project and baseline guide
```

Training creates an `experiment_records/` directory with run-specific configurations, metrics, and saved predictions.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/zhaoqi106/AAP-MVAG.git
cd AAP-MVAG
pip install -r requirements.txt
```

The dependency list includes PyTorch, PyTorch Geometric, RDKit, scikit-learn, and related packages. A CUDA-capable GPU is recommended for training. Install binary dependencies compatible with the local Python, PyTorch, and CUDA environment if necessary.

## Data

Place the input datasets under `data/12/`, preserving their original CSV row order and column names:

```text
data/12/
|-- GNEtolC.csv
|-- datasets - HepG2 cytotoxicity.csv
|-- datasets - HSkMC cytotoxicity.csv
|-- datasets - IMR-90 viability.csv
|-- 345_compounds.csv
```

The GNEtolC antibacterial activity data are derived from Scalia et al.; the three human-cell-line datasets are derived from Wong et al. The 345-compound experimental evaluation set is derived from the Scalia et al. screening study and is separate from model training. Consult the original data providers for dataset provenance and reuse terms.

The input CSV paths, SMILES columns, label columns, and preprocessing parameters are specified in the YAML files under `configs/`. RDKit is used for molecular parsing and featurization. The pipeline retains available stereochemistry and multicomponent structures without an additional salt-removal step. Descriptor scaling is fitted on training data and then applied to validation and test data.

For consistent comparisons, do not reorder the source CSV rows after constructing data splits. The baseline packages index the zero-based row sequence after their documented missing-value and label filtering. Preserve both the source row order and the preprocessing rules when using those indices.

## Experimental settings

AAP-MVAG is evaluated on four tasks under two splitting strategies:

- **Random:** stratified random splitting.
- **Scaffold:** stratified Bemis-Murcko scaffold splitting.

The benchmark uses random seeds **42-48** (seven runs) and train/validation/test partitions of approximately **80%/10%/10%**. When comparing different methods, the same pre-established sample indices must be used for the corresponding task, split, and seed.

The YAML files in `configs/` define the model architecture, fingerprint settings, optimization, loss function, checkpoint selection, and data-processing parameters. Individual YAML files may have `training.num_runs: 3` as a default; use the command-line override below to run the seven-seed protocol.

## Training

Train AAP-MVAG on one task and one splitting strategy:

```bash
python main.py --config configs/GNEtolC_balanced_scaffold.yaml --num-runs 7
```

For a quick single-run execution:

```bash
python main.py --config configs/GNEtolC_balanced_scaffold.yaml --num-runs 1
```

The `--num-runs 7` option runs consecutive split/model seeds starting at 42, i.e., seeds 42, 43, 44, 45, 46, 47, and 48, for the example configuration. The best checkpoint is selected using validation AUPRC.

To run all available task/split configurations on GPU 0:

```bash
PYTHON_BIN="$(command -v python)" NUM_RUNS=7 NUM_WORKERS=4 PREPROCESS_NUM_PROCESSES=4 bash run_all.sh 0
```

`run_all.sh` prepares dataset caches, iterates over `configs/*_balanced_random.yaml` and `configs/*_balanced_scaffold.yaml`, records logs, and collects run results. The GPU argument may be changed to a comma-separated list of device IDs (for example, `0,1`). Set `PYTHON_BIN` explicitly because the batch script otherwise uses a machine-specific default Python path.

## Comparison baselines

Four comparison models are included under `baselines/`:

- **RF_ECFP:** random forest using ECFP4 fingerprints.
- **XGBoost_ECFP:** XGBoost using the same ECFP4 fingerprints.
- **LightGBM_ECFP:** LightGBM using the same ECFP4 fingerprints.
- **GINE:** a supervised molecular graph neural network with edge features; it is not a contrastive-learning method.

The three ECFP models share one implementation in `baselines/RF_XGBoost_LightGBM/`; GINE is in `baselines/GINE/`. Only the outer package directories have been renamed; their contents are preserved without modification, including their own README, launchers, original scripts, configurations, input data, fixed splits, historical reference results, verification reports, dependency specifications, and SHA256 manifests. Keeping the packages self-contained preserves their relative paths; they use their bundled `data/` directories rather than the top-level `data/12/`.

See the [baseline guide](baselines/README.md), the [ECFP package documentation](baselines/RF_XGBoost_LightGBM/README.md), and the [GINE package documentation](baselines/GINE/README.md) for detailed settings and limitations.

### Installation

The packaged baseline environment uses **Python 3.9**. Use a separate virtual environment for each package and install its own `requirements.txt`; the top-level requirements are for AAP-MVAG and do not replace these pinned baseline dependencies. For GINE GPU training, install compatible PyTorch/PyG and CUDA builds as described in its README.

### RF, XGBoost and LightGBM

Starting from the project root, enter the ECFP package and run:

```bash
cd baselines/RF_XGBoost_LightGBM
python -m pip install -r requirements.txt
python run_baselines.py --task GNEtolC --split balanced_random --output results/GNEtolC_random --check-only
python run_baselines.py --task GNEtolC --split balanced_random --output results/GNEtolC_random
```

The training command runs **all three ECFP models**, using seeds **42-48** by default. Add `--seeds 42` for one seed, or `--n-jobs 8` to set CPU threads. The portable launcher uses CPU. Output directories must not already exist. `--check-only` checks imports and required file presence without fitting models.

### GINE

Starting again from the project root, enter the GINE package and run:

```bash
cd baselines/GINE
python -m pip install -r requirements.txt
python run_baseline.py --task GNEtolC --split balanced_random --seed 42 --output results/GNEtolC/balanced_random/seed42 --num-workers 0 --check-only
python run_baseline.py --task GNEtolC --split balanced_random --seed 42 --output results/GNEtolC/balanced_random/seed42 --num-workers 0
```

Each invocation runs **one task, split and seed**. Repeat with seeds **42-48** for the seven-seed protocol. The launcher defaults to CPU; add `--device cuda` when a compatible GPU environment is available. `--num-workers 0` avoids data-loader multiprocessing issues on Windows. Output directories must not already exist. Historical trained weights are not bundled; training produces new `best_model.pt` files.

Summarize newly generated GINE results from within its package directory:

```bash
python scripts/summarize_gine_results.py --root results --out results/gine_grouped_metrics.csv
```

### Tasks, splits and reference results

Both launchers accept tasks `GNEtolC`, `HepG2`, `HSkMC`, and `IMR90`, and split names `balanced_random` and `balanced_scaffold`. Each package includes **56 fixed partitions** (4 tasks x 2 split strategies x 7 seeds). Use the supplied launchers and partitions rather than regenerating splits. GINE and the ECFP packages document matched inputs and partitions; align their test predictions by `row_id`, not CSV output position. Before comparing with an AAP-MVAG run, also verify that its actual sample partitions match; matching seed numbers alone is not sufficient.

The `reference_results/` folders contain **historical `log-shengao` results**, not results generated during this integration. Confirm that these correspond to the manuscript tables. The bundled scripts do not automatically apply the manuscript-specific five-run selection described below. No model training or fresh dependency installation was performed when adding these packages to this project. Original data redistribution terms still apply.

## Results and reproducibility

Each training run is saved under `experiment_records/`. Run directories contain the configuration and execution metadata, the selected checkpoint information, evaluation metrics, and test-set predictions.

Collect available run-level results and a grouped summary:

```bash
python scripts/collect_latest_results.py --root experiment_records --out results_summary.csv
```

The script writes:

- `results_summary.csv`: collected run-level results.
- `results_summary_grouped.csv`: grouped summary statistics for the available runs.

**Manuscript aggregation:** The revised benchmark executes seven seeds, but reports mean and standard deviation from five matched runs per dataset, splitting strategy, and evaluation metric. The runs with the highest and lowest AAP-MVAG test scores are excluded, and the same five seed IDs are used for the corresponding competing-method comparison. The generic collection script aggregates available runs and does not, by itself, implement this manuscript-specific selection or regenerate every comparison table.

The manuscript and Supporting Information provide the baseline configurations and the definitions of the reported AUPRC, AUROC, BEDROC, EF, and precision-at-k metrics. The four baseline implementations and their reproduction materials are included under `baselines/` as described above. Comparisons still require predictions evaluated on identical partitions and consistent metric definitions; the top-level result collector is for AAP-MVAG records, not a unified cross-model aggregator.

## External experimental evaluation

The retrospective experimental evaluation uses 345 compounds (82 active and 263 inactive) selected for testing from GNEprop-identified virtual hits. These compounds are not used for AAP-MVAG training or model selection.

For each splitting strategy, predictions from the five models trained with seeds **42-46** are averaged per compound to obtain ensemble scores. Ensemble AUPRC is computed from these averaged scores. Its 95% confidence interval is estimated using **5,000 compound-level bootstrap resamples with replacement**, with bootstrap seed **20260727**. For comparisons requiring matched GNEprop predictions, the bootstrap subset contains **344 compounds**, excluding one inactive compound without a GNEprop score. The other external metrics are summarized as mean and sample standard deviation across the five model runs evaluated on all 345 compounds.

The external-set analysis uses saved model predictions and does not require retraining for bootstrap resampling. The core `main.py` training workflow and the cross-model external-evaluation analysis are separate procedures.

## License and data attribution

The project code is intended to be distributed under the **MIT License**. Original dataset licenses and usage conditions remain with their respective providers; the code license does not supersede those terms.

For questions or issues, please use the repository's GitHub Issues page.
