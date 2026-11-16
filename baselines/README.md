# Comparison baselines

This directory groups the four comparison models supplied in the two reviewer
archives. The extracted package contents are unchanged, preserving executable
relative paths and the original SHA256 manifests.

## Packages

- [RF_XGBoost_LightGBM](RF_XGBoost_LightGBM/README.md):
  **RF_ECFP**, **XGBoost_ECFP**, and **LightGBM_ECFP** share
  `scripts/run_ml_ecfp_stratified_random.py`. Use `run_baselines.py` to run all
  three models with the supplied settings and fixed partitions.
- [GINE](GINE/README.md):
  **GINE**, a supervised molecular graph baseline. Use `run_baseline.py` to run
  one task/split/seed configuration with `scripts/run_gine_baseline.py`.

Each package contains its own input data, configurations, fixed split indices,
historical reference results, verification reports, pinned requirements,
environment snapshot, and `MANIFEST.sha256`. Keep these files together.
The original ZIP archives remain in the project root.

## Running the models

See the [detailed project README](../README%20%281%29.md#comparison-baselines)
for installation and execution examples, and the package READMEs above for
full model settings. Use Python 3.9 and a separate virtual environment for each
package. Install that package's requirements rather than relying on the
top-level AAP-MVAG requirements.

- Tasks: `GNEtolC`, `HepG2`, `HSkMC`, `IMR90`.
- Splits: `balanced_random`, `balanced_scaffold`.
- Seeds: 42 through 48; the ECFP launcher runs all seven by default, whereas
  the GINE launcher runs one seed per invocation.
- Use `--check-only` for import and input-file checks without training.
- Training output directories must not already exist.
- Both launchers default to CPU. GINE supports `--device cuda`; on Windows,
  `--num-workers 0` avoids data-loader multiprocessing issues.

## Reproducibility notes

The reference results are historical `log-shengao` outputs, not new runs.
No training or fresh dependency installation was performed during integration.
GINE pretrained weights are not included. Preserve input row order and the
documented filtering rules; compare baseline predictions by `row_id` and verify
partition alignment with AAP-MVAG before reporting comparisons. The generic
summaries do not implement the manuscript-specific five-run selection.

Consult each package's verification limitations and confirm dataset
redistribution rights before public release.
