# AAP-MVAG

Official PyTorch implementation of AAP-MVAG.

## Requirements

```bash
pip install -r requirements.txt
```

## Data

Put the raw CSVs in `data/12/`:

- `GNEtolC.csv`
- `datasets - HepG2 cytotoxicity.csv`
- `datasets - HSkMC cytotoxicity.csv`
- `datasets - IMR-90 viability.csv`

Configs for 4 tasks × 2 splits are in `configs/`.

## Training

```bash
python main.py --config configs/GNEtolC_balanced_scaffold.yaml
```

## Results

Each run is saved under `experiment_records/`. To summarize:

```bash
python scripts/collect_latest_results.py --root experiment_records --out results_summary.csv
```
