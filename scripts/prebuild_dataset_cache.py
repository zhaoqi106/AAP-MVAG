#!/usr/bin/env python3
"""Prebuild dataset caches without starting training."""
import argparse
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import Dataset


def normalize_config(config):
    dataset_cfg = config.setdefault("dataset", {})
    dataset_cfg["task"] = "classification"
    target_name = dataset_cfg["target"]
    dataset_cfg["primary_task"] = target_name
    return config


def main():
    parser = argparse.ArgumentParser(description="Build Dataset caches once before parallel runs.")
    parser.add_argument("configs", nargs="+", help="Base YAML config(s) whose dataset cache should be built")
    parser.add_argument("--preprocess-num-processes", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    for config_name in args.configs:
        config_path = Path(config_name)
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        normalize_config(config)
        dataset_cfg = config.setdefault("dataset", {})
        dataset_cfg["preprocess_num_processes"] = args.preprocess_num_processes
        dataset_cfg["num_workers"] = args.num_workers

        print(
            "Prebuilding cache | "
            f"config={config_path} task={config.get('task_name')} "
            f"target={dataset_cfg.get('target')} processes={args.preprocess_num_processes}",
            flush=True,
        )
        Dataset(config)
        print(f"Finished cache | config={config_path}", flush=True)


if __name__ == "__main__":
    main()
