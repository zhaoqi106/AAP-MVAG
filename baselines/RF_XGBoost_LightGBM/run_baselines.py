"""Portable CPU entry point; training logic remains in the original script."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["GNEtolC", "HepG2", "HSkMC", "IMR90"], required=True)
    parser.add_argument("--split", choices=["balanced_random", "balanced_scaffold"], required=True)
    parser.add_argument("--seeds", default="42,43,44,45,46,47,48")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--output", required=True, help="New output directory; existing paths are refused")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs/dependencies without training")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "scripts"))
    import run_ml_ecfp_stratified_random as baseline

    if not baseline.HAS_XGB or not baseline.HAS_LGBM:
        raise RuntimeError("All three baseline libraries are required; install requirements.txt")
    configuration = json.loads((root / "configs" / f"{arguments.task}_{arguments.split}.json").read_text())
    tasks = configuration["tasks"]
    for task in tasks:
        task["path"] = str(root / task["path"])
        if not Path(task["path"]).is_file():
            raise FileNotFoundError(task["path"])
    seeds = [int(value) for value in arguments.seeds.split(",")]
    fixed_splits = {str(seed): str(root / configuration["fixed_splits"][str(seed)]) for seed in seeds}
    for split_path in fixed_splits.values():
        if not Path(split_path).is_file():
            raise FileNotFoundError(split_path)
    if arguments.check_only:
        print(f"PASS: {arguments.task} {arguments.split}; seeds={seeds}; all three libraries available")
        return
    output = Path(arguments.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    task_file = output / "tasks.resolved.json"
    split_file = output / "fixed_splits.resolved.json"
    task_file.write_text(json.dumps(tasks, indent=2) + "\n")
    split_file.write_text(json.dumps(fixed_splits, indent=2) + "\n")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    command = [
        sys.executable, str(root / "scripts" / "run_ml_ecfp_stratified_random.py"),
        "--tasks_json", str(task_file), "--fixed_split_json", str(split_file),
        "--out_dir", str(output), "--seeds", arguments.seeds,
        "--n_jobs", str(arguments.n_jobs), "--radius", "2", "--n_bits", "2048",
        "--valid_size", "0.1", "--test_size", "0.1", "--split_method", arguments.split,
        "--xgb_device", "cpu", "--lgbm_device", "cpu", "--lgbm_gpu_device_id", "",
    ]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
