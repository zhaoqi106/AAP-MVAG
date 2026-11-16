"""Run one checkpoint-derived configuration using portable paths."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["GNEtolC", "HepG2", "HSkMC", "IMR90"])
    parser.add_argument("--split", required=True, choices=["balanced_random", "balanced_scaffold"])
    parser.add_argument("--seed", type=int, default=42, choices=range(42, 49))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parent
    configuration_path = root / "configs" / arguments.task / arguments.split / f"seed{arguments.seed}.json"
    configuration = json.loads(configuration_path.read_text())
    settings = configuration["args"].copy()
    for field in ["data_csv", "external_split"]:
        settings[field] = str(root / settings[field])
        if not Path(settings[field]).is_file():
            raise FileNotFoundError(settings[field])
    settings["device"] = arguments.device
    settings["out_dir"] = str(Path(arguments.output).resolve())
    if arguments.num_workers is not None:
        settings["num_workers"] = arguments.num_workers
    environment = os.environ.copy()
    if arguments.device == "cpu" or arguments.check_only:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    runner = root / "scripts" / "run_gine_baseline.py"
    if arguments.check_only:
        subprocess.run([sys.executable, str(runner), "--help"], env=environment, check=True, stdout=subprocess.DEVNULL)
        print(f"PASS: {arguments.task}/{arguments.split}/seed{arguments.seed}: imports and input files")
        return
    output = Path(settings["out_dir"])
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(runner)]
    for name, value in settings.items():
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    (output / "resolved_config.json").write_text(json.dumps(settings, indent=2) + "\n")
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
