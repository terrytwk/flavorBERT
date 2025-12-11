#!/usr/bin/env python3
"""
Batch runner to evaluate all ChemBERTa runs using fart/models/fart_evaluate.py.

Each run is assumed to live under /home/terrytwk/orcd/pool/chemberta/{RUN_NAME}
and contain a checkpoint directory named "final".

For every run with a "final" folder, this script calls:
python fart/models/fart_evaluate.py --model_checkpoint /home/terrytwk/orcd/pool/chemberta/{RUN_NAME}/final --data_dir fart/dataset/splits --output_dir /home/terrytwk/orcd/pool/fart-metrics --run_name {RUN_NAME} --batch_size 16 --no_augmentation
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_CHEMBERTA_ROOT = Path("/home/terrytwk/orcd/pool/chemberta")
DEFAULT_OUTPUT_ROOT = Path("/home/terrytwk/orcd/pool/fart-metrics")
DEFAULT_DATA_DIR = Path("fart/dataset/splits")


def find_runs(root: Path) -> List[Path]:
    """Return run directories that contain a 'final' subdirectory."""
    if not root.exists():
        print(f"[WARN] ChemBERTa root does not exist: {root}", file=sys.stderr)
        return []
    runs = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "final").is_dir():
            runs.append(child)
    return runs


def build_command(run_dir: Path, data_dir: Path, output_dir: Path, batch_size: int) -> List[str]:
    """Construct the fart_evaluate.py command for a given run."""
    run_name = run_dir.name
    model_checkpoint = run_dir / "final"
    return [
        sys.executable,
        "fart/models/fart_evaluate.py",
        "--model_checkpoint",
        str(model_checkpoint),
        "--data_dir",
        str(data_dir),
        "--output_dir",
        str(output_dir),
        "--run_name",
        run_name,
        "--batch_size",
        str(batch_size),
        "--no_augmentation",
    ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate all ChemBERTa runs on FART.")
    parser.add_argument(
        "--chemberta_root",
        type=Path,
        default=DEFAULT_CHEMBERTA_ROOT,
        help="Root directory containing ChemBERTa runs (each with a 'final' folder).",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to FART data splits (contains fart_train.csv, fart_val.csv, fart_test.csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Base output directory for metrics and plots.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation/training.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing.",
    )
    args = parser.parse_args()

    runs = find_runs(args.chemberta_root)
    if not runs:
        print("[INFO] No runs found with a 'final' checkpoint.", file=sys.stderr)
        return

    print(f"[INFO] Found {len(runs)} runs with 'final' checkpoints under {args.chemberta_root}")

    for run_dir in runs:
        cmd = build_command(run_dir, args.data_dir, args.output_dir, args.batch_size)
        print(f"[RUN] {' '.join(cmd)}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
        if result.returncode != 0:
            print(f"[WARN] Command failed for run {run_dir.name} (exit code {result.returncode})", file=sys.stderr)


if __name__ == "__main__":
    main()
