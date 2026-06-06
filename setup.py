"""Convenience setup runner for Gunpla Studio.

This is a project execution script (get data, build features, train model)
"""

import argparse
import subprocess
from pathlib import Path

PROJECT_DIRS = [
    "models",
    "data/raw",
    "data/processed",
    "data/outputs",
    "notebooks",
]


def ensure_dirs() -> None:
    for directory in PROJECT_DIRS:
        Path(directory).mkdir(parents=True, exist_ok=True)


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up Gunpla Studio.")
    parser.add_argument("--zip-path", type=Path, help="Optional dataset ZIP to extract.")
    parser.add_argument("--train", action="store_true", help="Run training after setup.")
    parser.add_argument("--data", default="data/processed/gunpla_seg/data.yaml", help="YOLO data.yaml path.")
    args = parser.parse_args()

    ensure_dirs()

    if args.zip_path:
        run(["python", "scripts/make_dataset.py", "--zip-path", str(args.zip_path)])
        run(["python", "scripts/build_features.py", "--data-dir", "data/processed/gunpla_seg"])

    if args.train:
        run(["python", "scripts/train_model.py", "--data", args.data])

    print("Gunpla Studio setup complete.")


if __name__ == "__main__":
    main()
