from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)


def maybe_copy_existing_assets(output_dir: Path) -> None:
    source_dir = Path("data/bandai")
    target_dir = output_dir / "unlabeled_examples"
    if source_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        for image_path in source_dir.glob("*.*"):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                shutil.copy2(image_path, target_dir / image_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw Gunpla dataset assets.")
    parser.add_argument("--zip-path", type=Path, help="Annotated YOLO segmentation export ZIP.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/gunpla_seg"))
    args = parser.parse_args()

    if args.zip_path:
        extract_zip(args.zip_path, args.output_dir)
        print(f"Extracted {args.zip_path} to {args.output_dir}")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print("No ZIP provided. Created output directory only.")

    maybe_copy_existing_assets(args.output_dir)
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
