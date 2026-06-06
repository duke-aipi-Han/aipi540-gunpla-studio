from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def summarize_images(data_dir: Path, output_csv: Path) -> None:
    rows = []
    for image_path in data_dir.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        rows.append(
            {
                "path": str(image_path),
                "split": _infer_split(image_path),
                "width": width,
                "height": height,
                "aspect_ratio": width / height if height else None,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Wrote {len(rows)} image rows to {output_csv}")


def _infer_split(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    for split in ("train", "valid", "val", "test"):
        if split in parts:
            return "valid" if split == "val" else split
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lightweight image metadata features.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/gunpla_seg"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/image_summary.csv"))
    args = parser.parse_args()
    summarize_images(args.data_dir, args.output_csv)


if __name__ == "__main__":
    main()
