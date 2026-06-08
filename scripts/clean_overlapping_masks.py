from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Annotation:
    line: str
    class_id: str
    points: np.ndarray
    area: int
    mask: np.ndarray


def clean_dataset(
    input_dir: Path,
    output_dir: Path,
    splits: tuple[str, ...],
    min_overlap_ratio: float,
    overwrite: bool,
) -> list[dict[str, str | int | float]]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    shutil.copytree(input_dir, output_dir)
    _write_ultralytics_yaml(output_dir)

    report_rows: list[dict[str, str | int | float]] = []
    for split in splits:
        image_dir = output_dir / split / "images"
        label_dir = output_dir / split / "labels"
        if not image_dir.exists() or not label_dir.exists():
            continue

        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = _matching_image_path(image_dir, label_path.stem)
            if image_path is None:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]

            original_lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            annotations = [_parse_annotation(line, width, height) for line in original_lines]
            annotations = [annotation for annotation in annotations if annotation is not None and annotation.area > 0]
            if len(annotations) <= 1:
                continue

            kept_indexes, removed_indexes = _keep_largest_in_overlap_groups(annotations, min_overlap_ratio)
            if not removed_indexes:
                continue

            kept_lines = [annotations[index].line for index in kept_indexes]
            label_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")

            report_rows.append(
                {
                    "split": split,
                    "label": str(label_path.relative_to(output_dir)),
                    "original_masks": len(annotations),
                    "kept_masks": len(kept_indexes),
                    "removed_masks": len(removed_indexes),
                    "min_overlap_ratio": min_overlap_ratio,
                }
            )

    return report_rows


def _parse_annotation(line: str, width: int, height: int) -> Annotation | None:
    parts = line.split()
    if len(parts) < 7 or len(parts) % 2 == 0:
        return None

    coords = np.array([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
    coords[:, 0] = np.clip(coords[:, 0] * width, 0, width - 1)
    coords[:, 1] = np.clip(coords[:, 1] * height, 0, height - 1)
    points = coords.astype(np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    area = int(mask.sum())
    return Annotation(line=line, class_id=parts[0], points=points, area=area, mask=mask)


def _keep_largest_in_overlap_groups(
    annotations: list[Annotation],
    min_overlap_ratio: float,
) -> tuple[list[int], list[int]]:
    parent = list(range(len(annotations)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(annotations)):
        for right in range(left + 1, len(annotations)):
            if annotations[left].class_id != annotations[right].class_id:
                continue
            intersection = int(np.logical_and(annotations[left].mask, annotations[right].mask).sum())
            if intersection == 0:
                continue
            overlap_ratio = intersection / max(1, min(annotations[left].area, annotations[right].area))
            if overlap_ratio >= min_overlap_ratio:
                union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(len(annotations)):
        groups.setdefault(find(index), []).append(index)

    keep = []
    remove = []
    for indexes in groups.values():
        if len(indexes) == 1:
            keep.extend(indexes)
            continue
        largest = max(indexes, key=lambda index: annotations[index].area)
        keep.append(largest)
        remove.extend(index for index in indexes if index != largest)

    return sorted(keep), sorted(remove)


def _matching_image_path(image_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        image_path = image_dir / f"{stem}{suffix}"
        if image_path.exists():
            return image_path
    for image_path in image_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES and image_path.stem == stem:
            return image_path
    return None


def _write_ultralytics_yaml(output_dir: Path) -> None:
    data_yaml = output_dir / "data.yaml"
    content = (
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names:\n"
        "  0: gunpla\n"
    )
    data_yaml.write_text(content, encoding="utf-8")


def write_report(rows: list[dict[str, str | int | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "label", "original_masks", "kept_masks", "removed_masks", "min_overlap_ratio"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a dataset copy with overlapping duplicate masks removed.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/gunpla-yolov7"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/gunpla-yolov7-largest-mask"))
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--min-overlap-ratio", type=float, default=0.05)
    parser.add_argument("--report", type=Path, default=Path("data/outputs/overlapping_mask_cleanup.csv"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = clean_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        splits=tuple(args.splits),
        min_overlap_ratio=args.min_overlap_ratio,
        overwrite=args.overwrite,
    )
    write_report(rows, args.report)
    removed = sum(int(row["removed_masks"]) for row in rows)
    print(f"Wrote cleaned dataset to {args.output_dir}")
    print(f"Removed {removed} overlapping masks across {len(rows)} label files")
    print(f"Wrote cleanup report to {args.report}")


if __name__ == "__main__":
    main()
