from __future__ import annotations

import argparse
import ast
import sys
from datetime import datetime
import shutil
from pathlib import Path
from typing import Literal


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
Task = Literal["seg"]


def inspect_dataset(data_yaml: Path) -> dict:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data file: {data_yaml}")

    data = _load_data_yaml(data_yaml)

    root = data_yaml.parent
    summary = {
        "root": str(root),
        "names": data.get("names", {0: "gunpla"}),
        "splits": {},
        "label_column_counts": {},
        "out_of_range_values": 0,
        "empty_label_files": 0,
        "class_ids": set(),
        "path_rewrites": {},
    }

    for split_name in ("train", "val", "valid", "test"):
        split_path = data.get(split_name)
        if split_path is None:
            continue
        if split_name == "val" and "valid" in summary["splits"]:
            continue

        image_dir, was_rewritten = _resolve_image_dir(root, str(split_path))
        if was_rewritten:
            summary["path_rewrites"][split_name] = {
                "from": str(split_path),
                "to": str(image_dir),
            }
        label_dir = image_dir.parent / "labels"
        image_files = [p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
        label_files = list(label_dir.glob("*.txt")) if label_dir.exists() else []

        missing_labels = 0
        for image_path in image_files:
            if not (label_dir / f"{image_path.stem}.txt").exists():
                missing_labels += 1

        summary["splits"][split_name] = {
            "images": len(image_files),
            "labels": len(label_files),
            "missing_labels": missing_labels,
            "image_dir": str(image_dir),
            "label_dir": str(label_dir),
        }

        for label_path in label_files:
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                summary["empty_label_files"] += 1
                continue
            for line in lines:
                parts = line.split()
                summary["label_column_counts"][len(parts)] = summary["label_column_counts"].get(len(parts), 0) + 1
                try:
                    summary["class_ids"].add(int(float(parts[0])))
                    values = [float(value) for value in parts[1:]]
                except ValueError:
                    continue
                summary["out_of_range_values"] += sum(value < 0.0 or value > 1.0 for value in values)

    summary["class_ids"] = sorted(summary["class_ids"])
    return summary


def infer_task(summary: dict) -> Task:
    column_counts = set(summary["label_column_counts"])
    if column_counts == {9}:
        raise ValueError(
            "Dataset appears to use YOLO OBB labels, not instance segmentation polygons. "
            "Use a YOLO segmentation export."
        )
    if all(count >= 7 and count % 2 == 1 for count in column_counts):
        return "seg"
    raise ValueError(f"Could not infer YOLO task from label column counts: {summary['label_column_counts']}")


def _load_data_yaml(data_yaml: Path) -> dict:
    try:
        import yaml

        with data_yaml.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except ModuleNotFoundError:
        data: dict[str, str | dict[int, str]] = {}
        names: dict[int, str] = {}
        in_names = False
        for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("names:"):
                in_names = True
                continue
            if in_names and ":" in line:
                key, value = line.split(":", 1)
                if key.strip().isdigit():
                    names[int(key.strip())] = value.strip().strip("'\"")
                    continue
            in_names = False
            if ":" in line:
                key, value = line.split(":", 1)
                parsed_value = value.strip().strip("'\"")
                if key.strip() == "names":
                    try:
                        parsed_value = ast.literal_eval(value.strip())
                    except (SyntaxError, ValueError):
                        pass
                data[key.strip()] = parsed_value
        if names:
            data["names"] = names
        return data


def _resolve_image_dir(root: Path, split_path: str) -> tuple[Path, bool]:
    image_dir = (root / split_path).resolve()
    if image_dir.exists():
        return image_dir, False

    while split_path.startswith("../"):
        split_path = split_path[3:]
    fallback = (root / split_path).resolve()
    if fallback.exists():
        return fallback, True

    return image_dir, False


def print_dataset_summary(summary: dict, task: Task) -> None:
    print("Dataset validation")
    print(f"  root: {summary['root']}")
    print(f"  inferred task: {task}")
    print(f"  class ids: {summary['class_ids']}")
    print(f"  label column counts: {summary['label_column_counts']}")
    print(f"  out-of-range coordinate values: {summary['out_of_range_values']}")
    print(f"  empty label files: {summary['empty_label_files']}")
    if summary["path_rewrites"]:
        print(f"  corrected split paths: {summary['path_rewrites']}")
    for split_name, split in summary["splits"].items():
        print(
            f"  {split_name}: images={split['images']} labels={split['labels']} "
            f"missing_labels={split['missing_labels']}"
        )


def prepare_clipped_dataset(data_yaml: Path, output_dir: Path) -> Path:
    """Copy a YOLO dataset and clip normalized label coordinates into [0, 1]."""
    source_root = data_yaml.parent
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(source_root, output_dir)

    for label_path in output_dir.rglob("labels/*.txt"):
        clipped_lines = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            class_id = parts[0]
            coords = []
            for raw_value in parts[1:]:
                value = float(raw_value)
                coords.append(str(min(1.0, max(0.0, value))))
            clipped_lines.append(" ".join([class_id, *coords]))
        label_path.write_text("\n".join(clipped_lines) + ("\n" if clipped_lines else ""), encoding="utf-8")

    return output_dir / data_yaml.name


def prepare_ultralytics_data_yaml(data_yaml: Path, summary: dict) -> Path:
    output_dir = Path("data/processed") / f"{data_yaml.parent.name}-ultralytics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_yaml = output_dir / "data.yaml"

    names = summary.get("names") or {0: "gunpla"}
    if isinstance(names, list):
        names_block = "\n".join(f"  {index}: {name}" for index, name in enumerate(names))
        nc = len(names)
    elif isinstance(names, dict):
        normalized_names = {int(key): value for key, value in names.items()}
        names_block = "\n".join(f"  {index}: {name}" for index, name in sorted(normalized_names.items()))
        nc = len(normalized_names)
    else:
        names_block = "  0: gunpla"
        nc = 1

    dataset_root = data_yaml.parent.resolve()
    content = (
        f"path: {dataset_root.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        f"nc: {nc}\n"
        "names:\n"
        f"{names_block}\n"
    )
    output_yaml.write_text(content, encoding="utf-8")
    return output_yaml


def train(
    data_yaml: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str | None,
    workers: int,
    cache: bool | str,
    amp: bool,
    mask_ratio: int,
    task: str,
    model_name: str | None,
    run_name: str | None,
    clip_labels: bool,
    augment_mode: str,
) -> Path:
    from ultralytics import YOLO

    summary = inspect_dataset(data_yaml)
    resolved_task = infer_task(summary) if task == "auto" else task
    if resolved_task != "seg":
        raise ValueError("--task must be one of: auto, seg")

    print_dataset_summary(summary, resolved_task)

    train_data_yaml = data_yaml
    if clip_labels and summary["out_of_range_values"] > 0:
        clipped_root = Path("data/processed") / f"{data_yaml.parent.name}-clipped"
        train_data_yaml = prepare_clipped_dataset(data_yaml, clipped_root)
        print(f"Prepared clipped training copy at {clipped_root}")
    elif summary["out_of_range_values"] > 0:
        print("WARNING: labels contain out-of-range normalized coordinates and may be rejected by YOLO.")

    if summary["path_rewrites"]:
        train_data_yaml = prepare_ultralytics_data_yaml(train_data_yaml, summary)
        print(f"Prepared Ultralytics data file with corrected split paths at {train_data_yaml}")

    if model_name is None:
        model_name = "yolo11n-seg.pt"

    model = YOLO(model_name)
    if run_name is None:
        run_name = build_run_name(model_name, resolved_task, imgsz, batch, augment_mode)
    kwargs = {
        "data": str(train_data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": "models/runs",
        "name": run_name,
        "exist_ok": True,
        "patience": 15,
        "cos_lr": True,
        "pretrained": True,
        "single_cls": True,
        "plots": True,
        "workers": workers,
        "cache": cache,
        "amp": amp,
        "mask_ratio": mask_ratio,
    }
    augmentation_args = get_augmentation_args(augment_mode)
    kwargs.update(augmentation_args)
    print(f"Using {augment_mode} data augmentation: {augmentation_args}")
    if device:
        kwargs["device"] = device

    results = model.train(**kwargs)
    save_dir = Path(results.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    target_path = Path(f"models/{run_name}.pt")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if best_path.exists():
        shutil.copy2(best_path, target_path)
        print(f"Copied best model to {target_path}")
        return target_path

    raise FileNotFoundError(f"Training finished, but best weights were not found at {best_path}")


def build_run_name(model_name: str, task: str, imgsz: int, batch: int, augment_mode: str) -> str:
    model_stem = Path(model_name).stem.replace("-", "_")
    task_label = "" if model_stem.endswith(f"_{task}") else f"_{task}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"gunpla_{model_stem}{task_label}_img{imgsz}_b{batch}_{augment_mode}_{timestamp}"


def get_augmentation_args(mode: str) -> dict[str, float | int | str]:
    if mode == "none":
        return {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.0,
            "auto_augment": None,
            "close_mosaic": 0,
        }
    if mode == "light":
        return {
            "hsv_h": 0.01,
            "hsv_s": 0.25,
            "hsv_v": 0.20,
            "degrees": 5.0,
            "translate": 0.06,
            "scale": 0.20,
            "shear": 1.0,
            "perspective": 0.0002,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 0.15,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.05,
            "auto_augment": None,
            "close_mosaic": 10,
        }
    if mode == "strong":
        return {
            "hsv_h": 0.015,
            "hsv_s": 0.45,
            "hsv_v": 0.35,
            "degrees": 12.0,
            "translate": 0.12,
            "scale": 0.55,
            "shear": 4.0,
            "perspective": 0.0008,
            "flipud": 0.05,
            "fliplr": 0.5,
            "mosaic": 0.7,
            "mixup": 0.08,
            "copy_paste": 0.25,
            "erasing": 0.25,
            "auto_augment": "randaugment",
            "close_mosaic": 10,
        }
    raise ValueError("--augment-mode must be one of: none, light, strong")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11n-seg for single-class Gunpla instance segmentation.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/gunpla-yolov7/data.yaml"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=0 if sys.platform.startswith("win") else 4,
        help="Dataloader workers. Use 0 on Windows to avoid CUDA DLL paging-file errors.",
    )
    parser.add_argument("--cache", choices=["none", "ram", "disk"], default="none")
    parser.add_argument("--no-amp", action="store_true", help="Disable Automatic Mixed Precision checks/training.")
    parser.add_argument("--mask-ratio", type=int, default=4, help="YOLO segmentation mask downsampling ratio. Try 2 for finer masks.")
    parser.add_argument("--task", choices=["auto", "seg"], default="auto")
    parser.add_argument("--model", type=str, default=None, help="Optional YOLO segmentation base model, e.g. yolo11n-seg.pt.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional explicit Ultralytics run name.")
    parser.add_argument("--no-clip-labels", action="store_true", help="Do not clip labels with coordinates outside [0, 1].")
    parser.add_argument("--augment-mode", choices=["none", "light", "strong"], default="light")
    parser.add_argument("--no-augment", action="store_true", help="Deprecated alias for --augment-mode none.")
    parser.add_argument("--validate-only", action="store_true", help="Inspect dataset format and exit without training.")
    args = parser.parse_args()
    if args.validate_only:
        summary = inspect_dataset(args.data)
        resolved_task = infer_task(summary) if args.task == "auto" else args.task
        print_dataset_summary(summary, resolved_task)
        return

    train(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=False if args.cache == "none" else args.cache,
        amp=not args.no_amp,
        mask_ratio=args.mask_ratio,
        task=args.task,
        model_name=args.model,
        run_name=args.run_name,
        clip_labels=not args.no_clip_labels,
        augment_mode="none" if args.no_augment else args.augment_mode,
    )


if __name__ == "__main__":
    main()
