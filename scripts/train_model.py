from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def train(data_yaml: Path, epochs: int, imgsz: int, batch: int, device: str | None) -> Path:
    from ultralytics import YOLO

    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data file: {data_yaml}")

    model = YOLO("yolo11n-seg.pt")
    kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": "models/runs",
        "name": "gunpla_yolo11n_seg",
        "exist_ok": True,
    }
    if device:
        kwargs["device"] = device

    results = model.train(**kwargs)
    save_dir = Path(results.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    target_path = Path("models/gunpla_yolo11n_seg.pt")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if best_path.exists():
        shutil.copy2(best_path, target_path)
        print(f"Copied best model to {target_path}")
        return target_path

    raise FileNotFoundError(f"Training finished, but best weights were not found at {best_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11n-seg for single-class Gunpla segmentation.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/gunpla_seg/data.yaml"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    train(args.data, args.epochs, args.imgsz, args.batch, args.device)


if __name__ == "__main__":
    main()
