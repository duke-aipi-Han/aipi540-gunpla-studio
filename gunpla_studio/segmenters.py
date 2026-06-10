from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class SegmentationResult:
    mask: np.ndarray
    method: str
    message: str


class BaseSegmenter:
    name = "base"

    def segment(self, image: Image.Image) -> SegmentationResult:
        raise NotImplementedError

    @staticmethod
    def _empty_mask(image: Image.Image) -> np.ndarray:
        width, height = image.size
        return np.zeros((height, width), dtype=np.float32)


class NaiveBaselineSegmenter(BaseSegmenter):
    name = "Naive baseline"

    def segment(self, image: Image.Image) -> SegmentationResult:
        width, height = image.size
        mask = np.zeros((height, width), dtype=np.float32)

        center_x, center_y = width / 2.0, height / 2.0
        radius_x, radius_y = width * 0.34, height * 0.43
        yy, xx = np.mgrid[:height, :width]
        ellipse = ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2 <= 1.0
        mask[ellipse] = 1.0

        return SegmentationResult(mask, self.name, "Used a center ellipse to create default foreground mask.")


class ClassicalMLSegmenter(BaseSegmenter):
    name = "Classical ML (GrabCut)"

    def segment(self, image: Image.Image) -> SegmentationResult:
        rgb = np.array(image.convert("RGB"))
        height, width = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        inset_x = max(1, int(width * 0.08))
        inset_y = max(1, int(height * 0.05))
        rect = (inset_x, inset_y, width - 2 * inset_x, height - 2 * inset_y)

        grabcut_mask = np.zeros((height, width), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(bgr, grabcut_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where(
                (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
                1.0,
                0.0,
            ).astype(np.float32)
            mask = _postprocess_mask(mask)
            message = "Estimated foreground with GrabCut initialized from an image-border."
        except cv2.error as exc:
            mask = NaiveBaselineSegmenter().segment(image).mask
            message = f"GrabCut failed, fell back to naive baseline: {exc}"

        return SegmentationResult(mask, self.name, message)


class YOLOSegSegmenter(BaseSegmenter):
    name = "Deep learning (YOLO11n-seg)"

    def __init__(self, model_path: str | Path, imgsz: int = 1280) -> None:
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self._model = None
        self._load_error: str | None = None

    def _load_model(self):
        if self._model is not None or self._load_error is not None:
            return self._model
        if not self.model_path.exists():
            self._load_error = f"Model not found at {self.model_path}."
            return None
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        except Exception as exc:  # pragma: no cover - depends on optional runtime stack.
            self._load_error = str(exc)
        return self._model

    def segment(self, image: Image.Image) -> SegmentationResult:
        model = self._load_model()
        if model is None:
            fallback = ClassicalMLSegmenter().segment(image)
            return SegmentationResult(
                fallback.mask,
                self.name,
                f"{self._load_error} Used classical fallback.",
            )

        rgb = np.array(image.convert("RGB"))
        results = model.predict(rgb, imgsz=self.imgsz, conf=0.25, verbose=False)
        height, width = rgb.shape[:2]

        if not results or results[0].masks is None:
            obb_mask = _mask_from_obb_result(results[0], width, height) if results else None
            if obb_mask is not None:
                return SegmentationResult(obb_mask, self.name, "Used YOLO OBB polygon as a coarse mask.")

            fallback = ClassicalMLSegmenter().segment(image)
            return SegmentationResult(fallback.mask, self.name, "No YOLO mask found. Used classical fallback.")

        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes
        scores = boxes.conf.cpu().numpy() if boxes is not None else np.ones(len(masks))
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.float32)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = _postprocess_mask(mask)
        return SegmentationResult(mask, self.name, f"Used YOLO mask with confidence {scores[best_idx]:.2f}.")


def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    return (mask_u8 / 255.0).astype(np.float32)


def _mask_from_obb_result(result, width: int, height: int) -> np.ndarray | None:
    obb = getattr(result, "obb", None)
    if obb is None or len(obb) == 0:
        return None

    try:
        points = obb.xyxyxyxy.cpu().numpy()
        scores = obb.conf.cpu().numpy()
    except AttributeError:
        return None

    best_idx = int(np.argmax(scores))
    polygon = points[best_idx].reshape(-1, 2)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return _postprocess_mask(mask / 255.0)
