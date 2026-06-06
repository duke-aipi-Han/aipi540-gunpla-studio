from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def ensure_rgb_image(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    if mask.max() > 1:
        mask = mask / 255.0
    return np.clip(mask, 0.0, 1.0)


def composite_foreground(image: Image.Image, mask: np.ndarray, background: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    background = background.convert("RGB").resize(image.size, Image.Resampling.LANCZOS)

    alpha = Image.fromarray((normalize_mask(mask) * 255).astype(np.uint8), mode="L")
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=1.2))

    output = background.copy()
    output.paste(image, (0, 0), alpha)
    return output
