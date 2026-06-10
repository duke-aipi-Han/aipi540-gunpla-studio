from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def ensure_rgb_image(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


def resize_image_to_max_side(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= max_side:
        return image

    scale = max_side / longest_side
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


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


def make_foreground_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    image = image.convert("RGBA")
    alpha = Image.fromarray((normalize_mask(mask) * 255).astype(np.uint8), mode="L")
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=1.2))
    image.putalpha(alpha)

    bbox = alpha.getbbox()
    if bbox is None:
        return image
    return image.crop(bbox)


def composite_overlay(
    background: Image.Image,
    overlay: Image.Image,
    rotation_degrees: float = 0.0,
    zoom: float = 1.0,
    pan_x_percent: float = 0.0,
    pan_y_percent: float = 0.0,
) -> Image.Image:
    background = background.convert("RGB")
    overlay = overlay.convert("RGBA")

    fitted = _fit_overlay_to_background(overlay, background.size, zoom)
    if rotation_degrees:
        fitted = fitted.rotate(rotation_degrees, resample=Image.Resampling.BICUBIC, expand=True)

    bg_width, bg_height = background.size
    x = int((bg_width - fitted.width) / 2 + (pan_x_percent / 100.0) * bg_width)
    y = int((bg_height - fitted.height) / 2 + (pan_y_percent / 100.0) * bg_height)

    output = background.copy()
    output.paste(fitted, (x, y), fitted)
    return output


def _fit_overlay_to_background(overlay: Image.Image, background_size: tuple[int, int], zoom: float) -> Image.Image:
    bg_width, bg_height = background_size
    max_width = bg_width * 0.72
    max_height = bg_height * 0.82
    fit_ratio = min(max_width / max(overlay.width, 1), max_height / max(overlay.height, 1))
    scale = max(fit_ratio * zoom, 0.05)
    size = (max(1, int(overlay.width * scale)), max(1, int(overlay.height * scale)))
    return overlay.resize(size, Image.Resampling.LANCZOS)
