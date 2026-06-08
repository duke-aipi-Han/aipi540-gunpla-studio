from __future__ import annotations

from pathlib import Path

from PIL import Image


BACKGROUND_DIR = Path("backgrounds")
PACKAGE_BACKGROUND_DIR = Path(__file__).parent / "backgrounds"
BACKGROUND_DIRS = (BACKGROUND_DIR, PACKAGE_BACKGROUND_DIR)
BACKGROUND_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FALLBACK_BACKGROUND_COLOR = (236, 238, 241)


def get_background_choices(background_dirs: tuple[Path, ...] = BACKGROUND_DIRS) -> list[str]:
    choices = list_background_files(background_dirs)
    if choices:
        return choices
    return ["Studio gray"]


def list_background_files(background_dirs: tuple[Path, ...] = BACKGROUND_DIRS) -> list[str]:
    choices = []
    seen = set()
    for background_dir in background_dirs:
        if not background_dir.exists():
            continue
        for image_path in sorted(background_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in BACKGROUND_SUFFIXES:
                continue
            name = image_path.stem
            if name in seen:
                continue
            seen.add(name)
            choices.append(name)
    return choices


def load_background(name: str, size: tuple[int, int], background_dirs: tuple[Path, ...] = BACKGROUND_DIRS) -> Image.Image:
    image_path = find_background_file(name, background_dirs)
    if image_path is None:
        return Image.new("RGB", size, FALLBACK_BACKGROUND_COLOR)

    return Image.open(image_path).convert("RGB")


def find_background_file(name: str, background_dirs: tuple[Path, ...] = BACKGROUND_DIRS) -> Path | None:
    for background_dir in background_dirs:
        if not background_dir.exists():
            continue
        for image_path in sorted(background_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in BACKGROUND_SUFFIXES and image_path.stem == name:
                return image_path
    return None
