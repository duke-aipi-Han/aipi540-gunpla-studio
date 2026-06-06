from __future__ import annotations

from PIL import Image, ImageDraw


DEFAULT_BACKGROUNDS = {
    "Studio gray": ((236, 238, 241), (186, 192, 201)),
    "Hangar blue": ((197, 216, 232), (74, 101, 132)),
    "Sunset display": ((255, 213, 164), (155, 86, 117)),
    "Workbench": ((226, 204, 170), (122, 101, 78)),
}


def make_default_background(name: str, size: tuple[int, int]) -> Image.Image:
    top, bottom = DEFAULT_BACKGROUNDS.get(name, DEFAULT_BACKGROUNDS["Studio gray"])
    width, height = size
    image = Image.new("RGB", size, top)
    draw = ImageDraw.Draw(image)

    for y in range(height):
        ratio = y / max(height - 1, 1)
        color = tuple(int(top[i] * (1 - ratio) + bottom[i] * ratio) for i in range(3))
        draw.line([(0, y), (width, y)], fill=color)

    floor_y = int(height * 0.72)
    draw.rectangle([(0, floor_y), (width, height)], fill=tuple(max(c - 18, 0) for c in bottom))
    draw.line([(0, floor_y), (width, floor_y)], fill=(255, 255, 255), width=max(1, height // 160))
    return image
