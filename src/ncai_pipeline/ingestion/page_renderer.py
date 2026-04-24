from __future__ import annotations

from pathlib import Path

from PIL import Image


def save_page_image(image: Image.Image, path: Path) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_image = image.convert("RGB")
    rgb_image.save(path, format="PNG")
    return rgb_image.width, rgb_image.height

