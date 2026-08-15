from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError


def _load_grayscale(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError(f"IMAGE_FORMAT_INVALID: {path.name}")
            image.load()
            return image.convert("L")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"IMAGE_DECODE_FAILED: {path.name}") from exc


def stack_rgb_grayscale(r_path: Path, g_path: Path, b_path: Path) -> Image.Image:
    channels = tuple(_load_grayscale(path) for path in (r_path, g_path, b_path))
    if len({channel.size for channel in channels}) != 1:
        raise ValueError("IMAGE_SIZE_MISMATCH: R/G/B dimensions differ")
    return Image.merge("RGB", channels)
