from __future__ import annotations

from PIL import Image, UnidentifiedImageError

from app.inference.base import InferenceImage


REQUIRED_LIGHTS = ("R", "G", "B", "RING")


def stack_rgb_grayscale(images: tuple[InferenceImage, ...]) -> Image.Image:
    by_light = {item.light_id: item for item in images}
    if len(images) != len(REQUIRED_LIGHTS) or set(by_light) != set(REQUIRED_LIGHTS):
        raise ValueError("LIGHT_SET_INVALID: exactly one R, G, B, and RING image is required")
    dimensions = {(item.width, item.height) for item in images}
    if len(dimensions) != 1:
        raise ValueError("IMAGE_SIZE_MISMATCH: four-light dimensions differ")
    channels: list[Image.Image] = []
    try:
        for light in ("R", "G", "B"):
            item = by_light[light]
            with Image.open(item.path) as image:
                image.load()
                if image.size != (item.width, item.height):
                    raise ValueError("IMAGE_SIZE_MISMATCH: image changed after validation")
                channels.append(image.convert("L"))
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("IMAGE_DECODE_FAILED: validated image is unavailable") from exc
    return Image.merge("RGB", tuple(channels))
