from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import warnings

from PIL import Image, UnidentifiedImageError

from app.adapters.base import NormalizedAttachment


REQUIRED_LIGHTS = ("R", "G", "B", "RING")
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_IMAGE_SIDE = 8192
MAX_IMAGE_PIXELS = 40_000_000
MEDIA_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png"}


class EvidenceValidationError(ValueError):
    def __init__(self, reason_code: str, public_reason: str) -> None:
        super().__init__(public_reason)
        self.reason_code = reason_code
        self.public_reason = public_reason


@dataclass(frozen=True, slots=True)
class ValidatedImage:
    light_id: str
    path: Path
    sha256: str
    width: int
    height: int
    media_type: str


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_image(path: Path) -> tuple[int, int, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image_format = image.format
                width, height = image.size
                if (
                    width <= 0
                    or height <= 0
                    or width > MAX_IMAGE_SIDE
                    or height > MAX_IMAGE_SIDE
                    or width * height > MAX_IMAGE_PIXELS
                ):
                    raise EvidenceValidationError(
                        "IMAGE_DIMENSIONS_INVALID", "Image dimensions exceed the accepted limits"
                    )
                if image_format not in MEDIA_TYPES:
                    raise EvidenceValidationError("IMAGE_DECODE_FAILED", "Image format must be JPEG or PNG")
                image.verify()
            with Image.open(path) as decoded:
                decoded.load()
    except EvidenceValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise EvidenceValidationError(
            "IMAGE_DIMENSIONS_INVALID", "Image dimensions exceed the accepted limits"
        ) from None
    except (OSError, UnidentifiedImageError, ValueError):
        raise EvidenceValidationError("IMAGE_DECODE_FAILED", "Image could not be decoded") from None
    return width, height, MEDIA_TYPES[image_format]


def validate_image_set(
    attachments: tuple[NormalizedAttachment, ...],
    image_root: Path,
) -> tuple[ValidatedImage, ...]:
    lights = [item.light_id for item in attachments]
    if len(lights) != len(REQUIRED_LIGHTS) or set(lights) != set(REQUIRED_LIGHTS):
        raise EvidenceValidationError(
            "LIGHT_SET_INVALID", "Exactly one R, G, B, and RING image is required"
        )

    root = image_root.resolve(strict=False)
    cached: dict[tuple[Path, str], tuple[int, int, str]] = {}
    validated: list[ValidatedImage] = []
    for attachment in attachments:
        path = Path(attachment.file_path).resolve(strict=False)
        if not path.is_relative_to(root):
            raise EvidenceValidationError("PATH_OUTSIDE_ROOT", "Image path is outside the configured root")
        if not path.is_file():
            raise EvidenceValidationError("FILE_MISSING", "Image file is missing")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise EvidenceValidationError("FILE_TOO_LARGE", "Image file exceeds 50 MiB")

        expected_hash = attachment.file_hash.lower()
        actual_hash = _hash_file(path)
        if actual_hash != expected_hash:
            raise EvidenceValidationError("HASH_MISMATCH", "Image SHA-256 does not match")

        cache_key = (path, actual_hash)
        if cache_key not in cached:
            cached[cache_key] = _decode_image(path)
        width, height, media_type = cached[cache_key]
        validated.append(
            ValidatedImage(
                light_id=attachment.light_id,
                path=path,
                sha256=actual_hash,
                width=width,
                height=height,
                media_type=media_type,
            )
        )

    order = {light: index for index, light in enumerate(REQUIRED_LIGHTS)}
    return tuple(sorted(validated, key=lambda item: order[item.light_id]))
