from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from app.adapters.base import NormalizedAttachment
from app.services.image_evidence import EvidenceValidationError, validate_image_set
from tests.image_fixtures import make_attachments, write_image


@pytest.fixture
def case_dir() -> Path:
    path = Path(__file__).parents[3] / "tmp" / "image-evidence-cases" / uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def assert_reason(
    reason_code: str,
    attachments: tuple[NormalizedAttachment, ...],
    root: Path,
) -> None:
    with pytest.raises(EvidenceValidationError) as error:
        validate_image_set(attachments, root)
    assert error.value.reason_code == reason_code


@pytest.mark.parametrize(
    ("image_format", "suffix", "media_type"),
    [("JPEG", ".jpg", "image/jpeg"), ("PNG", ".png", "image/png")],
)
def test_validates_and_orders_complete_light_set(
    case_dir: Path,
    image_format: str,
    suffix: str,
    media_type: str,
) -> None:
    path = case_dir / f"board{suffix}"
    file_hash = write_image(path, image_format=image_format)

    result = validate_image_set(tuple(reversed(make_attachments(path, file_hash))), case_dir)

    assert [item.light_id for item in result] == ["R", "G", "B", "RING"]
    assert result[0].path == path.resolve()
    assert result[0].sha256 == file_hash
    assert (result[0].width, result[0].height) == (64, 48)
    assert result[0].media_type == media_type


@pytest.mark.parametrize(
    ("lights", "expected"),
    [
        (("R", "G", "B"), "LIGHT_SET_INVALID"),
        (("R", "G", "B", "RING", "RING"), "LIGHT_SET_INVALID"),
        (("R", "G", "B", "SIDE"), "LIGHT_SET_INVALID"),
    ],
)
def test_rejects_invalid_light_sets(case_dir: Path, lights: tuple[str, ...], expected: str) -> None:
    path = case_dir / "board.jpg"
    file_hash = write_image(path)
    attachments = tuple(NormalizedAttachment(light, str(path), file_hash) for light in lights)

    assert_reason(expected, attachments, case_dir)


def test_rejects_path_outside_root(case_dir: Path) -> None:
    root = case_dir / "root"
    root.mkdir()
    path = case_dir / "outside.jpg"
    file_hash = write_image(path)

    assert_reason("PATH_OUTSIDE_ROOT", make_attachments(path, file_hash), root)


def test_rejects_missing_and_empty_files(case_dir: Path) -> None:
    missing = case_dir / "missing.jpg"
    assert_reason("FILE_MISSING", make_attachments(missing, "0" * 64), case_dir)

    empty = case_dir / "empty.jpg"
    empty.write_bytes(b"")
    assert_reason("IMAGE_DECODE_FAILED", make_attachments(empty, sha256_hex(empty)), case_dir)


def test_rejects_hash_mismatch(case_dir: Path) -> None:
    path = case_dir / "board.jpg"
    write_image(path)

    assert_reason("HASH_MISMATCH", make_attachments(path, "0" * 64), case_dir)


def test_rejects_truncated_and_unsupported_images(case_dir: Path) -> None:
    truncated = case_dir / "truncated.jpg"
    truncated.write_bytes(b"\xff\xd8\xff\xe0truncated")
    assert_reason("IMAGE_DECODE_FAILED", make_attachments(truncated, sha256_hex(truncated)), case_dir)

    gif = case_dir / "board.gif"
    gif.write_bytes(b"GIF89a" + b"\x00" * 24)
    assert_reason("IMAGE_DECODE_FAILED", make_attachments(gif, sha256_hex(gif)), case_dir)


def test_rejects_dimensions_over_limit(case_dir: Path) -> None:
    path = case_dir / "wide.png"
    file_hash = write_image(path, image_format="PNG", size=(8193, 1))

    assert_reason("IMAGE_DIMENSIONS_INVALID", make_attachments(path, file_hash), case_dir)


def test_rejects_file_over_limit_before_decode(case_dir: Path) -> None:
    path = case_dir / "large.jpg"
    with path.open("wb") as handle:
        handle.truncate(50 * 1024 * 1024 + 1)

    assert_reason("FILE_TOO_LARGE", make_attachments(path, "0" * 64), case_dir)


def test_rejects_symlink_escape_when_supported(case_dir: Path) -> None:
    root = case_dir / "root"
    root.mkdir()
    outside = case_dir / "outside.jpg"
    file_hash = write_image(outside)
    link = root / "linked.jpg"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is not available")

    assert_reason("PATH_OUTSIDE_ROOT", make_attachments(link, file_hash), root)


def sha256_hex(path: Path) -> str:
    from hashlib import sha256

    return sha256(path.read_bytes()).hexdigest()
