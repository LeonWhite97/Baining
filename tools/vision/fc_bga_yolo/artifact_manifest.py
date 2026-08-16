from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlparse


MIN_ARTIFACT_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    name: str
    source_url: str
    license_url: str
    retrieved_at: str
    size_bytes: int
    sha256: str


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("ARTIFACT_URL_INVALID")


def _validate_record(record: ArtifactRecord) -> ArtifactRecord:
    if Path(record.name).name != record.name or not record.name:
        raise ValueError("ARTIFACT_NAME_INVALID")
    _validate_url(record.source_url)
    _validate_url(record.license_url)
    try:
        retrieved = datetime.fromisoformat(record.retrieved_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("ARTIFACT_RETRIEVED_AT_INVALID") from exc
    if not record.retrieved_at.endswith("Z") or retrieved.utcoffset() is None:
        raise ValueError("ARTIFACT_RETRIEVED_AT_INVALID")
    if record.size_bytes < MIN_ARTIFACT_BYTES:
        raise ValueError("ARTIFACT_TOO_SMALL")
    if len(record.sha256) != 64 or any(char not in "0123456789abcdef" for char in record.sha256):
        raise ValueError("ARTIFACT_SHA256_INVALID")
    return record


def capture_artifact_record(
    path: Path,
    *,
    source_url: str,
    license_url: str,
    retrieved_at: str,
) -> ArtifactRecord:
    if not path.is_file():
        raise ValueError("ARTIFACT_UNAVAILABLE")
    return _validate_record(
        ArtifactRecord(
            name=path.name,
            source_url=source_url,
            license_url=license_url,
            retrieved_at=retrieved_at,
            size_bytes=path.stat().st_size,
            sha256=_sha256_file(path),
        )
    )


def load_artifact_records(path: Path) -> tuple[ArtifactRecord, ...]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("ARTIFACT_BASELINE_UNAVAILABLE") from exc
    if not isinstance(document, dict) or set(document) != {"artifacts"}:
        raise ValueError("ARTIFACT_MANIFEST_INVALID")
    items = document["artifacts"]
    if not isinstance(items, list):
        raise ValueError("ARTIFACT_MANIFEST_INVALID")
    expected = {field.name for field in ArtifactRecord.__dataclass_fields__.values()}
    records: list[ArtifactRecord] = []
    for item in items:
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError("ARTIFACT_MANIFEST_INVALID")
        try:
            record = ArtifactRecord(
                name=str(item["name"]),
                source_url=str(item["source_url"]),
                license_url=str(item["license_url"]),
                retrieved_at=str(item["retrieved_at"]),
                size_bytes=int(item["size_bytes"]),
                sha256=str(item["sha256"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("ARTIFACT_MANIFEST_INVALID") from exc
        records.append(_validate_record(record))
    if len({record.name for record in records}) != len(records):
        raise ValueError("ARTIFACT_NAME_DUPLICATE")
    return tuple(records)


def verify_artifact_record(path: Path, record: ArtifactRecord) -> ArtifactRecord:
    _validate_record(record)
    if path.name != record.name or not path.is_file():
        raise ValueError("ARTIFACT_UNAVAILABLE")
    if path.stat().st_size != record.size_bytes or _sha256_file(path) != record.sha256:
        raise ValueError("ARTIFACT_HASH_MISMATCH")
    return record


def write_artifact_records(path: Path, records: tuple[ArtifactRecord, ...]) -> Path:
    validated = tuple(_validate_record(record) for record in records)
    if len({record.name for record in validated}) != len(validated):
        raise ValueError("ARTIFACT_NAME_DUPLICATE")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    document = {"artifacts": [asdict(record) for record in sorted(validated, key=lambda item: item.name)]}
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
