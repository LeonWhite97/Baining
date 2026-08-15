import json
from pathlib import Path

import pytest

from tools.vision.fc_bga_yolo.download_models import verify_weight
from tools.vision.fc_bga_yolo.download_public_smoke import download_public_smoke


class RecordingDownloader:
    def __init__(self) -> None:
        self.call: tuple[str, str, int, str] | None = None

    def download(
        self,
        *,
        workspace: str,
        project: str,
        version: int,
        format_name: str,
        destination: Path,
        api_key: str,
    ) -> Path:
        self.call = (workspace, project, version, format_name)
        dataset = destination / "downloaded"
        for split in ("train", "valid", "test"):
            (dataset / split / "images").mkdir(parents=True)
            (dataset / split / "labels").mkdir(parents=True)
        (dataset / "data.yaml").write_text("names: [OK, NG]\n", encoding="utf-8")
        return dataset


def test_small_weight_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    path.write_bytes(b"not-a-weight")
    with pytest.raises(ValueError, match="too small"):
        verify_weight(path)


def test_valid_weight_reports_hash_and_size(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    path.write_bytes(b"x" * (1024 * 1024))
    info = verify_weight(path)
    assert info.size_bytes == 1024 * 1024
    assert len(info.sha256) == 64


def test_public_download_requires_api_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ROBOFLOW_API_KEY"):
        download_public_smoke(tmp_path, "", downloader=RecordingDownloader())


def test_public_download_uses_pinned_project_and_writes_manifest(tmp_path: Path) -> None:
    downloader = RecordingDownloader()
    result = download_public_smoke(tmp_path, "secret-value", downloader=downloader)
    assert downloader.call == ("paween", "bga-ram-chips-detection-t3cqn", 1, "yolov8")
    assert (result / "val/images").is_dir()
    assert not (result / "valid").exists()
    manifest = json.loads((result / "source-manifest.json").read_text(encoding="utf-8"))
    assert manifest["purpose"] == "public_smoke"
    assert manifest["license"] == "CC BY 4.0"
    assert "secret-value" not in json.dumps(manifest)
