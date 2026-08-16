import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.vision.fc_bga_yolo.download_models import prepare_models, verify_weight
from tools.vision.fc_bga_yolo.download_public_smoke import (
    RoboflowDownloader,
    download_public_smoke,
)


class RecordingDownloader:
    def __init__(self, *, include_test: bool = True) -> None:
        self.call: tuple[str, str, int, str] | None = None
        self.include_test = include_test

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
        splits = ("train", "valid", "test") if self.include_test else ("train", "valid")
        for split in splits:
            (dataset / split / "images").mkdir(parents=True)
            (dataset / split / "labels").mkdir(parents=True)
            (dataset / split / "images" / f"{split}.jpg").write_bytes(b"image")
            (dataset / split / "labels" / f"{split}.txt").write_text("", encoding="utf-8")
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


def test_public_download_derives_test_from_valid_when_export_has_no_test_split(
    tmp_path: Path,
) -> None:
    result = download_public_smoke(
        tmp_path,
        "secret-value",
        downloader=RecordingDownloader(include_test=False),
    )

    assert (result / "val/images/valid.jpg").is_file()
    assert (result / "test/images/valid.jpg").is_file()
    assert not (result / "valid").exists()
    manifest = json.loads((result / "source-manifest.json").read_text(encoding="utf-8"))
    assert manifest["test_derived_from"] == "valid"


def test_roboflow_downloader_does_not_precreate_sdk_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    class FakeVersion:
        def download(self, format_name: str, *, location: str, overwrite: bool = False) -> object:
            target = Path(location)
            calls["download"] = (format_name, target, overwrite, target.exists())
            target.mkdir(parents=True)
            dataset = target / "downloaded"
            for split in ("train", "valid", "test"):
                (dataset / split / "images").mkdir(parents=True)
                (dataset / split / "labels").mkdir(parents=True)
            return SimpleNamespace(location=str(dataset))

    class FakeProject:
        def version(self, version: int) -> FakeVersion:
            calls["version"] = version
            return FakeVersion()

    class FakeWorkspace:
        def project(self, project: str) -> FakeProject:
            calls["project"] = project
            return FakeProject()

    class FakeRoboflow:
        def __init__(self, *, api_key: str) -> None:
            calls["api_key"] = api_key

        def workspace(self, workspace: str) -> FakeWorkspace:
            calls["workspace"] = workspace
            return FakeWorkspace()

    monkeypatch.setitem(sys.modules, "roboflow", SimpleNamespace(Roboflow=FakeRoboflow))

    result = RoboflowDownloader().download(
        workspace="paween",
        project="bga-ram-chips-detection-t3cqn",
        version=1,
        format_name="yolov8",
        destination=tmp_path / "dataset",
        api_key="secret-value",
    )

    assert result == tmp_path / "dataset/downloaded"
    assert calls["download"] == ("yolov8", tmp_path / "dataset", True, False)


def test_prepare_models_skips_valid_existing_weight_unless_forced(tmp_path: Path) -> None:
    destination = tmp_path / "pretrained"
    destination.mkdir()
    (destination / "yolov8n.pt").write_bytes(b"n" * (1024 * 1024))
    calls: list[tuple[str, bool]] = []

    def downloader(model_name: str, output: Path, *, force: bool) -> object:
        calls.append((model_name, force))
        target = output / model_name
        target.write_bytes(model_name.encode("ascii") * (1024 * 1024))
        return verify_weight(target)

    infos = prepare_models(
        ("yolov8n.pt", "yolov8s.pt"),
        destination,
        force=False,
        downloader=downloader,
    )

    assert [info.path.name for info in infos] == ["yolov8n.pt", "yolov8s.pt"]
    assert calls == [("yolov8s.pt", False)]
