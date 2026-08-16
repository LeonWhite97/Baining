from __future__ import annotations

import argparse
from datetime import date
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Protocol


WORKSPACE = "paween"
PROJECT = "bga-ram-chips-detection-t3cqn"
VERSION = 1
FORMAT_NAME = "yolov8"
SOURCE_URL = "https://universe.roboflow.com/paween/bga-ram-chips-detection-t3cqn"


class PublicDatasetDownloader(Protocol):
    def download(
        self,
        *,
        workspace: str,
        project: str,
        version: int,
        format_name: str,
        destination: Path,
        api_key: str,
    ) -> Path: ...


class RoboflowDownloader:
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
        try:
            from roboflow import Roboflow
        except ImportError as exc:
            raise RuntimeError("install requirements-train.txt before downloading public data") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        dataset = (
            Roboflow(api_key=api_key)
            .workspace(workspace)
            .project(project)
            .version(version)
            .download(format_name, location=str(destination), overwrite=True)
        )
        return Path(dataset.location)


def _file_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        hashes[path.relative_to(root).as_posix()] = sha256(path.read_bytes()).hexdigest()
    return hashes


def download_public_smoke(
    destination: Path,
    api_key: str,
    *,
    downloader: PublicDatasetDownloader | None = None,
) -> Path:
    if not api_key.strip():
        raise ValueError("ROBOFLOW_API_KEY is required")
    selected = downloader or RoboflowDownloader()
    root = selected.download(
        workspace=WORKSPACE,
        project=PROJECT,
        version=VERSION,
        format_name=FORMAT_NAME,
        destination=destination,
        api_key=api_key,
    ).resolve()
    for split in ("train", "valid"):
        if not (root / split / "images").is_dir() or not (root / split / "labels").is_dir():
            raise ValueError(f"PUBLIC_DATASET_INVALID: missing {split} split")
    valid = root / "valid"
    val = root / "val"
    if val.exists():
        raise ValueError("PUBLIC_DATASET_INVALID: both valid and val exist")
    valid.rename(val)
    test_derived_from = None
    if not (root / "test" / "images").is_dir() or not (root / "test" / "labels").is_dir():
        shutil.copytree(val, root / "test")
        test_derived_from = "valid"
    manifest = {
        "accessed_on": date.today().isoformat(),
        "format": FORMAT_NAME,
        "license": "CC BY 4.0",
        "project": PROJECT,
        "purpose": "public_smoke",
        "source_url": SOURCE_URL,
        "version": VERSION,
        "workspace": WORKSPACE,
        "files": _file_hashes(root),
    }
    if test_derived_from is not None:
        manifest["test_derived_from"] = test_derived_from
    (root / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the pinned CC BY 4.0 public smoke dataset.")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/external/fc_bga_public_smoke/downloads") / f"{PROJECT}-v{VERSION}",
    )
    args = parser.parse_args()
    root = download_public_smoke(args.destination, os.getenv("ROBOFLOW_API_KEY", ""))
    print(json.dumps({"dataset": str(root), "purpose": "public_smoke"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
