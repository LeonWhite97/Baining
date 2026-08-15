from pathlib import Path

import pytest

from tools.vision.fc_bga_yolo.train import build_train_kwargs, load_training_settings


def test_poc_defaults_match_approved_design() -> None:
    settings = load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_poc.yaml"))
    assert settings.model.endswith("yolov8s.pt")
    assert settings.imgsz == 1280
    assert settings.epochs == 100
    assert settings.patience == 20
    assert settings.profile == "fc_bga"


def test_smoke_profile_is_separate_from_formal_training() -> None:
    settings = load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_smoke.yaml"))
    assert settings.profile == "public_smoke"
    assert settings.model.endswith("yolov8n.pt")
    assert settings.epochs == 3


def test_train_kwargs_are_explicit_and_reproducible() -> None:
    settings = load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_poc.yaml"))
    kwargs = build_train_kwargs(settings)
    assert kwargs["imgsz"] == 1280
    assert kwargs["seed"] == 42
    assert kwargs["deterministic"] is True
    assert kwargs["project"] == "tools/vision/fc_bga_yolo/runs"


def test_invalid_confidence_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text(
        "profile: fc_bga\nmodel: yolov8s.pt\ndata: data.yaml\nimgsz: 640\nepochs: 1\n"
        "patience: 1\nbatch: 1\ndevice: cpu\nworkers: 0\nseed: 42\nconf: 1.5\n"
        "project: runs\nname: invalid-conf\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conf"):
        load_training_settings(path)
