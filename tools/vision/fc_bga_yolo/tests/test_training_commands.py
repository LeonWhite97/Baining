from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from PIL import Image
import pytest

from dataclasses import replace

from tools.vision.fc_bga_yolo.contracts import DEFECT_NAMES, INPUT_CONTRACT, REQUIRED_LIGHTS
from tools.vision.fc_bga_yolo.train import (
    apply_training_overrides,
    build_train_kwargs,
    build_training_metadata,
    check_training_settings,
    load_training_settings,
    run_training,
)


def _formal_config(tmp_path: Path, *, names: tuple[str, ...], populated_splits: tuple[str, ...]) -> Path:
    root = tmp_path / "dataset"
    records: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        if split not in populated_splits:
            continue
        image = root / split / "images" / f"{split}-sample.png"
        label = root / split / "labels" / f"{split}-sample.txt"
        Image.new("RGB", (4, 4), "white").save(image)
        label.write_text("", encoding="utf-8")
        records.append(
            {
                "sample_id": f"{split}-sample",
                "group_id": f"{split}-lot",
                "split": split,
                "input_contract": INPUT_CONTRACT,
                "input_sha256": {light: "a" * 64 for light in REQUIRED_LIGHTS},
                "label_sha256": sha256(label.read_bytes()).hexdigest(),
                "output_image": image.relative_to(root).as_posix(),
                "output_label": label.relative_to(root).as_posix(),
                "output_sha256": sha256(image.read_bytes()).hexdigest(),
            }
        )
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        json.dumps(
            {
                "path": "dataset",
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {index: name for index, name in enumerate(names)},
            }
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_poc_defaults_match_approved_design() -> None:
    settings = load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_poc.yaml"))
    assert settings.model.endswith("weights/pretrained/yolov8s.pt")
    assert settings.imgsz == 1280
    assert settings.epochs == 100
    assert settings.patience == 20
    assert settings.profile == "fc_bga"


def test_smoke_profile_is_separate_from_formal_training() -> None:
    settings = load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_smoke.yaml"))
    assert settings.profile == "public_smoke"
    assert settings.model.endswith("weights/pretrained/yolov8n.pt")
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
        "lr0: 0.01\nproject: runs\nname: invalid-conf\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conf"):
        load_training_settings(path)


def test_training_metadata_hashes_weight_and_data_manifest(tmp_path: Path) -> None:
    data_yaml = _formal_config(
        tmp_path,
        names=DEFECT_NAMES,
        populated_splits=("train", "val", "test"),
    )
    model = tmp_path / "best.pt"
    model.write_bytes(b"trained-model")
    settings = replace(
        load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_poc.yaml")),
        data=str(data_yaml),
        name="fc-bga-test-v1",
    )

    metadata = build_training_metadata(
        settings,
        model,
        result_paths={"test": "runs/test/results.csv"},
        runtime_versions={"python": "test"},
    )

    assert metadata.model_version == "fc-bga-test-v1"
    assert metadata.model_sha256 != metadata.dataset_manifest_sha256
    assert metadata.names[0] == "BALL_BRIDGE"
    assert metadata.input_contract == "rgb_grayscale_stack_v1"


def test_formal_preflight_rejects_yaml_class_order_mismatch(tmp_path: Path) -> None:
    data_yaml = _formal_config(
        tmp_path,
        names=tuple(reversed(DEFECT_NAMES)),
        populated_splits=("train", "val", "test"),
    )
    settings = replace(
        load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_poc.yaml")),
        data=str(data_yaml),
    )

    with pytest.raises(ValueError, match="DATA_CLASS_MISMATCH"):
        check_training_settings(settings)


def test_formal_preflight_requires_nonempty_train_val_and_test(tmp_path: Path) -> None:
    data_yaml = _formal_config(tmp_path, names=DEFECT_NAMES, populated_splits=("train",))
    settings = replace(
        load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_poc.yaml")),
        data=str(data_yaml),
    )

    with pytest.raises(ValueError, match="EMPTY_SPLIT:val"):
        check_training_settings(settings)


def test_formal_preflight_rejects_yaml_split_path_drift(tmp_path: Path) -> None:
    data_yaml = _formal_config(
        tmp_path,
        names=DEFECT_NAMES,
        populated_splits=("train", "val", "test"),
    )
    document = json.loads(data_yaml.read_text(encoding="utf-8"))
    document["train"] = "../unvalidated/images"
    data_yaml.write_text(json.dumps(document), encoding="utf-8")
    settings = replace(
        load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_poc.yaml")),
        data=str(data_yaml),
    )

    with pytest.raises(ValueError, match="DATA_SPLIT_PATH_MISMATCH:train"):
        check_training_settings(settings)


def test_public_smoke_cannot_build_deployable_metadata(tmp_path: Path) -> None:
    data_yaml = tmp_path / "smoke.yaml"
    data_yaml.write_text("names: [OK, NG]\n", encoding="utf-8")
    model = tmp_path / "best.pt"
    model.write_bytes(b"smoke-model")
    settings = replace(
        load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_smoke.yaml")),
        data=str(data_yaml),
    )

    with pytest.raises(ValueError, match="PUBLIC_SMOKE_MODEL_NOT_DEPLOYABLE"):
        build_training_metadata(
            settings,
            model,
            result_paths={"test": "runs/smoke"},
            runtime_versions={"python": "test"},
        )


def test_explicit_training_overrides_replace_configured_values() -> None:
    settings = load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_poc.yaml"))

    overridden = apply_training_overrides(
        settings,
        epochs=5,
        batch=2,
        device="cpu",
        workers=0,
        lr0=0.005,
    )

    kwargs = build_train_kwargs(overridden)
    assert kwargs["epochs"] == 5
    assert kwargs["batch"] == 2
    assert kwargs["device"] == "cpu"
    assert kwargs["workers"] == 0
    assert kwargs["lr0"] == 0.005


def test_training_rejects_best_checkpoint_with_reordered_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_yaml = _formal_config(
        tmp_path,
        names=DEFECT_NAMES,
        populated_splits=("train", "val", "test"),
    )
    settings = replace(
        load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_poc.yaml")),
        data=str(data_yaml),
        project=str(tmp_path / "runs"),
    )
    save_dir = tmp_path / "runs" / "fc-bga-test"

    class InitialModel:
        def train(self, **_: object) -> object:
            best = save_dir / "weights" / "best.pt"
            best.parent.mkdir(parents=True)
            best.write_bytes(b"trained-model")
            return SimpleNamespace(save_dir=save_dir)

    class ReorderedBestModel:
        names = {index: name for index, name in enumerate(reversed(DEFECT_NAMES))}

        def val(self, **_: object) -> object:
            pytest.fail("independent evaluation must not run for a class-mismatched checkpoint")

    calls = 0

    def fake_yolo(_: str) -> object:
        nonlocal calls
        calls += 1
        return InitialModel() if calls == 1 else ReorderedBestModel()

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = fake_yolo
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    with pytest.raises(ValueError, match="MODEL_CLASS_MISMATCH"):
        run_training(settings)
