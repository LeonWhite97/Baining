from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from PIL import Image
import numpy as np
import pytest
import yaml

from dataclasses import replace

from tools.vision.fc_bga_yolo.contracts import DEFECT_NAMES, INPUT_CONTRACT, REQUIRED_LIGHTS
from tools.vision.fc_bga_yolo.train import (
    TrainingArtifacts,
    apply_training_overrides,
    build_train_kwargs,
    build_training_metadata,
    check_training_settings,
    evaluate_best,
    load_training_settings,
    run_training,
    train_only,
    _materialize_training_data,
    _resolve_dataset_root,
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


def test_smoke_profile_matches_stage_a_design() -> None:
    settings = load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_smoke.yaml"))
    assert settings.profile == "public_smoke"
    assert settings.model.endswith("weights/pretrained/yolov8n.pt")
    assert (settings.imgsz, settings.epochs, settings.patience) == (640, 30, 10)
    assert (settings.batch, settings.device, settings.workers, settings.seed) == (4, "auto", 0, 42)


def test_public_smoke_class_order_matches_pinned_roboflow_source() -> None:
    config = yaml.safe_load(Path("tools/vision/fc_bga_yolo/configs/public_smoke.yaml").read_text())
    assert config["names"] == {0: "NG", 1: "OK"}


def test_public_smoke_path_is_repo_root_relative_for_ultralytics() -> None:
    config = yaml.safe_load(Path("tools/vision/fc_bga_yolo/configs/public_smoke.yaml").read_text())
    assert config["path"] == "data/external/fc_bga_public_smoke/downloads/bga-ram-chips-detection-t3cqn-v1"


def test_runtime_data_yaml_materializes_an_absolute_dataset_root(tmp_path: Path) -> None:
    root = tmp_path / "download"
    for split in ("train", "val", "test"):
        (root / split / "images").mkdir(parents=True)
    source = tmp_path / "public-smoke.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "path": "download",
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {0: "NG", 1: "OK"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    settings = replace(
        load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_smoke.yaml")),
        data=str(source),
        project=str(tmp_path / "runs"),
    )

    runtime_yaml = _materialize_training_data(settings)
    runtime_document = yaml.safe_load(runtime_yaml.read_text(encoding="utf-8"))

    assert Path(runtime_document["path"]).is_absolute()
    assert Path(runtime_document["path"]) == root.resolve()
    assert runtime_document["val"] == "val/images"


def test_formal_dataset_root_accepts_repo_root_relative_path() -> None:
    data_yaml = Path("tools/vision/fc_bga_yolo/configs/fc_bga_defects.template.yaml")
    root = _resolve_dataset_root(data_yaml, "data/vision/fc_bga_defects")
    assert root == (Path.cwd() / "data/vision/fc_bga_defects").resolve()


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


def test_public_external_profiles_match_b0_and_b1_design() -> None:
    b0 = load_training_settings(
        Path("tools/vision/fc_bga_yolo/configs/train_public_external_b0.yaml")
    )
    b1 = load_training_settings(
        Path("tools/vision/fc_bga_yolo/configs/train_public_external_b1.yaml")
    )

    assert b0.profile == b1.profile == "public_external"
    assert (b0.imgsz, b0.epochs, b0.patience, b0.batch, b0.device) == (640, 10, 5, 4, "auto")
    assert (b1.imgsz, b1.epochs, b1.patience, b1.batch, b1.device) == (640, 50, 10, 4, "auto")
    assert b0.workers == b1.workers == 0
    assert b0.seed == b1.seed == 42
    assert (b0.public_stage, b0.dataset_revision) == ("B0", "public-external-v0.1")
    assert (b1.public_stage, b1.dataset_revision) == ("B1", "public-external-v0.2")


def test_public_external_cannot_build_deployable_metadata(tmp_path: Path) -> None:
    settings = replace(
        load_training_settings(
            Path("tools/vision/fc_bga_yolo/configs/train_public_external_b0.yaml")
        ),
        data=str(tmp_path / "data.yaml"),
    )

    with pytest.raises(ValueError, match="PUBLIC_EXTERNAL_MODEL_NOT_DEPLOYABLE"):
        build_training_metadata(
            settings,
            tmp_path / "best.pt",
            result_paths={},
            runtime_versions={},
        )


def _public_revision_yaml(
    tmp_path: Path,
    *,
    stage: str = "B0",
    version: str = "public-external-v0.1",
) -> Path:
    root = tmp_path / version
    manifest_lines = []
    assignment_lines = []
    for split in ("train", "val", "test"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        sample_id = f"{split}-sample"
        image = image_dir / f"{sample_id}.png"
        label = label_dir / f"{sample_id}.txt"
        Image.new("RGB", (256, 256), "white").save(image)
        label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="ascii")
        manifest_lines.append(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "source_group_id": f"{split}-group",
                    "split": split,
                    "image_path": image.relative_to(root).as_posix(),
                }
            )
            + "\n"
        )
        assignment_lines.append(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "source_group_id": f"{split}-group",
                    "split": split,
                }
            )
            + "\n"
        )
    manifest = root / "manifest.jsonl"
    assignments = root / "assignments.jsonl"
    manifest.write_text("".join(manifest_lines), encoding="utf-8")
    assignments.write_text("".join(assignment_lines), encoding="utf-8")
    (root / "revision.json").write_text(
        json.dumps(
            {
                "version": version,
                "stage": stage,
                "published_manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
                "assignments_sha256": sha256(assignments.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": root.resolve().as_posix(),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {index: name for index, name in enumerate(DEFECT_NAMES)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_public_external_preflight_accepts_verified_revision(tmp_path: Path) -> None:
    data_yaml = _public_revision_yaml(tmp_path)
    settings = replace(
        load_training_settings(
            Path("tools/vision/fc_bga_yolo/configs/train_public_external_b0.yaml")
        ),
        data=str(data_yaml),
    )

    check_training_settings(settings)


def test_public_external_preflight_rejects_manifest_hash_drift(tmp_path: Path) -> None:
    data_yaml = _public_revision_yaml(tmp_path)
    (data_yaml.parent / "manifest.jsonl").write_text("tampered\n", encoding="utf-8")
    settings = replace(
        load_training_settings(
            Path("tools/vision/fc_bga_yolo/configs/train_public_external_b0.yaml")
        ),
        data=str(data_yaml),
    )

    with pytest.raises(ValueError, match="REVISION_MANIFEST_HASH_MISMATCH"):
        check_training_settings(settings)


def test_public_external_preflight_rejects_wrong_stage_revision(tmp_path: Path) -> None:
    data_yaml = _public_revision_yaml(
        tmp_path,
        stage="B1",
        version="public-external-v0.2",
    )
    settings = replace(
        load_training_settings(
            Path("tools/vision/fc_bga_yolo/configs/train_public_external_b0.yaml")
        ),
        data=str(data_yaml),
    )

    with pytest.raises(ValueError, match="REVISION_STAGE_MISMATCH"):
        check_training_settings(settings)


def test_public_external_preflight_rejects_assignment_hash_drift(tmp_path: Path) -> None:
    data_yaml = _public_revision_yaml(tmp_path)
    (data_yaml.parent / "assignments.jsonl").write_text("tampered\n", encoding="utf-8")
    settings = replace(
        load_training_settings(
            Path("tools/vision/fc_bga_yolo/configs/train_public_external_b0.yaml")
        ),
        data=str(data_yaml),
    )

    with pytest.raises(ValueError, match="REVISION_ASSIGNMENTS_HASH_MISMATCH"):
        check_training_settings(settings)


def test_public_external_preflight_requires_nonempty_splits(tmp_path: Path) -> None:
    data_yaml = _public_revision_yaml(tmp_path)
    (data_yaml.parent / "test" / "images" / "test-sample.png").unlink()
    settings = replace(
        load_training_settings(
            Path("tools/vision/fc_bga_yolo/configs/train_public_external_b0.yaml")
        ),
        data=str(data_yaml),
    )

    with pytest.raises(ValueError, match="EMPTY_SPLIT:test"):
        check_training_settings(settings)


def test_public_external_preflight_requires_fixed_class_order(tmp_path: Path) -> None:
    data_yaml = _public_revision_yaml(tmp_path)
    document = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    document["names"] = {index: name for index, name in enumerate(reversed(DEFECT_NAMES))}
    data_yaml.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    settings = replace(
        load_training_settings(
            Path("tools/vision/fc_bga_yolo/configs/train_public_external_b0.yaml")
        ),
        data=str(data_yaml),
    )

    with pytest.raises(ValueError, match="DATA_CLASS_MISMATCH"):
        check_training_settings(settings)


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
            (best.parent / "last.pt").write_bytes(b"last-model")
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


def test_train_only_resumes_from_last_checkpoint_with_final_epoch_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeModel:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path

        def train(self, **kwargs: object) -> object:
            save_dir = tmp_path / f"segment-{len(calls)}"
            weights = save_dir / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"best")
            (weights / "last.pt").write_bytes(b"last")
            calls.append((self.model_path, kwargs))
            return SimpleNamespace(save_dir=save_dir)

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeModel
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    settings = replace(
        load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_smoke.yaml")),
        project=str(tmp_path / "runs"),
    )

    calibration = train_only(settings, epochs=3)
    resumed = train_only(settings, epochs=30, resume_from=calibration.last)

    assert calibration.best.is_file() and calibration.last.is_file()
    assert resumed.best.is_file() and resumed.last.is_file()
    assert calls[1][0] == str(calibration.last)
    assert calls[1][1]["resume"] == str(calibration.last)
    assert calls[1][1]["epochs"] == 30


def test_run_training_preserves_best_path_after_independent_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_dir = tmp_path / "run"

    class FakeModel:
        names = {0: "NG", 1: "OK"}

        def __init__(self, model_path: str) -> None:
            self.model_path = model_path

        def train(self, **_: object) -> object:
            weights = save_dir / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"best")
            (weights / "last.pt").write_bytes(b"last")
            return SimpleNamespace(save_dir=save_dir)

        def val(self, **_: object) -> object:
            return SimpleNamespace(save_dir=tmp_path / "test")

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeModel
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    settings = replace(
        load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_smoke.yaml")),
        project=str(tmp_path / "runs"),
    )

    best = run_training(settings)

    assert best == save_dir / "weights" / "best.pt"


def test_public_external_evaluation_writes_non_deployable_observed_class_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_yaml = _public_revision_yaml(tmp_path)
    settings = replace(
        load_training_settings(
            Path("tools/vision/fc_bga_yolo/configs/train_public_external_b0.yaml")
        ),
        data=str(data_yaml),
        project=str(tmp_path / "runs"),
    )
    save_dir = tmp_path / "run"
    weights = save_dir / "weights"
    weights.mkdir(parents=True)
    best = weights / "best.pt"
    last = weights / "last.pt"
    best.write_bytes(b"best")
    last.write_bytes(b"last")

    class FakeMetrics:
        names = {index: name for index, name in enumerate(DEFECT_NAMES)}
        nt_per_class = np.array([1, 0, 0, 0, 0, 0, 0])
        ap_class_index = np.array([0])
        save_dir = tmp_path / "test-results"

        @staticmethod
        def class_result(index: int) -> tuple[float, float, float, float]:
            assert index == 0
            return 0.8, 0.7, 0.6, 0.5

        @staticmethod
        def mean_results() -> tuple[float, float, float, float]:
            return 0.8, 0.7, 0.6, 0.5

    class FakeModel:
        names = {index: name for index, name in enumerate(DEFECT_NAMES)}

        def __init__(self, _: str) -> None:
            self.callback = None

        def add_callback(self, event: str, callback: object) -> None:
            assert event == "on_val_batch_end"
            self.callback = callback

        def val(self, **_: object) -> object:
            assert self.callback is not None
            stats = {
                "tp": [np.ones((1, 10), dtype=bool)],
                "conf": [np.array([0.9])],
                "pred_cls": [np.array([0.0])],
                "target_cls": [np.array([0.0])],
                "target_img": [np.array([0.0])],
            }
            box = SimpleNamespace(image_metrics={"test-sample.png": {}})
            self.callback(SimpleNamespace(metrics=SimpleNamespace(stats=stats, box=box)))
            FakeMetrics.save_dir.mkdir(parents=True)
            return FakeMetrics()

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeModel
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    evaluate_best(settings, TrainingArtifacts(save_dir, best, last))

    report_path = FakeMetrics.save_dir / "public_evaluation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["observed_class_mAP50"] == pytest.approx(0.6)
    assert report["classes"]["MISSING_BALL"]["status"] == "no_evidence"
    assert report["test_images"] == 1
    assert report["test_boxes"] == 1
    assert "bootstrap_95" not in report
    assert not (weights / "model_metadata.json").exists()
