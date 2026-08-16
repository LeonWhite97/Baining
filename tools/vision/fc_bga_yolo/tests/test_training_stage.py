from dataclasses import replace
from pathlib import Path
from typing import Callable

import pytest

from tools.vision.fc_bga_yolo import run_training_stage as stage_module
from tools.vision.fc_bga_yolo.artifact_manifest import (
    capture_artifact_record,
    write_artifact_records,
)
from tools.vision.fc_bga_yolo.train import (
    TrainingArtifacts,
    TrainingSettings,
    load_training_settings,
)
from tools.vision.fc_bga_yolo.training_runtime import HardwareProbe


CPU_PROBE = HardwareProbe("2.13.0+cpu", False, 0, None, None, "test-cpu")


def _settings_with_baseline(tmp_path: Path, *, epochs: int = 30) -> TrainingSettings:
    weight = tmp_path / "pretrained" / "yolov8n.pt"
    weight.parent.mkdir()
    weight.write_bytes(b"x" * (1024 * 1024))
    record = capture_artifact_record(
        weight,
        source_url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
        license_url="https://www.ultralytics.com/license",
        retrieved_at="2026-08-16T12:00:00Z",
    )
    write_artifact_records(weight.parent / "artifact-manifest.json", (record,))
    return replace(
        load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_smoke.yaml")),
        model=str(weight),
        project=str(tmp_path / "runs"),
        epochs=epochs,
        device="auto",
    )


def _recording_train_only(
    tmp_path: Path,
    epochs_seen: list[int],
    resumes_seen: list[Path | None],
) -> Callable[..., TrainingArtifacts]:
    def fake_train_only(
        settings: TrainingSettings,
        *,
        epochs: int | None = None,
        resume_from: Path | None = None,
    ) -> TrainingArtifacts:
        assert epochs is not None
        epochs_seen.append(epochs)
        resumes_seen.append(resume_from)
        save_dir = tmp_path / f"segment-{len(epochs_seen)}"
        weights = save_dir / "weights"
        weights.mkdir(parents=True)
        best = weights / "best.pt"
        last = weights / "last.pt"
        best.write_bytes(b"best")
        last.write_bytes(b"last")
        return TrainingArtifacts(save_dir, best, last)

    return fake_train_only


def _clock(values: tuple[float, ...]) -> Callable[[], float]:
    iterator = iter(values)
    return lambda: next(iterator)


def test_stage_a_calibrates_then_continues_for_remaining_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    epochs_seen: list[int] = []
    resumes_seen: list[Path | None] = []
    monkeypatch.setattr(stage_module, "probe_hardware", lambda: CPU_PROBE)
    monkeypatch.setattr(
        stage_module,
        "train_only",
        _recording_train_only(tmp_path, epochs_seen, resumes_seen),
    )
    monkeypatch.setattr(stage_module, "evaluate_best", lambda settings, artifacts: object())
    monkeypatch.setattr(stage_module.time, "perf_counter", _clock((0.0, 30.0)))

    report = stage_module.run_calibrated_stage(
        _settings_with_baseline(tmp_path),
        "A",
        tmp_path / "stage-a.json",
    )

    assert epochs_seen == [3, 27]
    assert resumes_seen[0] is None
    assert resumes_seen[1] is not None
    assert report.status == "executed"
    assert report.selected_device == "cpu"
    assert (tmp_path / "stage-a.json").is_file()


def test_cpu_projection_over_limit_stops_after_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    epochs_seen: list[int] = []
    monkeypatch.setattr(stage_module, "probe_hardware", lambda: CPU_PROBE)
    monkeypatch.setattr(
        stage_module,
        "train_only",
        _recording_train_only(tmp_path, epochs_seen, []),
    )
    monkeypatch.setattr(stage_module.time, "perf_counter", _clock((0.0, 900.0)))

    report = stage_module.run_calibrated_stage(
        _settings_with_baseline(tmp_path),
        "A",
        tmp_path / "runtime-skip.json",
    )

    assert epochs_seen == [3]
    assert report.status == "skipped_runtime"
    assert report.projected_seconds == 9000.0
    assert report.gpu_command is not None and report.gpu_command.endswith("--stage A")


def test_b1_without_cuda_skips_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stage_module, "probe_hardware", lambda: CPU_PROBE)
    monkeypatch.setattr(
        stage_module,
        "train_only",
        lambda *args, **kwargs: pytest.fail("B1 must not train without CUDA"),
    )

    report = stage_module.run_calibrated_stage(
        _settings_with_baseline(tmp_path, epochs=50),
        "B1",
        tmp_path / "b1-skip.json",
    )

    assert report.status == "skipped_resource"
    assert report.reason == "CUDA_REQUIRED_FOR_B1"
