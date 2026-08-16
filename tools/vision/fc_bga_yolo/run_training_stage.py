from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time
from typing import Literal, Mapping

try:
    from .artifact_manifest import load_artifact_records, verify_artifact_record
    from .train import (
        TrainingSettings,
        evaluate_best,
        load_training_settings,
        train_only,
    )
    from .training_runtime import (
        StageName,
        estimate_total_seconds,
        probe_hardware,
        select_stage_device,
    )
except ImportError:
    from artifact_manifest import load_artifact_records, verify_artifact_record
    from train import TrainingSettings, evaluate_best, load_training_settings, train_only
    from training_runtime import StageName, estimate_total_seconds, probe_hardware, select_stage_device


@dataclass(frozen=True, slots=True)
class StageRunReport:
    stage: str
    status: Literal["executed", "skipped_runtime", "skipped_resource", "failed"]
    hardware: Mapping[str, object]
    calibration_seconds: float | None
    projected_seconds: float | None
    selected_device: str | None
    best_checkpoint: str | None
    gpu_command: str | None
    reason: str


def _write_report(path: Path, report: StageRunReport) -> StageRunReport:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return report


def _verify_starting_checkpoint(settings: TrainingSettings) -> None:
    checkpoint = Path(settings.model)
    manifest = checkpoint.parent / "artifact-manifest.json"
    records = load_artifact_records(manifest)
    record = next((item for item in records if item.name == checkpoint.name), None)
    if record is None:
        raise ValueError("ARTIFACT_BASELINE_UNAVAILABLE")
    verify_artifact_record(checkpoint, record)


def run_calibrated_stage(
    settings: TrainingSettings,
    stage: StageName,
    report_path: Path,
    *,
    runtime_limit_seconds: float = 7200.0,
) -> StageRunReport:
    if runtime_limit_seconds <= 0:
        raise ValueError("RUNTIME_LIMIT_INVALID")
    hardware = probe_hardware()
    hardware_record = asdict(hardware)
    decision = select_stage_device(stage, hardware)
    if decision.status == "skipped_resource":
        return _write_report(
            report_path,
            StageRunReport(
                stage=stage,
                status="skipped_resource",
                hardware=hardware_record,
                calibration_seconds=None,
                projected_seconds=None,
                selected_device=None,
                best_checkpoint=None,
                gpu_command=None,
                reason=decision.reason,
            ),
        )

    _verify_starting_checkpoint(settings)
    selected = replace(settings, device=str(decision.device))
    calibration_epochs = min(3, selected.epochs)
    started = time.perf_counter()
    calibration = train_only(selected, epochs=calibration_epochs)
    calibration_seconds = time.perf_counter() - started
    projected = estimate_total_seconds(
        calibration_seconds,
        completed_epochs=calibration_epochs,
        target_epochs=selected.epochs,
    )
    if decision.device == "cpu" and projected > runtime_limit_seconds:
        return _write_report(
            report_path,
            StageRunReport(
                stage=stage,
                status="skipped_runtime",
                hardware=hardware_record,
                calibration_seconds=calibration_seconds,
                projected_seconds=projected,
                selected_device="cpu",
                best_checkpoint=str(calibration.best),
                gpu_command=(
                    "python tools/vision/fc_bga_yolo/run_training_stage.py "
                    f"--stage {stage}"
                ),
                reason="CPU_PROJECTED_RUNTIME_EXCEEDED",
            ),
        )

    final_artifacts = calibration
    if selected.epochs > calibration_epochs:
        final_artifacts = train_only(
            selected,
            epochs=selected.epochs,
            resume_from=calibration.last,
        )
    evaluate_best(selected, final_artifacts)
    return _write_report(
        report_path,
        StageRunReport(
            stage=stage,
            status="executed",
            hardware=hardware_record,
            calibration_seconds=calibration_seconds,
            projected_seconds=projected,
            selected_device=str(decision.device),
            best_checkpoint=str(final_artifacts.best),
            gpu_command=None,
            reason="TRAINING_AND_TEST_EVALUATION_COMPLETED",
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a calibrated FC-BGA YOLO training stage.")
    parser.add_argument("--stage", choices=("A", "B0", "B1"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run_calibrated_stage(
        load_training_settings(args.config),
        args.stage,
        args.report,
    )
    print(json.dumps(asdict(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
