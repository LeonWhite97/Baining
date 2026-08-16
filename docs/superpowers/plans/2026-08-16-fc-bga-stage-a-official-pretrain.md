# FC-BGA Stage A Official Pretrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the scratch-only public smoke run with a reproducible, resource-gated Stage A run that starts from the official `yolov8n.pt` checkpoint.

**Architecture:** Add a generic artifact-baseline module, a pure hardware/runtime decision module, and a calibrated stage runner. Keep `train.py` as the single-model training implementation and preserve its public `run_training(settings) -> Path` API while extracting reusable train/evaluate primitives for calibration and resume.

**Tech Stack:** Python 3.13, Ultralytics 8.4.120 within the repository's `>=8.3,<9` constraint, PyTorch, PyYAML, pytest, PowerShell on Windows.

## Global Constraints

- This is portfolio/internal PoC evidence, not production FC-BGA accuracy evidence.
- Stage A remains the separate `NG`/`OK` `public_smoke` profile and cannot emit deployable `model_metadata.json`.
- Use only the official Ultralytics `yolov8n.pt` asset; do not silently switch to an untrusted mirror.
- Prefer a hash-verified permitted cache; otherwise retry the official asset at most three total attempts and fail closed.
- Read API keys only from environment variables and never print or persist them.
- Keep checkpoints, virtual environments, downloads, and run binaries out of Git.
- Prefer CUDA device `0` after a successful probe; Stage A may fall back to CPU.
- Run three calibration epochs before the remaining Stage A epochs; stop with a resource report when projected CPU runtime exceeds 7,200 seconds.
- Stage A target parameters are image size 640, 30 total epochs, patience 10, batch 4, workers 0, and deterministic seed 42.
- Preserve the existing `run_training(settings) -> Path` interface and all formal `fc_bga` gates.

---

## File Map

- Create `tools/vision/fc_bga_yolo/artifact_manifest.py`: capture and verify source, size, hash, license, and retrieval metadata for cached artifacts.
- Create `tools/vision/fc_bga_yolo/training_runtime.py`: pure hardware probing and resource-decision types/functions.
- Create `tools/vision/fc_bga_yolo/run_training_stage.py`: calibrated Stage A/B0/B1 orchestration and JSON resource reports.
- Create `tools/vision/fc_bga_yolo/tests/test_artifact_manifest.py`: artifact baseline tests.
- Create `tools/vision/fc_bga_yolo/tests/test_training_runtime.py`: device and runtime-decision tests.
- Create `tools/vision/fc_bga_yolo/tests/test_training_stage.py`: calibration, resume, skip, and report tests.
- Modify `tools/vision/fc_bga_yolo/download_models.py`: record/verify an official model baseline without breaking direct `prepare_models()` callers.
- Modify `tools/vision/fc_bga_yolo/train.py`: extract train-only and evaluate-only primitives while retaining current behavior.
- Modify `tools/vision/fc_bga_yolo/configs/train_smoke.yaml`: approved Stage A defaults.
- Modify `tools/vision/fc_bga_yolo/tests/test_downloads.py`: downloader/baseline integration coverage.
- Modify `tools/vision/fc_bga_yolo/tests/test_training_commands.py`: refactor compatibility and Stage A default coverage.
- Modify `tools/vision/fc_bga_yolo/README.md`: Stage A commands and factual boundary.

---

### Task 1: Verified Official-Asset Baselines

**Files:**
- Create: `tools/vision/fc_bga_yolo/artifact_manifest.py`
- Create: `tools/vision/fc_bga_yolo/tests/test_artifact_manifest.py`
- Modify: `tools/vision/fc_bga_yolo/download_models.py`
- Modify: `tools/vision/fc_bga_yolo/tests/test_downloads.py`

**Interfaces:**
- Produces: `ArtifactRecord`, `capture_artifact_record()`, `load_artifact_records()`, `verify_artifact_record()`, and `write_artifact_records()`.
- Consumes later: `run_training_stage.py` requires `verify_artifact_record()` before loading a starting checkpoint.

- [ ] **Step 1: Write the failing artifact round-trip and tamper tests**

```python
def test_artifact_record_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    artifact = tmp_path / "yolov8n.pt"
    artifact.write_bytes(b"x" * (1024 * 1024))
    manifest = tmp_path / "artifact-manifest.json"
    record = capture_artifact_record(
        artifact,
        source_url=OFFICIAL_MODEL_URLS["yolov8n.pt"],
        license_url="https://www.ultralytics.com/license",
        retrieved_at="2026-08-16T12:00:00Z",
    )
    write_artifact_records(manifest, (record,))
    assert verify_artifact_record(artifact, load_artifact_records(manifest)[0]) == record
    artifact.write_bytes(b"y" * (1024 * 1024))
    with pytest.raises(ValueError, match="ARTIFACT_HASH_MISMATCH"):
        verify_artifact_record(artifact, record)
```

- [ ] **Step 2: Run the focused test and confirm the missing-module failure**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_artifact_manifest.py -q --basetemp .test-tmp/artifact-red`

Expected: FAIL during import because `artifact_manifest.py` does not exist.

- [ ] **Step 3: Implement the strict artifact record contract**

```python
@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    name: str
    source_url: str
    license_url: str
    retrieved_at: str
    size_bytes: int
    sha256: str
```

Implement the exact interfaces `capture_artifact_record(path: Path, *, source_url: str, license_url: str, retrieved_at: str) -> ArtifactRecord`, `load_artifact_records(path: Path) -> tuple[ArtifactRecord, ...]`, `verify_artifact_record(path: Path, record: ArtifactRecord) -> ArtifactRecord`, and `write_artifact_records(path: Path, records: tuple[ArtifactRecord, ...]) -> Path`. Validate exact JSON keys, HTTPS URLs, ISO-8601 UTC timestamps, minimum 1 MiB size, 64-character lowercase SHA-256, unique names, and atomic writes through `<name>.tmp` followed by `Path.replace()`.

- [ ] **Step 4: Run artifact tests and confirm green**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_artifact_manifest.py -q --basetemp .test-tmp/artifact-green`

Expected: all tests pass.

- [ ] **Step 5: Write failing downloader integration tests**

```python
def test_prepare_models_records_and_reuses_verified_baseline(tmp_path: Path) -> None:
    destination = tmp_path / "pretrained"
    manifest = destination / "artifact-manifest.json"
    calls: list[str] = []

    def recording_downloader(model_name: str, output: Path, *, force: bool) -> WeightInfo:
        calls.append(model_name)
        target = output / model_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * (1024 * 1024))
        return verify_weight(target)

    infos = prepare_models(
        ("yolov8n.pt",), destination, force=False,
        manifest_path=manifest, downloader=recording_downloader,
    )
    assert verify_artifact_record(infos[0].path, load_artifact_records(manifest)[0])

    def fail_if_called(model_name: str, output: Path, *, force: bool) -> WeightInfo:
        pytest.fail(f"verified cache unexpectedly downloaded again: {model_name}")

    prepare_models(
        ("yolov8n.pt",), destination, force=False,
        manifest_path=manifest, downloader=fail_if_called,
    )
    assert calls == ["yolov8n.pt"]

def test_prepare_models_stops_after_three_official_attempts(tmp_path: Path) -> None:
    calls = 0
    def failing_downloader(model_name: str, output: Path, *, force: bool) -> WeightInfo:
        nonlocal calls
        calls += 1
        raise OSError("simulated network failure")
    with pytest.raises(RuntimeError, match="OFFICIAL_DOWNLOAD_FAILED"):
        prepare_models(
            ("yolov8n.pt",), tmp_path / "pretrained", force=False,
            downloader=failing_downloader,
            manifest_path=tmp_path / "artifact-manifest.json",
            max_attempts=3,
        )
    assert calls == 3

def test_cached_weight_requires_matching_baseline(tmp_path: Path) -> None:
    destination = tmp_path / "pretrained"
    destination.mkdir()
    weight = destination / "yolov8n.pt"
    weight.write_bytes(b"x" * (1024 * 1024))
    manifest = destination / "artifact-manifest.json"
    with pytest.raises(ValueError, match="ARTIFACT_BASELINE_UNAVAILABLE"):
        prepare_models((weight.name,), destination, force=False, manifest_path=manifest)
    record = capture_artifact_record(
        weight,
        source_url=OFFICIAL_MODEL_URLS[weight.name],
        license_url=ULTRALYTICS_LICENSE_URL,
        retrieved_at="2026-08-16T12:00:00Z",
    )
    write_artifact_records(manifest, (record,))
    weight.write_bytes(b"y" * (1024 * 1024))
    with pytest.raises(ValueError, match="ARTIFACT_HASH_MISMATCH"):
        prepare_models((weight.name,), destination, force=False, manifest_path=manifest)
```

The cache test deliberately supplies no downloader: both failures must occur before any Ultralytics import or network attempt.

- [ ] **Step 6: Run the downloader tests and confirm red**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_downloads.py -q --basetemp .test-tmp/download-baseline-red`

Expected: FAIL because `prepare_models()` has no `manifest_path` parameter.

- [ ] **Step 7: Integrate official source metadata into the downloader**

Add:

```python
OFFICIAL_MODEL_URLS = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
    "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt",
}
ULTRALYTICS_LICENSE_URL = "https://www.ultralytics.com/license"
```

Extend `prepare_models(model_names: tuple[str, ...], destination: Path, *, force: bool, downloader: Callable[..., WeightInfo] = download_model, manifest_path: Path | None = None, max_attempts: int = 3)`. Preserve existing behavior when `manifest_path is None`; the CLI always defaults it to `<destination>/artifact-manifest.json`. Before any network request, reuse a cache only when its artifact baseline matches exactly. Otherwise call the official downloader at most `max_attempts` times, reject values outside 1..3, and never change the official URL or use a mirror. After a successful official download, capture and atomically write the baseline. Before reusing a cached CLI artifact, require and verify its baseline.

- [ ] **Step 8: Run downloader and artifact tests**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_artifact_manifest.py tools/vision/fc_bga_yolo/tests/test_downloads.py -q --basetemp .test-tmp/download-baseline-green`

Expected: all tests pass.

- [ ] **Step 9: Commit Task 1**

```powershell
git add tools/vision/fc_bga_yolo/artifact_manifest.py tools/vision/fc_bga_yolo/download_models.py tools/vision/fc_bga_yolo/tests/test_artifact_manifest.py tools/vision/fc_bga_yolo/tests/test_downloads.py
git commit -m "feat: verify official YOLO artifact baselines"
```

---

### Task 2: Hardware and Runtime Decisions

**Files:**
- Create: `tools/vision/fc_bga_yolo/training_runtime.py`
- Create: `tools/vision/fc_bga_yolo/tests/test_training_runtime.py`

**Interfaces:**
- Produces: `HardwareProbe`, `ResourceDecision`, `probe_hardware()`, `select_stage_device()`, and `estimate_total_seconds()`.
- Consumes later: `run_training_stage.run_calibrated_stage()`.

- [ ] **Step 1: Write failing CPU, CUDA, B1-skip, and timeout tests**

```python
def test_stage_a_falls_back_to_cpu() -> None:
    probe = HardwareProbe("2.13.0+cpu", False, 0, None, None, "test-cpu")
    assert select_stage_device("A", probe).device == "cpu"

def test_stage_b1_without_cuda_is_expected_skip() -> None:
    probe = HardwareProbe("2.13.0+cpu", False, 0, None, None, "test-cpu")
    decision = select_stage_device("B1", probe)
    assert decision.status == "skipped_resource"
    assert decision.device is None

def test_cpu_projection_over_two_hours_stops_long_run() -> None:
    assert estimate_total_seconds(900.0, completed_epochs=3, target_epochs=30) == 9000.0
```

- [ ] **Step 2: Run the focused test and confirm red**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_training_runtime.py -q --basetemp .test-tmp/runtime-red`

Expected: FAIL because `training_runtime.py` does not exist.

- [ ] **Step 3: Implement pure runtime types and functions**

```python
@dataclass(frozen=True, slots=True)
class HardwareProbe:
    torch_version: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    cuda_total_memory_bytes: int | None
    cpu_name: str

@dataclass(frozen=True, slots=True)
class ResourceDecision:
    status: Literal["ready", "skipped_resource", "skipped_runtime"]
    device: str | None
    projected_seconds: float | None
    reason: str
```

Implement `probe_hardware(torch_module: object | None = None) -> HardwareProbe`, `select_stage_device(stage: Literal["A", "B0", "B1"], probe: HardwareProbe) -> ResourceDecision`, and `estimate_total_seconds(elapsed_seconds: float, *, completed_epochs: int, target_epochs: int) -> float`. `select_stage_device()` returns CUDA `0` when available, CPU for A/B0 otherwise, and `skipped_resource` for B1 without CUDA. Reject invalid stages and nonpositive calibration values.

- [ ] **Step 4: Run focused tests and confirm green**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_training_runtime.py -q --basetemp .test-tmp/runtime-green`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add tools/vision/fc_bga_yolo/training_runtime.py tools/vision/fc_bga_yolo/tests/test_training_runtime.py
git commit -m "feat: add YOLO training resource gates"
```

---

### Task 3: Calibrated Training and Resume

**Files:**
- Create: `tools/vision/fc_bga_yolo/run_training_stage.py`
- Create: `tools/vision/fc_bga_yolo/tests/test_training_stage.py`
- Modify: `tools/vision/fc_bga_yolo/train.py`
- Modify: `tools/vision/fc_bga_yolo/tests/test_training_commands.py`

**Interfaces:**
- Produces in `train.py`: `TrainingArtifacts`, `train_only()`, and `evaluate_best()`; preserves `run_training() -> Path`.
- Produces in `run_training_stage.py`: `StageRunReport`, `run_calibrated_stage()`, and CLI `--stage/--config/--report`.
- Consumes: `HardwareProbe`, `ResourceDecision`, `verify_artifact_record()`.

- [ ] **Step 1: Write failing train-refactor compatibility tests**

```python
def test_train_only_resume_uses_last_checkpoint_and_target_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    class FakeModel:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path
        def train(self, **kwargs: object) -> object:
            run = tmp_path / f"run-{len(calls)}"
            weights = run / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"best")
            (weights / "last.pt").write_bytes(b"last")
            calls.append((self.model_path, kwargs))
            return SimpleNamespace(save_dir=run)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeModel))
    settings = replace(load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_smoke.yaml")), project=str(tmp_path))
    calibration = train_only(settings, epochs=3)
    train_only(settings, resume_from=calibration.last, epochs=30)
    assert calls[1][0] == str(calibration.last)
    assert calls[1][1]["resume"] == str(calibration.last)
    assert calls[1][1]["epochs"] == 30
```

- [ ] **Step 2: Run the train tests and confirm red**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_training_commands.py -q --basetemp .test-tmp/train-refactor-red`

Expected: FAIL because `TrainingArtifacts` and `train_only()` do not exist.

- [ ] **Step 3: Extract train-only and evaluate-only primitives**

```python
@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    save_dir: Path
    best: Path
    last: Path

def run_training(settings: TrainingSettings) -> Path:
    artifacts = train_only(settings)
    evaluate_best(settings, artifacts)
    return artifacts.best
```

Implement `train_only(settings: TrainingSettings, *, epochs: int | None = None, resume_from: Path | None = None) -> TrainingArtifacts` and `evaluate_best(settings: TrainingSettings, artifacts: TrainingArtifacts) -> object` beside the shown functions. Move the existing `YOLO(...).train(...)` call into `train_only()`, keep the independent test-set validation and formal metadata generation in `evaluate_best()`, and keep `run_training()` as the compatibility wrapper shown above. Require both `best.pt` and `last.pt`. For resume, construct `YOLO(str(resume_from))` and pass `resume=str(resume_from)` with the final total epoch target.

- [ ] **Step 4: Run training-command tests and confirm green**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_training_commands.py -q --basetemp .test-tmp/train-refactor-green`

Expected: all existing and new tests pass.

- [ ] **Step 5: Write failing calibrated-run tests**

```python
from tools.vision.fc_bga_yolo import run_training_stage as stage_module

CPU_PROBE = HardwareProbe("2.13.0+cpu", False, 0, None, None, "test-cpu")

def recording_train_only(
    recorded_epochs: list[int],
    recorded_resume: list[Path | None],
    tmp_path: Path,
) -> Callable[..., TrainingArtifacts]:
    def fake_train_only(
        settings: TrainingSettings,
        *,
        epochs: int | None = None,
        resume_from: Path | None = None,
    ) -> TrainingArtifacts:
        assert epochs is not None
        recorded_epochs.append(epochs)
        recorded_resume.append(resume_from)
        save_dir = tmp_path / f"segment-{len(recorded_epochs)}"
        weights = save_dir / "weights"
        weights.mkdir(parents=True)
        best = weights / "best.pt"
        last = weights / "last.pt"
        best.write_bytes(b"best")
        last.write_bytes(b"last")
        return TrainingArtifacts(save_dir, best, last)
    return fake_train_only

def increasing_clock(values: tuple[float, ...]) -> Callable[[], float]:
    iterator = iter(values)
    return lambda: next(iterator)

def test_stage_a_calibrates_then_resumes_without_repeating_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_epochs: list[int] = []
    recorded_resume: list[Path | None] = []
    monkeypatch.setattr(stage_module, "probe_hardware", lambda: CPU_PROBE)
    monkeypatch.setattr(stage_module, "train_only", recording_train_only(recorded_epochs, recorded_resume, tmp_path))
    monkeypatch.setattr(stage_module, "evaluate_best", lambda settings, artifacts: SimpleNamespace(save_dir=tmp_path / "test"))
    monkeypatch.setattr(stage_module.time, "perf_counter", increasing_clock((0.0, 30.0)))
    settings = load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_smoke.yaml"))
    report = run_calibrated_stage(settings, "A", tmp_path / "report.json")
    assert recorded_epochs == [3, 30]
    assert recorded_resume[0] is None
    assert recorded_resume[1] is not None
    assert report.status == "executed"

def test_cpu_projection_over_limit_stops_after_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_epochs: list[int] = []
    monkeypatch.setattr(stage_module, "probe_hardware", lambda: CPU_PROBE)
    monkeypatch.setattr(stage_module, "train_only", recording_train_only(recorded_epochs, [], tmp_path))
    monkeypatch.setattr(stage_module.time, "perf_counter", increasing_clock((0.0, 900.0)))
    settings = load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_smoke.yaml"))
    report = run_calibrated_stage(settings, "A", tmp_path / "report.json")
    assert recorded_epochs == [3]
    assert report.status == "skipped_runtime"
    assert report.gpu_command.endswith("--stage A")
```

- [ ] **Step 6: Run calibrated-run tests and confirm red**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_training_stage.py -q --basetemp .test-tmp/stage-runner-red`

Expected: FAIL because `run_training_stage.py` does not exist.

- [ ] **Step 7: Implement the calibrated stage runner**

```python
@dataclass(frozen=True, slots=True)
class StageRunReport:
    stage: str
    status: str
    hardware: Mapping[str, object]
    calibration_seconds: float | None
    projected_seconds: float | None
    selected_device: str | None
    best_checkpoint: str | None
    gpu_command: str | None
    reason: str
```

Implement `run_calibrated_stage(settings: TrainingSettings, stage: Literal["A", "B0", "B1"], report_path: Path, *, runtime_limit_seconds: float = 7200.0) -> StageRunReport`. Verify the starting checkpoint against its artifact baseline, resolve `device: auto`, run three epochs, measure with `time.perf_counter()`, and resume from calibration `last.pt` only when allowed. Write the report atomically for `executed`, `skipped_runtime`, `skipped_resource`, and failure outcomes. The CLI exits `0` for `executed` and expected skips, and nonzero for invalid configuration or failed training.

- [ ] **Step 8: Run stage-runner and training tests**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_training_stage.py tools/vision/fc_bga_yolo/tests/test_training_commands.py -q --basetemp .test-tmp/stage-runner-green`

Expected: all tests pass.

- [ ] **Step 9: Commit Task 3**

```powershell
git add tools/vision/fc_bga_yolo/train.py tools/vision/fc_bga_yolo/run_training_stage.py tools/vision/fc_bga_yolo/tests/test_training_commands.py tools/vision/fc_bga_yolo/tests/test_training_stage.py
git commit -m "feat: calibrate and resume YOLO training stages"
```

---

### Task 4: Stage A Configuration, Documentation, and Real Run

**Files:**
- Modify: `tools/vision/fc_bga_yolo/configs/train_smoke.yaml`
- Modify: `tools/vision/fc_bga_yolo/README.md`
- Modify: `tools/vision/fc_bga_yolo/tests/test_training_commands.py`
- Runtime-only: `tools/vision/fc_bga_yolo/weights/pretrained/yolov8n.pt`
- Runtime-only: `tools/vision/fc_bga_yolo/weights/pretrained/artifact-manifest.json`
- Runtime-only: `.test-tmp/training-runs/stage-a-official/`

**Interfaces:**
- Consumes: Stage runner, official artifact baseline, and existing public smoke dataset.
- Produces: verified local `best.pt`, `results.csv`, curves, confusion matrix, test metrics, and `stage-a-resource-report.json`; none are deployable or committed.

- [ ] **Step 1: Update the Stage A defaults test first**

```python
def test_smoke_profile_matches_stage_a_design() -> None:
    settings = load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_smoke.yaml"))
    assert settings.profile == "public_smoke"
    assert settings.model.endswith("weights/pretrained/yolov8n.pt")
    assert (settings.imgsz, settings.epochs, settings.patience) == (640, 30, 10)
    assert (settings.batch, settings.device, settings.workers, settings.seed) == (4, "auto", 0, 42)
```

- [ ] **Step 2: Run the default test and confirm red**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_training_commands.py::test_smoke_profile_matches_stage_a_design -q`

Expected: FAIL because the current config is 640/3 epochs and CPU.

- [ ] **Step 3: Apply the approved Stage A config and README commands**

Set `imgsz: 640`, `epochs: 30`, `patience: 10`, `batch: 4`, `device: auto`, `workers: 0`, `seed: 42`, and `name: public_smoke_official_yolov8n`. Document the calibration report, CPU fallback, official-baseline verification, and non-deployable boundary.

- [ ] **Step 4: Run the full FC-BGA toolkit tests**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests -q --basetemp .test-tmp/stage-a-suite`

Expected: all tests pass.

- [ ] **Step 5: Download and verify the official checkpoint**

```powershell
.\.test-tmp\venv-yolo\Scripts\python.exe tools\vision\fc_bga_yolo\download_models.py --models yolov8n.pt
.\.test-tmp\venv-yolo\Scripts\python.exe tools\vision\fc_bga_yolo\download_models.py --verify-only tools\vision\fc_bga_yolo\weights\pretrained\yolov8n.pt
```

Expected: checkpoint size is at least 1 MiB, the SHA-256 is printed, and `artifact-manifest.json` matches it.

- [ ] **Step 6: Validate the public smoke dataset before training**

Run: `.\.test-tmp\venv-yolo\Scripts\python.exe tools\vision\fc_bga_yolo\train.py --config tools\vision\fc_bga_yolo\configs\train_smoke.yaml --check-only`

Expected: `training preflight passed`.

- [ ] **Step 7: Execute Stage A through the resource gate**

```powershell
.\.test-tmp\venv-yolo\Scripts\python.exe tools\vision\fc_bga_yolo\run_training_stage.py --stage A --config tools\vision\fc_bga_yolo\configs\train_smoke.yaml --report .test-tmp\training-runs\stage-a-resource-report.json
```

Expected on the current host: three CPU calibration epochs, projected runtime below two hours, resume to 30 total epochs, test evaluation, and status `executed`. If the measured projection exceeds the limit, expected status is `skipped_runtime` with a GPU command; do not bypass the gate.

- [ ] **Step 8: Verify Stage A artifacts and factual boundary**

Run the downloader `--verify-only` against the generated `best.pt`, inspect `results.csv`, and assert no `model_metadata.json` exists beside the public smoke weight. Record the overall and per-class test metrics in the final handoff with the statement that they are public `NG`/`OK` smoke metrics only.

- [ ] **Step 9: Commit Task 4 documentation/config changes**

```powershell
git add tools/vision/fc_bga_yolo/configs/train_smoke.yaml tools/vision/fc_bga_yolo/README.md tools/vision/fc_bga_yolo/tests/test_training_commands.py
git commit -m "feat: configure official YOLOv8n Stage A training"
```

- [ ] **Step 10: Final Plan 1 verification**

Run:

```powershell
python -m pytest tools/vision/fc_bga_yolo/tests -q --basetemp .test-tmp/stage-a-final
git diff --check
git status --short
```

Expected: test suite passes, no whitespace errors, runtime artifacts remain ignored, and only intentional committed files appear in Git history.
