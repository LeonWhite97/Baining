# FC-BGA YOLO Training and Inference Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or execute inline with test-driven-development, requesting-code-review, and verification-before-completion. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible FC-BGA YOLOv8 training toolkit, a fail-closed real-image inference adapter, and publish the verified AOI project to `https://github.com/LeonWhite97/Baining` without rewriting remote history.

**Architecture:** Keep offline training under `tools/vision/fc_bga_yolo/` and runtime integration under `apps/api/app/inference/`. Formal data uses a versioned seven-class contract and R/G/B grayscale stacking; public CC BY 4.0 data stays isolated as smoke data. The existing demo runtime remains dependency-light, while shadow mode can opt into Ultralytics and persists REVIEW when the model is unavailable or cannot prove a normal result.

**Tech Stack:** Python 3.12, Ultralytics YOLOv8, Pillow, PyYAML, FastAPI, Pydantic, SQLAlchemy, pytest, Git.

## Global Constraints

- Project purpose is `portfolio_internal_poc`; Ultralytics commercial/closed-source licensing is not claimed.
- Formal detector class order is exactly `BALL_BRIDGE`, `MISSING_BALL`, `EXTRA_BALL`, `BALL_SIZE_ABNORMAL`, `BALL_OFFSET`, `BALL_SHAPE_ABNORMAL`, `FOREIGN_MATERIAL`.
- Runtime input contract is exactly `rgb_grayscale_stack_v1`: grayscale R, G, and B frames stacked in that order; RING remains required evidence but is not a model channel.
- Public datasets are smoke/reference data only and cannot be reported as FC-BGA production accuracy evidence.
- `NORMAL` is an empty label, not a detector class.
- `COPLANARITY` remains a 3D rule; solder void, head-in-pillow, and open remain X-ray scope.
- `UNKNOWN`, `MISSING_3D`, and `MISSING_LIGHT` remain decision/input states.
- The first detector sets `normal_confidence=0.0`; no-box and low-score outcomes are REVIEW, never automatic PASS.
- Demo installation and existing GPU-free Compose behavior must remain available without Ultralytics.
- Model, dependency, metadata, hash, or inference failures must fail closed to `MODEL_UNAVAILABLE`/REVIEW and preserve image evidence.
- Do not commit real AOI images, labels, downloaded public datasets, weights, run outputs, secrets, `.idea`, `.superpowers`, `.workbuddy`, caches, or `tmp`.
- Do not force-push. Publish by adding commits on top of the existing `origin/main` commit `58e4d21` or its then-current descendant.

---

## File Map

**Create training toolkit:**

- `tools/vision/fc_bga_yolo/__init__.py` - package marker.
- `tools/vision/fc_bga_yolo/contracts.py` - class and input contracts.
- `tools/vision/fc_bga_yolo/preprocessing.py` - offline R/G/B stack implementation.
- `tools/vision/fc_bga_yolo/model_metadata.py` - model package metadata and hash checks.
- `tools/vision/fc_bga_yolo/convert_dataset.py` - formal manifest conversion.
- `tools/vision/fc_bga_yolo/validate_yolo_dataset.py` - structural and leakage validation.
- `tools/vision/fc_bga_yolo/deduplicate_yolo_dataset.py` - conservative duplicate audit/apply.
- `tools/vision/fc_bga_yolo/download_models.py` - official pretrained-weight preparation.
- `tools/vision/fc_bga_yolo/download_public_smoke.py` - authorized Roboflow smoke download.
- `tools/vision/fc_bga_yolo/train.py` - smoke and PoC training entry point.
- `tools/vision/fc_bga_yolo/predict.py` - image/manifest prediction and JSONL output.
- `tools/vision/fc_bga_yolo/export_model.py` - ONNX/Engine export and metadata.
- `tools/vision/fc_bga_yolo/requirements-train.txt` - isolated training dependencies.
- `tools/vision/fc_bga_yolo/configs/*.yaml` - class, dataset, smoke, and PoC configurations.
- `tools/vision/fc_bga_yolo/tests/*.py` - toolkit tests.

**Create data contracts:**

- `data/external/fc_bga_public_smoke/README.md` - public data boundary.
- `data/external/fc_bga_public_smoke/sources.json` - exact URLs, licenses, versions, and purposes.
- `data/vision/fc_bga_defects/README.md` - formal manifest and annotation contract.
- `data/vision/fc_bga_defects/{manifests,train/images,train/labels,val/images,val/labels,test/images,test/labels}/.gitkeep` - directory template.

**Create or modify runtime integration:**

- `apps/api/app/inference/preprocessing.py` - runtime R/G/B stack.
- `apps/api/app/inference/ultralytics.py` - lazy Ultralytics adapter.
- `apps/api/app/inference/factory.py` - mode/backend selection.
- `apps/api/app/inference/base.py` - image and detection contracts.
- `apps/api/app/inference/demo.py` - new output contract.
- `apps/api/app/inference/tensorrt.py` - model version and new protocol.
- `apps/api/app/config.py` - inference settings.
- `apps/api/app/main.py` - adapter construction/injection.
- `apps/api/app/services/inference_orchestrator.py` - confidence selection and unavailable handling.
- `apps/api/app/api/routes/operations.py` - pass validated images and persist detections.
- `apps/api/pyproject.toml` - optional `vision` dependency.
- `apps/api/tests/test_adapters.py` - adapter/factory tests.
- `apps/api/tests/test_operations_api.py` - real-input integration/fail-closed tests.
- `apps/api/tests/test_runtime_mode.py` - backend/mode configuration tests.

**Modify project delivery files:**

- `README.md`, `.gitignore`, `.env.example`.
- `docs/PIS-IN_AOI_AI智能质检_V3.5_项目总说明书.md`.
- `docs/PIS-IN_AOI_AI智能质检_V3.5_部署运维手册.md`.
- `docs/runbooks/model-rollback.md`.
- `docs/superpowers/specs/2026-08-15-fc-bga-yolo-training-design.md`.
- `docs/superpowers/plans/2026-08-15-fc-bga-yolo-training.md`.

---

### Task 1: Establish the Git-backed Worktree and Import the AOI Baseline

**Files:**

- Create/Modify: `README.md`
- Modify: `.gitignore`
- Import: current AOI project files from `C:\Users\Windows\Desktop\PIS-IN_AOI_AI智能质检项目`

**Interfaces:**

- Consumes: existing remote `origin/main` and the current desktop AOI directory.
- Produces: a Git-backed implementation worktree on top of `origin/main`, with a baseline commit and no local-only artifacts.

- [ ] **Step 1: Clone the existing remote without rewriting its history**

Run from `D:\YOLO`:

```powershell
git clone https://github.com/LeonWhite97/Baining.git D:\YOLO\Baining
git -C D:\YOLO\Baining rev-parse --verify origin/main
```

Expected: `origin/main` resolves and the working tree contains the existing one-line `README.md`.

- [ ] **Step 2: Copy the AOI project snapshot into the Git worktree**

Use a non-mirroring copy so no destination history is deleted:

```powershell
robocopy 'C:\Users\Windows\Desktop\PIS-IN_AOI_AI智能质检项目' 'D:\YOLO\Baining' /E /XD .git .idea .superpowers .workbuddy tmp .pytest_cache .test-tmp __pycache__ /XF .env
if ($LASTEXITCODE -gt 7) { exit $LASTEXITCODE }
```

Expected: source files are present, `.git` still belongs to the clone, and excluded local directories are absent.

- [ ] **Step 3: Add repository hygiene before staging**

Add these exact patterns to `.gitignore` while preserving existing entries:

```gitignore
.env
.idea/
.superpowers/
.workbuddy/
tmp/
**/.test-tmp/
**/.pytest_cache/
**/__pycache__/
*.pyc

data/external/fc_bga_public_smoke/downloads/
data/vision/fc_bga_defects/**/images/*
data/vision/fc_bga_defects/**/labels/*
!data/vision/fc_bga_defects/**/images/.gitkeep
!data/vision/fc_bga_defects/**/labels/.gitkeep
tools/vision/fc_bga_yolo/weights/
tools/vision/fc_bga_yolo/runs/
*.pt
*.onnx
*.engine
```

Replace the remote placeholder README with a project README that states the PoC scope, current factual boundary, local startup entry points, and links to the total guide and FC-BGA design.

- [ ] **Step 4: Verify that staging contains no obvious secret or local artifact**

Run:

```powershell
git status --short
rg -n --hidden -g '!.git/**' -g '!package-lock.json' '(sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|POSTGRES_PASSWORD=.+[^=]$)' .
```

Expected: the second command finds no credential value. Documentation placeholders such as `replace_with_a_strong_password` are allowed after manual inspection.

- [ ] **Step 5: Commit the imported baseline**

```powershell
git add .
git commit -m "chore: import AOI project baseline"
```

Expected: the commit is a descendant of `origin/main`; no push occurs yet.

---

### Task 2: Add the Versioned Defect and Data Contracts

**Files:**

- Create: `tools/vision/fc_bga_yolo/__init__.py`
- Create: `tools/vision/fc_bga_yolo/contracts.py`
- Create: `tools/vision/fc_bga_yolo/configs/classes.yaml`
- Create: `tools/vision/fc_bga_yolo/configs/fc_bga_defects.template.yaml`
- Create: `tools/vision/fc_bga_yolo/configs/public_smoke.yaml`
- Create: `data/vision/fc_bga_defects/README.md`
- Create: `data/vision/fc_bga_defects/**/.gitkeep`
- Test: `tools/vision/fc_bga_yolo/tests/test_contracts.py`

**Interfaces:**

- Produces: `DEFECT_NAMES: tuple[str, ...]`, `INPUT_CONTRACT`, `REQUIRED_LIGHTS`, and `load_class_names(path: Path) -> tuple[str, ...]`.
- Consumed by: conversion, validation, model metadata, training, export, and runtime adapter tasks.

- [ ] **Step 1: Write the failing contract tests**

```python
from pathlib import Path

import pytest

from tools.vision.fc_bga_yolo.contracts import (
    DEFECT_NAMES,
    INPUT_CONTRACT,
    REQUIRED_LIGHTS,
    load_class_names,
)


def test_formal_class_order_is_stable() -> None:
    assert DEFECT_NAMES == (
        "BALL_BRIDGE",
        "MISSING_BALL",
        "EXTRA_BALL",
        "BALL_SIZE_ABNORMAL",
        "BALL_OFFSET",
        "BALL_SHAPE_ABNORMAL",
        "FOREIGN_MATERIAL",
    )
    assert INPUT_CONTRACT == "rgb_grayscale_stack_v1"
    assert REQUIRED_LIGHTS == ("R", "G", "B", "RING")


def test_classes_yaml_matches_python_contract() -> None:
    path = Path("tools/vision/fc_bga_yolo/configs/classes.yaml")
    assert load_class_names(path) == DEFECT_NAMES


def test_duplicate_class_names_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "classes.yaml"
    path.write_text("names:\n  0: BALL_BRIDGE\n  1: BALL_BRIDGE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_class_names(path)
```

- [ ] **Step 2: Run the tests and verify RED**

```powershell
python -m pytest tools/vision/fc_bga_yolo/tests/test_contracts.py -v
```

Expected: collection fails because `contracts.py` does not exist.

- [ ] **Step 3: Implement the exact contract**

Create constants and a strict YAML loader that requires integer keys `0..n-1`, non-empty unique string names, and exactly the formal order when reading `classes.yaml`.

The formal dataset template must contain:

```yaml
path: ../../../../data/vision/fc_bga_defects
train: train/images
val: val/images
test: test/images
names:
  0: BALL_BRIDGE
  1: MISSING_BALL
  2: EXTRA_BALL
  3: BALL_SIZE_ABNORMAL
  4: BALL_OFFSET
  5: BALL_SHAPE_ABNORMAL
  6: FOREIGN_MATERIAL
```

Document the missing-ball grid requirement, empty-label normal samples, and the exact JSONL schema in the data README.

- [ ] **Step 4: Run tests and verify GREEN**

```powershell
python -m pytest tools/vision/fc_bga_yolo/tests/test_contracts.py -v
```

Expected: all contract tests pass.

- [ ] **Step 5: Commit**

```powershell
git add tools/vision/fc_bga_yolo data/vision/fc_bga_defects
git commit -m "feat: define FC-BGA defect data contract"
```

---

### Task 3: Implement Four-light Manifest Conversion and R/G/B Stacking

**Files:**

- Create: `tools/vision/fc_bga_yolo/preprocessing.py`
- Create: `tools/vision/fc_bga_yolo/convert_dataset.py`
- Test: `tools/vision/fc_bga_yolo/tests/test_convert_dataset.py`

**Interfaces:**

- Produces: `stack_rgb_grayscale(r_path: Path, g_path: Path, b_path: Path) -> Image.Image`.
- Produces: `parse_source_manifest(path: Path) -> tuple[SourceSample, ...]`.
- Produces: `convert_manifest(manifest_path: Path, output_root: Path) -> ConversionReport`.
- Output manifest records four input hashes, label hash, output hash, split, group ID, and `rgb_grayscale_stack_v1`.

- [ ] **Step 1: Write failing channel-order and manifest tests**

```python
import hashlib
import json
from pathlib import Path

from PIL import Image

from tools.vision.fc_bga_yolo.convert_dataset import convert_manifest
from tools.vision.fc_bga_yolo.preprocessing import stack_rgb_grayscale


def test_stack_uses_r_g_b_grayscale_channel_order(tmp_path: Path) -> None:
    paths = []
    for name, value in (("R", 20), ("G", 100), ("B", 220)):
        path = tmp_path / f"{name}.png"
        Image.new("L", (2, 2), value).save(path)
        paths.append(path)
    stacked = stack_rgb_grayscale(*paths)
    assert stacked.mode == "RGB"
    assert stacked.getpixel((0, 0)) == (20, 100, 220)


def test_convert_requires_ring_but_does_not_use_it_as_a_channel(tmp_path: Path) -> None:
    images: dict[str, Path] = {}
    for light, value in (("R", 20), ("G", 100), ("B", 220), ("RING", 7)):
        path = tmp_path / f"{light}.png"
        Image.new("L", (2, 2), value).save(path)
        images[light] = path

    label = tmp_path / "sample-1.txt"
    label.write_text("", encoding="utf-8")
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "group_id": "lot-1",
                "split": "train",
                "images": {light: path.name for light, path in images.items()},
                "label": label.name,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "dataset"
    report = convert_manifest(manifest, output_root)

    output_image = output_root / "train/images/sample-1.png"
    assert Image.open(output_image).getpixel((0, 0)) == (20, 100, 220)
    assert (output_root / "train/labels/sample-1.txt").read_text(encoding="utf-8") == ""
    provenance = json.loads(report.output_manifest.read_text(encoding="utf-8").splitlines()[0])
    assert provenance["input_contract"] == "rgb_grayscale_stack_v1"
    assert provenance["input_sha256"]["RING"] == hashlib.sha256(images["RING"].read_bytes()).hexdigest()
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tools/vision/fc_bga_yolo/tests/test_convert_dataset.py -v
```

Expected: imports fail because conversion modules do not exist.

- [ ] **Step 3: Implement strict source parsing and conversion**

Use frozen dataclasses:

```python
@dataclass(frozen=True, slots=True)
class SourceSample:
    sample_id: str
    group_id: str
    split: str
    images: Mapping[str, Path]
    label: Path


@dataclass(frozen=True, slots=True)
class ConversionReport:
    samples: int
    split_counts: Mapping[str, int]
    output_manifest: Path
```

Reject duplicate sample IDs, missing/extra lights, unsupported splits, missing labels, unreadable images, dimensions above the API limits, and unequal frame dimensions. Save output as lossless PNG using an atomic temporary file followed by `Path.replace`.

- [ ] **Step 4: Add explicit failure tests**

Cover missing RING, unequal dimensions, duplicate sample IDs, path outside the manifest root, malformed JSON, and non-YOLO label lines. Each test asserts the public error code/message fragment and that no partial output sample remains.

- [ ] **Step 5: Run tests and verify GREEN**

```powershell
python -m pytest tools/vision/fc_bga_yolo/tests/test_convert_dataset.py -v
```

Expected: all conversion tests pass.

- [ ] **Step 6: Commit**

```powershell
git add tools/vision/fc_bga_yolo/preprocessing.py tools/vision/fc_bga_yolo/convert_dataset.py tools/vision/fc_bga_yolo/tests/test_convert_dataset.py
git commit -m "feat: convert FC-BGA four-light datasets"
```

---

### Task 4: Implement Dataset Validation and Conservative Deduplication

**Files:**

- Create: `tools/vision/fc_bga_yolo/validate_yolo_dataset.py`
- Create: `tools/vision/fc_bga_yolo/deduplicate_yolo_dataset.py`
- Test: `tools/vision/fc_bga_yolo/tests/test_validate_yolo_dataset.py`
- Test: `tools/vision/fc_bga_yolo/tests/test_deduplicate_yolo_dataset.py`

**Interfaces:**

- Produces: `validate_dataset(root: Path, class_names: tuple[str, ...], manifest: Path | None) -> ValidationReport`.
- Produces: `audit_duplicates(root: Path) -> DeduplicationReport`.
- Produces: `apply_duplicate_report(report: DeduplicationReport) -> DeduplicationReport`.

- [ ] **Step 1: Write failing validator tests**

Create a small real PNG/label fixture and assert:

```python
def test_empty_label_is_a_valid_normal_sample(dataset_root: Path) -> None:
    report = validate_dataset(dataset_root, DEFECT_NAMES, dataset_root / "manifest.json")
    assert report.errors == ()
    assert report.empty_labels == 1


@pytest.mark.parametrize("line", ["7 0.5 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2", "0 0.5 0.5 0 0.2"])
def test_invalid_label_values_fail(dataset_root: Path, line: str) -> None:
    write_label(dataset_root, line)
    assert validate_dataset(dataset_root, DEFECT_NAMES, None).errors
```

Add a manifest fixture where the same `group_id` appears in train and test and assert a `GROUP_LEAKAGE` error.

- [ ] **Step 2: Run validator tests and verify RED**

```powershell
python -m pytest tools/vision/fc_bga_yolo/tests/test_validate_yolo_dataset.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement validation core and CLI**

Keep core functions free of `SystemExit`; only `main()` converts errors to exit code 1. Count images, boxes, empty labels, classes by split, missing pairs, hash mismatches, and leakage. Print a deterministic JSON summary with `--json-report`.

- [ ] **Step 4: Write failing deduplication tests**

```python
def test_audit_only_never_deletes_files(duplicate_dataset: Path) -> None:
    report = audit_duplicates(duplicate_dataset)
    assert report.redundant_images == 1
    assert all(path.exists() for path in duplicate_dataset.rglob("*.png"))


def test_same_image_with_different_labels_is_a_conflict(conflicting_dataset: Path) -> None:
    report = audit_duplicates(conflicting_dataset)
    assert report.conflicts == 1
    with pytest.raises(ValueError, match="LABEL_CONFLICT"):
        apply_duplicate_report(report)
```

- [ ] **Step 5: Implement SHA-256 audit/apply**

Canonical priority is test, val, train. Normalize label whitespace before comparing. Apply only groups whose normalized labels are identical. Generate pre-apply and postcheck JSON; never merge boxes.

- [ ] **Step 6: Run both suites and verify GREEN**

```powershell
python -m pytest tools/vision/fc_bga_yolo/tests/test_validate_yolo_dataset.py tools/vision/fc_bga_yolo/tests/test_deduplicate_yolo_dataset.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add tools/vision/fc_bga_yolo/validate_yolo_dataset.py tools/vision/fc_bga_yolo/deduplicate_yolo_dataset.py tools/vision/fc_bga_yolo/tests
git commit -m "feat: validate and deduplicate AOI datasets"
```

---

### Task 5: Add Authorized Model and Public-smoke Downloads

**Files:**

- Create: `tools/vision/fc_bga_yolo/download_models.py`
- Create: `tools/vision/fc_bga_yolo/download_public_smoke.py`
- Create: `data/external/fc_bga_public_smoke/README.md`
- Create: `data/external/fc_bga_public_smoke/sources.json`
- Test: `tools/vision/fc_bga_yolo/tests/test_downloads.py`

**Interfaces:**

- Produces: `sha256_file(path: Path) -> str` and `verify_weight(path: Path) -> WeightInfo`.
- Produces: `download_public_smoke(destination: Path, api_key: str, downloader: PublicDatasetDownloader) -> Path`.
- Default public project is workspace `paween`, project `bga-ram-chips-detection-t3cqn`, version `1`, format `yolov8`.

- [ ] **Step 1: Write failing offline tests**

```python
def test_small_weight_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    path.write_bytes(b"not-a-weight")
    with pytest.raises(ValueError, match="too small"):
        verify_weight(path)


def test_public_download_requires_api_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ROBOFLOW_API_KEY"):
        download_public_smoke(tmp_path, "", downloader=RecordingDownloader())
```

The recording downloader writes a minimal dataset tree and records the exact workspace/project/version/format call; assert those values and the generated source manifest.

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tools/vision/fc_bga_yolo/tests/test_downloads.py -v
```

Expected: imports fail.

- [ ] **Step 3: Implement lazy third-party imports**

Import `ultralytics` and `roboflow` only inside download functions. `--verify-only` must not import either package or access the network. Do not log the API key. Normalize Roboflow `valid` to `val` only after checking the downloaded structure.

- [ ] **Step 4: Add the exact source manifest**

`sources.json` records the three researched resources, but marks only BGA RAM Chips as `default_download: true`; BGA-Balls is `segmentation_reference`, and X-ray Void is `xray_reference` with `default_download: false`.

- [ ] **Step 5: Run tests and verify GREEN**

```powershell
python -m pytest tools/vision/fc_bga_yolo/tests/test_downloads.py -v
```

Expected: all tests pass without network.

- [ ] **Step 6: Commit**

```powershell
git add tools/vision/fc_bga_yolo/download_models.py tools/vision/fc_bga_yolo/download_public_smoke.py tools/vision/fc_bga_yolo/tests/test_downloads.py data/external/fc_bga_public_smoke
git commit -m "feat: add verified YOLO asset downloads"
```

---

### Task 6: Add Training, Prediction, Export, and Model Metadata

**Files:**

- Create: `tools/vision/fc_bga_yolo/model_metadata.py`
- Create: `tools/vision/fc_bga_yolo/train.py`
- Create: `tools/vision/fc_bga_yolo/predict.py`
- Create: `tools/vision/fc_bga_yolo/export_model.py`
- Create: `tools/vision/fc_bga_yolo/configs/train_smoke.yaml`
- Create: `tools/vision/fc_bga_yolo/configs/train_poc.yaml`
- Create: `tools/vision/fc_bga_yolo/requirements-train.txt`
- Test: `tools/vision/fc_bga_yolo/tests/test_model_metadata.py`
- Test: `tools/vision/fc_bga_yolo/tests/test_training_commands.py`

**Interfaces:**

- Produces: `TrainingSettings`, `load_training_settings(path: Path)`, and `build_train_kwargs(settings) -> dict[str, object]`.
- Produces: `ModelMetadata`, `write_model_metadata(path: Path, metadata: ModelMetadata) -> Path`, and `validate_model_package(model_path: Path, metadata_path: Path, expected_names: tuple[str, ...]) -> ModelMetadata`.
- Produces JSONL predictions with model/input hashes and pixel xywh detections.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_poc_defaults_match_approved_design() -> None:
    settings = load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_poc.yaml"))
    assert settings.model.endswith("yolov8s.pt")
    assert settings.imgsz == 1280
    assert settings.epochs == 100
    assert settings.patience == 20
    assert settings.profile == "fc_bga"


def test_smoke_profile_cannot_claim_formal_classes() -> None:
    settings = load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_smoke.yaml"))
    assert settings.profile == "public_smoke"
    assert settings.epochs == 3
```

- [ ] **Step 2: Write failing metadata tamper tests**

Create a 2 MiB fake model file, write metadata with its hash, and assert validation passes. Change one byte and assert `MODEL_HASH_MISMATCH`. Change class order and assert `MODEL_CLASS_MISMATCH`.

- [ ] **Step 3: Run tests and verify RED**

```powershell
python -m pytest tools/vision/fc_bga_yolo/tests/test_model_metadata.py tools/vision/fc_bga_yolo/tests/test_training_commands.py -v
```

Expected: imports fail.

- [ ] **Step 4: Implement settings, training, and independent test evaluation**

`train.py` must validate formal data before importing Ultralytics, pass explicit train kwargs, locate `weights/best.pt`, run `model.val(data=..., split="test")`, and optionally invoke export. `--check-only` validates config/data without downloading or loading a model.

- [ ] **Step 5: Implement prediction and export**

Prediction keeps all boxes at inference threshold and writes deterministic JSONL. Export defaults to ONNX and validates the produced file. `--format engine` records platform, GPU, CUDA, TensorRT, PyTorch, and Ultralytics versions; if those prerequisites are absent, exit nonzero without a success manifest.

- [ ] **Step 6: Implement metadata generation**

Use a frozen dataclass with exact keys from the design: model version, task, names, input contract, imgsz, dataset manifest hash, model hashes, runtime versions, export settings, result paths, and `intended_use=portfolio_internal_poc`.

- [ ] **Step 7: Run tests and CLI checks**

```powershell
python -m pytest tools/vision/fc_bga_yolo/tests/test_model_metadata.py tools/vision/fc_bga_yolo/tests/test_training_commands.py -v
python tools/vision/fc_bga_yolo/train.py --help
python tools/vision/fc_bga_yolo/predict.py --help
python tools/vision/fc_bga_yolo/export_model.py --help
```

Expected: tests pass and all commands exit 0 without network.

- [ ] **Step 8: Commit**

```powershell
git add tools/vision/fc_bga_yolo
git commit -m "feat: add FC-BGA YOLO training workflow"
```

---

### Task 7: Extend Runtime Inference Contracts and Add Matching Preprocessing

**Files:**

- Modify: `apps/api/app/inference/base.py`
- Modify: `apps/api/app/inference/demo.py`
- Modify: `apps/api/app/inference/tensorrt.py`
- Create: `apps/api/app/inference/preprocessing.py`
- Modify: `apps/api/tests/test_adapters.py`

**Interfaces:**

- Produces: `InferenceImage`, `Detection`, and `InferenceRequest.images`.
- `InferenceAdapter` exposes `model_version: str` and `predict(request) -> InferenceOutput`.
- Runtime `stack_rgb_grayscale(images: tuple[InferenceImage, ...]) -> Image.Image` must byte-match offline preprocessing for the same files.

- [ ] **Step 1: Write failing inference-contract tests**

```python
def test_runtime_stack_matches_training_stack(tmp_path: Path) -> None:
    images = write_four_light_images(tmp_path, values={"R": 20, "G": 100, "B": 220, "RING": 7})
    offline = offline_stack(images["R"], images["G"], images["B"])
    runtime = runtime_stack(tuple(to_inference_image(light, path) for light, path in images.items()))
    assert runtime.tobytes() == offline.tobytes()


def test_demo_output_exposes_structured_detection() -> None:
    output = DemoInferenceAdapter().predict(InferenceRequest("event-42", "DEFECT", True, ()))
    assert output.detections[0].defect_code == "BALL_BRIDGE"
    assert output.detections[0].confidence == output.defect_score
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest apps/api/tests/test_adapters.py -v
```

Expected: missing `InferenceImage`/`Detection` or constructor mismatch.

- [ ] **Step 3: Implement the new dataclasses and adapters**

Use keyword construction in demo/TensorRT adapters. Store boxes as integer pixel xywh inside `Detection`; derive compatibility `boxes` from detections only if existing callers still require it. Require exactly one R/G/B/RING image for runtime stacking and check dimensions before reading pixel data.

- [ ] **Step 4: Run tests and verify GREEN**

```powershell
python -m pytest apps/api/tests/test_adapters.py -v
```

Expected: existing adapter tests plus new contract tests pass.

- [ ] **Step 5: Commit**

```powershell
git add apps/api/app/inference apps/api/tests/test_adapters.py
git commit -m "refactor: add structured inference evidence"
```

---

### Task 8: Add Ultralytics Adapter, Backend Factory, and Configuration

**Files:**

- Create: `apps/api/app/inference/ultralytics.py`
- Create: `apps/api/app/inference/factory.py`
- Modify: `apps/api/app/config.py`
- Modify: `apps/api/app/main.py`
- Modify: `apps/api/pyproject.toml`
- Modify: `apps/api/tests/test_adapters.py`
- Modify: `apps/api/tests/test_runtime_mode.py`

**Interfaces:**

- Produces: `InferenceBackend(StrEnum)` and `InferenceSettings.from_values(...)`.
- Produces: `build_inference_adapter(runtime_mode, inference_settings) -> InferenceAdapter`.
- Produces: `UltralyticsInferenceAdapter` with lazy model loading and metadata validation.
- Add keyword-only `inference_adapter: InferenceAdapter | None = None` to the existing `create_app` function so tests can inject an adapter.

- [ ] **Step 1: Write failing backend-selection tests**

```python
def test_demo_mode_defaults_to_demo_adapter() -> None:
    adapter = build_inference_adapter(RuntimeMode.DEMO, InferenceSettings.from_values())
    assert isinstance(adapter, DemoInferenceAdapter)


def test_shadow_mode_never_defaults_to_demo_adapter() -> None:
    adapter = build_inference_adapter(RuntimeMode.SHADOW, InferenceSettings.from_values())
    assert isinstance(adapter, UnavailableInferenceAdapter)


def test_explicit_demo_backend_is_rejected_outside_demo() -> None:
    with pytest.raises(ValueError, match="demo backend"):
        build_inference_adapter(RuntimeMode.CONTROLLED, InferenceSettings.from_values(backend="demo"))
```

- [ ] **Step 2: Write failing lazy-adapter tests**

Construct a valid fake model package and an injected model loader. Assert the loader is not called in `__init__`, is called once across two predictions, rejects changed hashes/classes, and converts row tuples `(x1, y1, x2, y2, confidence, class_id)` to `Detection` objects.

- [ ] **Step 3: Run tests and verify RED**

```powershell
python -m pytest apps/api/tests/test_adapters.py apps/api/tests/test_runtime_mode.py -v
```

Expected: missing backend/factory classes.

- [ ] **Step 4: Implement optional dependencies and configuration**

Add this optional dependency group without changing base dependencies:

```toml
[project.optional-dependencies]
test = ["pytest>=8,<9", "httpx>=0.28,<1"]
vision = ["ultralytics>=8.3,<9"]
```

Parse exact environment variables `AOI_INFERENCE_BACKEND`, `AOI_MODEL_PATH`, `AOI_MODEL_METADATA_PATH`, `AOI_MODEL_DEVICE`, `AOI_MODEL_IMGSZ`, and `AOI_MODEL_CONF`. Validate imgsz positive and confidence in `[0,1]`.

- [ ] **Step 5: Implement the adapter**

Validate metadata/hash before importing Ultralytics. Load once under a lock. Compose R/G/B, call the model with configured imgsz/device/conf, keep all valid class rows, set `normal_confidence=0.0`, select maximum confidence as `defect_score`/primary code, and wrap runtime/model errors as `InferenceUnavailable` without exposing filesystem internals in API responses.

- [ ] **Step 6: Run tests and verify GREEN**

```powershell
python -m pytest apps/api/tests/test_adapters.py apps/api/tests/test_runtime_mode.py -v
```

Expected: all tests pass without Ultralytics installed.

- [ ] **Step 7: Commit**

```powershell
git add apps/api/app/inference apps/api/app/config.py apps/api/app/main.py apps/api/pyproject.toml apps/api/tests
git commit -m "feat: add fail-closed Ultralytics adapter"
```

---

### Task 9: Connect Validated Images to Inference and Persist All Detections

**Files:**

- Modify: `apps/api/app/services/inference_orchestrator.py`
- Modify: `apps/api/app/api/routes/operations.py`
- Modify: `apps/api/tests/test_operations_api.py`
- Modify: `apps/api/tests/test_decision.py`

**Interfaces:**

- `run_inference(adapter, ..., images: tuple[InferenceImage, ...]) -> OrchestratedInference`.
- `OrchestratedInference` carries selected confidence, primary defect, all detections, model version, latency, and optional error code.
- Persisted `defect_bbox` items use exact keys `x`, `y`, `w`, `h`, `class_id`, `defect_code`, `confidence`.

- [ ] **Step 1: Write a failing image-forwarding API test**

Inject a recording adapter through `create_app`, import a real four-light fixture, and assert:

```python
assert tuple(image.light_id for image in adapter.requests[0].images) == ("R", "G", "B", "RING")
assert all(image.path.is_file() for image in adapter.requests[0].images)
```

- [ ] **Step 2: Write a failing multi-detection persistence test**

Return two detections with different classes/confidences. Assert response primary code is the higher-confidence class and stored JSON contains both detections with all seven keys.

- [ ] **Step 3: Write a failing unavailable-model test**

Use an adapter that raises `InferenceUnavailable`. Assert import returns 201, decision REVIEW, reason `MODEL_UNAVAILABLE`, no fake defect, evidence attachments remain four, and one auditable inference row is present.

- [ ] **Step 4: Run targeted tests and verify RED**

```powershell
python -m pytest apps/api/tests/test_operations_api.py apps/api/tests/test_decision.py -v
```

Expected: request images are absent or unavailable inference propagates as an error.

- [ ] **Step 5: Implement orchestration and route wiring**

Convert `ValidatedImage` to `InferenceImage` after validation. Use `request.app.state.inference_adapter`; do not instantiate Demo inside the route. On success choose confidence as defect score for FAIL, normal confidence for PASS, and the maximum of both for REVIEW. On unavailable inference produce REVIEW with zero confidence and the adapter's requested model version.

- [ ] **Step 6: Preserve idempotency behavior**

Run the existing sequential and concurrent import tests. The first import persists one event/four attachments/one inference result; identical retries return the same event; conflicting evidence remains quarantined.

- [ ] **Step 7: Run targeted and full API tests**

```powershell
python -m pytest apps/api/tests/test_operations_api.py apps/api/tests/test_decision.py -v
python -m pytest apps/api/tests -v
```

Expected: all API tests pass.

- [ ] **Step 8: Commit**

```powershell
git add apps/api/app/services/inference_orchestrator.py apps/api/app/api/routes/operations.py apps/api/tests
git commit -m "feat: connect verified images to YOLO inference"
```

---

### Task 10: Complete Documentation, Packaging Checks, and Full Verification

**Files:**

- Create: `tools/vision/fc_bga_yolo/README.md`
- Modify: `.env.example`
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `docs/PIS-IN_AOI_AI智能质检_V3.5_项目总说明书.md`
- Modify: `docs/PIS-IN_AOI_AI智能质检_V3.5_部署运维手册.md`
- Modify: `docs/runbooks/model-rollback.md`

**Interfaces:**

- Produces exact commands for environment setup, data conversion, validation, dedup audit/apply, public smoke, formal training, prediction, ONNX export, shadow runtime, and rollback.
- Preserves the factual boundary that no formal dataset, weight, Engine, or production metric is included.

- [ ] **Step 1: Write documentation against verified CLI help**

Document commands only after running each `--help`. Include Windows PowerShell and portable Python paths. Add public-source citations and the Ultralytics license notice.

- [ ] **Step 2: Update environment example**

Add safe defaults:

```dotenv
AOI_INFERENCE_BACKEND=demo
AOI_MODEL_PATH=
AOI_MODEL_METADATA_PATH=
AOI_MODEL_DEVICE=cpu
AOI_MODEL_IMGSZ=1280
AOI_MODEL_CONF=0.25
ROBOFLOW_API_KEY=
```

No real API key is committed.

- [ ] **Step 3: Update factual-boundary documents**

Replace statements that the repository has no training/real adapter code with the exact new state: code and templates exist; real four-light annotations, formal weights, TensorRT Engine, blind-test evidence, P95, throughput, false-positive, and escape-rate evidence do not.

- [ ] **Step 4: Run toolkit verification**

```powershell
python -m pytest tools/vision/fc_bga_yolo/tests -v
python tools/vision/fc_bga_yolo/convert_dataset.py --help
python tools/vision/fc_bga_yolo/validate_yolo_dataset.py --help
python tools/vision/fc_bga_yolo/deduplicate_yolo_dataset.py --help
python tools/vision/fc_bga_yolo/download_models.py --help
python tools/vision/fc_bga_yolo/download_public_smoke.py --help
python tools/vision/fc_bga_yolo/train.py --help
python tools/vision/fc_bga_yolo/predict.py --help
python tools/vision/fc_bga_yolo/export_model.py --help
```

Expected: all tests and commands exit 0 without network.

- [ ] **Step 5: Run all repository tests**

```powershell
python -m pytest apps/api/tests -v
python -m pytest services/agent-rag/tests -v
python -m pytest services/simulator/tests -v
npm --prefix apps/web test -- --run
npm --prefix apps/web run build
```

Expected: zero failures. If a bundled runtime is required, use the workspace dependency paths instead of installing unrelated global software.

- [ ] **Step 6: Inspect the final diff and artifact boundary**

```powershell
git status --short
git diff --check origin/main...HEAD
git diff --stat origin/main...HEAD
git ls-files | rg '(\.pt$|\.onnx$|\.engine$|/runs/|downloads/|^\.env$|^\.idea/|^tmp/)'
```

Expected: `git diff --check` exits 0 and the artifact scan returns no forbidden tracked file.

- [ ] **Step 7: Commit documentation**

```powershell
git add README.md .gitignore .env.example tools/vision/fc_bga_yolo/README.md docs
git commit -m "docs: document FC-BGA YOLO workflow"
```

---

### Task 11: Sync the Desktop Project and Publish GitHub Main

**Files:**

- Sync: only files changed or created by Tasks 1-10 to `C:\Users\Windows\Desktop\PIS-IN_AOI_AI智能质检项目`.
- Publish: commits in `D:\YOLO\Baining` to `origin/main`.

**Interfaces:**

- Consumes: verified clean worktree and current remote main.
- Produces: desktop AOI project with the same implementation files and GitHub `main` containing the verified commits.

- [ ] **Step 1: Re-check remote history before publishing**

```powershell
git fetch origin main
git merge-base --is-ancestor origin/main HEAD
```

Expected: exit 0. If not, integrate current `origin/main` with a normal rebase or merge and rerun the full verification affected by conflicts. Never force-push.

- [ ] **Step 2: Check for overlapping desktop edits**

Compare files to be synchronized against the hashes captured during baseline import. If a desktop file changed independently, inspect and reconcile it in the Git worktree before copying; do not overwrite it blindly.

- [ ] **Step 3: Copy implementation files back to the desktop project**

Use explicit `Copy-Item` targets for `tools/vision/fc_bga_yolo`, `data/vision/fc_bga_defects`, `data/external/fc_bga_public_smoke`, changed API files, tests, configs, and docs. Exclude `.git`, downloaded datasets, weights, runs, caches, and local environment files. Verify hashes for every copied file.

- [ ] **Step 4: Perform a final secret and artifact scan**

```powershell
rg -n --hidden -g '!.git/**' -g '!package-lock.json' '(sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY)' .
git status --short
```

Expected: no secret match and a clean worktree.

- [ ] **Step 5: Push without force**

```powershell
git push origin main
git ls-remote --symref origin HEAD
```

Expected: push succeeds and remote HEAD resolves to the new local HEAD. If Git Credential Manager requests authentication, complete the normal GitHub sign-in flow; do not put a token in a command or file.

- [ ] **Step 6: Verify the published commit**

```powershell
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

Expected: both hashes are identical. Record the commit hash and repository URL in the completion report.

---

## Plan Self-review

- Every confirmed design requirement maps to a task: data boundary (Tasks 2-6), R/G/B stack (Tasks 3 and 7), validation/dedup (Task 4), train/predict/export (Task 6), runtime adapter (Tasks 7-9), fail-closed safety (Tasks 8-9), documentation (Task 10), and GitHub publication (Tasks 1 and 11).
- The formal seven-class order and `rgb_grayscale_stack_v1` names are identical throughout the plan.
- Public smoke data is physically and semantically isolated from formal data.
- No task requires real data, a real API key, model download, GPU, or TensorRT to pass the base test suite.
- Publication preserves the existing remote commit and explicitly forbids force-push.
