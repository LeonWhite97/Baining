# FC-BGA YOLOv8 PoC Toolkit

This toolkit adds reproducible FC-BGA data preparation, validation, deduplication, YOLOv8 fine-tuning, prediction, export, and API integration. It is a portfolio/internal PoC, not production accuracy evidence.

## Factual Boundary

- The formal detector has exactly seven visible-light classes: `BALL_BRIDGE`, `MISSING_BALL`, `EXTRA_BALL`, `BALL_SIZE_ABNORMAL`, `BALL_OFFSET`, `BALL_SHAPE_ABNORMAL`, and `FOREIGN_MATERIAL`.
- `NORMAL` is represented by an empty label. `UNKNOWN`, `MISSING_3D`, and `MISSING_LIGHT` are decision or input states, not model classes.
- `COPLANARITY` remains a 3D rule. Solder void, head-in-pillow, and open defects remain X-ray scope.
- Input contract `rgb_grayscale_stack_v1` converts R/G/B frames to grayscale and stacks them in R, G, B order. RING is required evidence but is not a model channel.
- No same-camera production study in this repository proves that R/G/B grayscale stacking improves FC-BGA detection. Treat it as a versioned hypothesis and compare it against single-light and other fusion baselines using site data.
- No formal four-light images, labels, weights, TensorRT Engine, blind-test results, production metrics, or site latency measurements are committed.

## License Boundary

Ultralytics is distributed under AGPL-3.0 with a separate enterprise license option. Review the [Ultralytics licensing terms](https://www.ultralytics.com/license) before closed-source or commercial deployment. This repository does not grant an Ultralytics enterprise license.

Public datasets listed under `data/external/fc_bga_public_smoke/sources.json` retain their own licenses and attribution requirements. Public smoke results cannot be reported as formal FC-BGA performance.

## Environment

From the repository root:

```powershell
python -m venv .venv-yolo
.\.venv-yolo\Scripts\python.exe -m pip install --upgrade pip
.\.venv-yolo\Scripts\python.exe -m pip install -r tools/vision/fc_bga_yolo/requirements-train.txt
```

Base tests and every `--help` command run without downloading a model or dataset:

```powershell
python -m pytest tools/vision/fc_bga_yolo/tests -q --basetemp .test-tmp/fc-bga
python tools/vision/fc_bga_yolo/train.py --help
python tools/vision/fc_bga_yolo/predict.py --help
python tools/vision/fc_bga_yolo/export_model.py --help
```

## Formal Data Conversion

Create a UTF-8 JSONL source manifest outside Git. Each line identifies one physical sample and all four registered images:

```json
{"sample_id":"LOT01-TRAY02-A07","group_id":"LOT01","split":"train","images":{"R":"raw/R.png","G":"raw/G.png","B":"raw/B.png","RING":"raw/RING.png"},"label":"annotations/LOT01-TRAY02-A07.txt"}
```

Then convert and validate:

```powershell
python tools/vision/fc_bga_yolo/convert_dataset.py D:/aoi-data/source.jsonl data/vision/fc_bga_defects
python tools/vision/fc_bga_yolo/validate_yolo_dataset.py data/vision/fc_bga_defects --manifest data/vision/fc_bga_defects/manifest.jsonl --json-report D:/aoi-runs/dataset-validation.json
```

The converter rejects path traversal, missing/extra lights, duplicate sample IDs, invalid YOLO rows, decode failures, and unequal dimensions. It builds in a same-volume staging directory and replaces the prior generated tree only after every sample succeeds, while preserving scaffold documentation. Formal image artifacts are manifest-registered PNG files; unsupported image or label files and unmanifested nested artifacts are rejected. The validator also rejects image/label mismatches, non-finite or out-of-range boxes, changed image or label hashes, input-contract drift, and `group_id` leakage across train/val/test.

## Mock Four-light Capture

Use this only to test the data contract before real same-camera R/G/B/RING captures exist. It copies each existing image into four PNG evidence files, writes an empty YOLO label, and emits a `source.jsonl` that can be passed to the formal converter:

```powershell
python tools/vision/fc_bga_yolo/mock_capture_dataset.py D:/temp-microscope-images D:/aoi-data/fc_bga_mock_capture --prefix MOCK
python tools/vision/fc_bga_yolo/convert_dataset.py D:/aoi-data/fc_bga_mock_capture/source.jsonl data/vision/fc_bga_defects
python tools/vision/fc_bga_yolo/validate_yolo_dataset.py data/vision/fc_bga_defects --manifest data/vision/fc_bga_defects/manifest.jsonl --json-report D:/aoi-runs/mock-dataset-validation.json
```

The default `group_id` isolates each mock sample so train/val/test validation can pass without group leakage. Use a fixed `--group-id` only when you intentionally want samples to stay in one physical group. The generated R/G/B/RING images are simulated copies from one source image. They prove file layout, manifest generation, conversion, and validation only; they are not formal four-light evidence and must not be used for production accuracy claims.

## Deduplication

Audit is the default and never changes data:

```powershell
python tools/vision/fc_bga_yolo/deduplicate_yolo_dataset.py data/vision/fc_bga_defects --json-report D:/aoi-runs/dedup-audit.json
```

After review, apply only exact-image duplicates with identical normalized labels:

```powershell
python tools/vision/fc_bga_yolo/deduplicate_yolo_dataset.py data/vision/fc_bga_defects --apply --json-report D:/aoi-runs/dedup-applied.json
```

Conflicting labels stop apply with `LABEL_CONFLICT`; boxes are never merged automatically. Canonical priority is test, val, then train. Apply updates `manifest.jsonl` and includes a fresh duplicate postcheck in the JSON report.

## Public Smoke

The default public smoke source is Roboflow `paween/bga-ram-chips-detection-t3cqn`, version 1, format `yolov8`, licensed CC BY 4.0. Supply the key through the environment; it is never printed or stored:

```powershell
$env:ROBOFLOW_API_KEY = "<set-in-your-shell>"
python tools/vision/fc_bga_yolo/download_public_smoke.py
python tools/vision/fc_bga_yolo/download_models.py --models yolov8n.pt
python tools/vision/fc_bga_yolo/train.py --config tools/vision/fc_bga_yolo/configs/train_smoke.yaml --check-only
python tools/vision/fc_bga_yolo/run_training_stage.py --stage A --config tools/vision/fc_bga_yolo/configs/train_smoke.yaml --report .test-tmp/training-runs/stage-a-resource-report.json
```

Stage A starts from the hash-verified official `yolov8n.pt` baseline and uses 640 pixels, 30 total epochs, patience 10, batch 4, workers 0, and seed 42. The runner prefers CUDA device `0`; on CPU it measures three calibration epochs and continues the remaining 27 epochs from the calibration `last.pt` weights only when the projected total remains within 7,200 seconds. Ultralytics marks a completed three-epoch checkpoint as non-resumable, so the continuation starts a fresh optimizer without repeating the first three weight-training epochs. A longer projection returns `skipped_runtime` with a GPU command instead of starting the remaining epochs.

The dataset's `OK/NG` labels are separate from the formal seven-class contract. Smoke training validates its actual checkpoint names but never writes a deployable seven-class `model_metadata.json`. Its metrics are workflow evidence only and cannot support FC-BGA production claims.

The 2026-08-17 CPU run completed with status `executed`. Three calibration epochs took 124.915 seconds and projected 30 epochs at 1,249.147 seconds. The independent test call covered 17 images and 292 `NG/OK` boxes, with diagnostic P=0.927, R=0.955, mAP50=0.957, and mAP50-95=0.767. The test images are derived from the same public validation source, so these values are not independent FC-BGA defect evidence and must not be compared with a future seven-class or four-light model.

## Public External Seven-Class Rehearsal

Use `train_public_external_b0.yaml` only with immutable `public-external-v0.1`, and `train_public_external_b1.yaml` only with `public-external-v0.2`. Preflight verifies the fixed seven-class order, revision stage/version, manifest and assignment hashes, and nonempty train/val/test splits. Public checkpoints cannot write deployable model metadata.

The evaluation wrapper preserves the native Ultralytics aggregate, reports empty test classes as `null` with status `no_evidence`, and computes the reported aggregate over classes with nonzero GT only. B1 adds a 1,000-resample 95% interval only when the test set contains at least 30 independent `source_group_id` values; every resample moves all images in a group as one block.

As of 2026-08-17, the licensed source queue contains 110 unique candidates across two Roboflow sources (56 from BGA RAM Chips Detection at 300 x 300, 54 from BGA-Balls at 640 x 640) but zero completed seven-class human reviews. B0 is `blocked_data`, no v0.1 revision or B0 checkpoint exists, and B1 is `skipped_resource` on the CPU-only host. See `data/external/fc_bga_public_external/README.md` for the exact gate evidence.

For additional public substitute sources, use boundaries, and manual download commands, see `data/external/fc_bga_public_smoke/public_alternatives.md`.

## Formal Fine-tuning

Download the official starting weight and run preflight before GPU work:

```powershell
python tools/vision/fc_bga_yolo/download_models.py
python tools/vision/fc_bga_yolo/train.py --config tools/vision/fc_bga_yolo/configs/train_poc.yaml --check-only
python tools/vision/fc_bga_yolo/train.py --config tools/vision/fc_bga_yolo/configs/train_poc.yaml --epochs 100 --batch 8 --device 0 --workers 4 --lr0 0.01
```

The default download prepares official `yolov8n.pt` and `yolov8s.pt` weights under `weights/pretrained`; use `--models` to select a subset, `--force` to replace destination files, or `--verify-only PATH` for an offline integrity check. The PoC profile uses 1280 pixels, 100 epochs, patience 20, batch 8, seed 42, `lr0=0.01`, and deterministic mode. It requires exact seven-class YAML names, nonempty train/val/test splits, and fixed YAML split paths (`train/images`, `val/images`, `test/images`). Preflight validates the manifest-closed dataset before importing Ultralytics, evaluates `best.pt` on the independent test split, checks the checkpoint's actual names, and writes `model_metadata.json` beside `best.pt`.

## Prediction and Export

```powershell
python tools/vision/fc_bga_yolo/predict.py --model D:/aoi-runs/weights/best.pt --metadata D:/aoi-runs/weights/model_metadata.json --source D:/aoi-data/predict --output D:/aoi-runs/predictions.jsonl --device 0
python tools/vision/fc_bga_yolo/export_model.py --model D:/aoi-runs/weights/best.pt --metadata D:/aoi-runs/weights/model_metadata.json --format onnx --device cpu --opset 17
```

Prediction writes the JSONL file, deterministic annotated PNG files under `annotated/`, and `<output-stem>.summary.json` with per-class image/box/confidence counts. Every record includes input, model, and inference-configuration hashes. Use `--annotated-dir` and `--summary` to override those output paths. ONNX metadata records opset, imgsz, device, dynamic, simplify, and output SHA-256; `--dynamic` and `--simplify` are opt-in.

TensorRT Engine export is intentionally host-specific:

```powershell
python tools/vision/fc_bga_yolo/export_model.py --model D:/aoi-runs/weights/best.pt --metadata D:/aoi-runs/weights/model_metadata.json --format engine --device 0
```

Engine export exits nonzero unless CUDA, a visible GPU, TensorRT, PyTorch, and Ultralytics are available. The metadata records the relevant runtime versions and hashes.

## API Shadow Mode

Install the optional runtime dependency in the API environment, then point only to a validated model package:

```powershell
Push-Location apps/api
python -m pip install -e ".[vision]"
$env:APP_MODE = "shadow"
$env:AOI_INFERENCE_BACKEND = "ultralytics"
$env:AOI_MODEL_PATH = "D:/aoi-runs/weights/best.pt"
$env:AOI_MODEL_METADATA_PATH = "D:/aoi-runs/weights/model_metadata.json"
$env:AOI_MODEL_DEVICE = "0"
$env:AOI_MODEL_IMGSZ = "1280"
$env:AOI_MODEL_CONF = "0.25"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
Pop-Location
```

The adapter checks metadata class order, the loaded checkpoint's actual names, input contract, imgsz, and model SHA-256 before inference. Missing dependencies, changed files, load failures, class drift, invalid output, or runtime errors persist an auditable `REVIEW` with reason `MODEL_UNAVAILABLE`. With this first detector, no-box and low-score results also remain REVIEW because `normal_confidence` is intentionally zero.

## Research References

- [Basler BGA inspection use case](https://www.baslerweb.cn/zh-cn/use-cases/semicon-bga-inspections/) for domain imaging context.
- [BGA RAM Chips Detection](https://universe.roboflow.com/paween/bga-ram-chips-detection-t3cqn), CC BY 4.0, public smoke only.
- [BGA-Balls](https://universe.roboflow.com/kenshin-blirtz/bga-balls-3ihxj), CC BY 4.0, segmentation reference only.
- [Void detection on X-ray](https://universe.roboflow.com/pcbdefectdetection-2qt8g/void-detection-on-x-ray-s8gso), CC BY 4.0, X-ray reference only.
