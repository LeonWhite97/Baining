# Public Alternative Data Sources for FC-BGA YOLO PoC

Last reviewed: 2026-08-15

This document lists public or semi-public datasets that can stand in for real FC-BGA capture data during script smoke tests, annotation workflow rehearsal, and portfolio/internal PoC experiments. These sources do not replace same-camera PIS-IN R/G/B/RING captures, reviewed seven-class labels, blind tests, TensorRT latency evidence, or production quality metrics.

## Use Boundary

- Allowed: downloader smoke tests, YOLO format rehearsal, mock four-light directory generation, label-tool rehearsal, demo screenshots, and non-production model experiments.
- Not allowed: formal FC-BGA accuracy claims, production false-positive or escape-rate claims, site P95 latency claims, automatic PASS release gates, or customer acceptance evidence.
- Keep public data isolated under `data/external/` or an external working directory. Do not merge public labels into `data/vision/fc_bga_defects` unless the file is explicitly marked smoke/reference and never used as the formal seven-class dataset.
- Review each source license and attribution requirement before redistribution. Some sources require account access, API keys, author approval, or research-only use.

## Recommended Order

| Priority | Source | Fit | Boundary |
| --- | --- | --- | --- |
| 1 | BGA RAM Chips Detection | Closest current public smoke source for BGA-like object detection and YOLO wiring. | Labels are not the formal seven FC-BGA defect classes. Use smoke only. |
| 2 | BGA-Balls | Useful reference for solder-ball/grid appearance and annotation practice. | Segmentation/reference data, not formal detector training data. |
| 3 | DeepPCB | Useful for generic PCB defect detection conversion and validation exercises. | Defect taxonomy is PCB trace defects, not BGA ball defects; research-purpose notice applies. |
| 4 | PCB Component Detection Consolidated Dataset | Useful for YOLO component-detection rehearsal and AOI UI demos. | Component classes are not defects and cannot train the FC-BGA defect contract. |
| 5 | FPIC/FICS-PCB | Useful for optical PCB assurance background, SMD/OCR annotation examples, and AOI portfolio context. | Access and license terms must be checked; not FC-BGA solder-ball defect data. |

## Source List

### BGA RAM Chips Detection

- URL: https://universe.roboflow.com/paween/bga-ram-chips-detection-t3cqn
- Current repository support: `tools/vision/fc_bga_yolo/download_public_smoke.py`
- Declared purpose here: `public_smoke`
- Notes: The repository pins workspace `paween`, project `bga-ram-chips-detection-t3cqn`, version `1`, format `yolov8`, and records `CC BY 4.0` in the local downloader manifest.

Download and smoke-train:

```powershell
$env:ROBOFLOW_API_KEY = "<set-in-your-shell>"
python tools/vision/fc_bga_yolo/download_public_smoke.py
python tools/vision/fc_bga_yolo/download_models.py --models yolov8n.pt
python tools/vision/fc_bga_yolo/train.py --config tools/vision/fc_bga_yolo/configs/train_smoke.yaml
```

### BGA-Balls

- URL: https://universe.roboflow.com/kenshin-blirtz/bga-balls-3ihxj
- Declared purpose here: `segmentation_reference`
- Notes: Use for solder-ball appearance study and annotation convention rehearsal. Do not merge it into the formal seven-class FC-BGA detector without a separate conversion design and license review.

Suggested manual handling:

```powershell
# Download from Roboflow Universe after reviewing license and selected export format.
# Keep it outside the formal dataset tree, for example:
New-Item -ItemType Directory -Force D:/aoi-data/public-reference/bga-balls
```

### DeepPCB

- URL: https://github.com/tangsanli5201/DeepPCB
- Declared purpose here: `pcb_defect_reference`
- Notes: DeepPCB contains aligned template/test image pairs and bounding boxes for six PCB trace-defect classes: open, short, mousebite, spur, pin hole, and spurious copper. The upstream README states the dataset is for research purpose.

Suggested manual handling:

```powershell
git clone https://github.com/tangsanli5201/DeepPCB.git D:/aoi-data/public-reference/DeepPCB
# Convert only in a separate smoke/reference workspace after mapping its annotation format.
```

### PCB Component Detection Consolidated Dataset

- URL: https://www.kaggle.com/datasets/aryanstein/pcb-component-detection-consolidated-dataset/data
- Declared purpose here: `pcb_component_yolo_reference`
- Notes: Kaggle describes this dataset as a YOLO-format consolidation of multiple PCB component datasets and lists Apache 2.0. It is useful for component-detection wiring, not FC-BGA defect training.

Suggested manual handling:

```powershell
# Requires Kaggle account/API setup outside this repository.
kaggle datasets download -d aryanstein/pcb-component-detection-consolidated-dataset -p D:/aoi-data/public-reference/pcb-components
```

### FPIC / FICS-PCB

- URLs:
  - https://physicaldb.ece.ufl.edu/index.php/fics-pcb-image-collection-fpic/
  - https://paperswithcode.com/dataset/fics-pcb-image-collection-fpic
- Declared purpose here: `pcb_aoi_context_reference`
- Notes: PhysicalDB describes FPIC as high-resolution optical PCB images with text, logo, and SMD annotations. Papers With Code lists the license as unknown, so treat it as reference-only until access and license terms are confirmed.

Suggested manual handling:

```powershell
# Request or download access through the dataset provider.
# Keep downloaded files outside Git, for example:
New-Item -ItemType Directory -Force D:/aoi-data/public-reference/fpic
```

## Mock Four-light Rehearsal

If a downloaded public dataset only provides ordinary RGB images, use the mock capture tool to rehearse the four-light file contract:

```powershell
python tools/vision/fc_bga_yolo/mock_capture_dataset.py D:/aoi-data/public-reference/images D:/aoi-data/fc_bga_mock_capture --prefix PUBLIC
python tools/vision/fc_bga_yolo/convert_dataset.py D:/aoi-data/fc_bga_mock_capture/source.jsonl data/vision/fc_bga_defects
python tools/vision/fc_bga_yolo/validate_yolo_dataset.py data/vision/fc_bga_defects --manifest data/vision/fc_bga_defects/manifest.jsonl --json-report D:/aoi-runs/public-mock-validation.json
```

The generated R/G/B/RING files are simulated copies. They prove data plumbing only and do not create multispectral or multi-light evidence.

## Promotion Gate

Before moving from public alternatives to real YOLO fine-tuning, collect or receive:

- 100-200 same-camera PIS-IN samples for the next data expansion pass.
- Real R/G/B/RING frame groups with stable trigger identity.
- Reviewed YOLO labels for the fixed seven FC-BGA classes.
- Separate train/val/test groups with no lot, tray, product, or sample leakage.
- Blind-test evidence before any production metric is claimed.
