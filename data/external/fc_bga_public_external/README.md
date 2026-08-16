# Public External FC-BGA Evidence

This directory defines licensed public single-RGB evidence for a portfolio/internal PoC. It is separate from same-camera R/G/B/RING evidence under `data/vision/fc_bga_defects/`. Public images and checkpoints are never formal FC-BGA production evidence.

The fixed provisional annotation classes are `BALL_BRIDGE`, `MISSING_BALL`, `EXTRA_BALL`, `BALL_SIZE_ABNORMAL`, `BALL_OFFSET`, `BALL_SHAPE_ABNORMAL`, and `FOREIGN_MATERIAL`. Original public `NG` and `OK` labels are ignored. A reviewer may create a box only when the visible defect boundary and class are auditable at native resolution. Ambiguous images remain `review_required` or become `quarantined`; labels must not be guessed to satisfy a sample gate.

Runtime layout:

- `cache/`: permitted source downloads and license snapshots.
- `review/images/`: exact-deduplicated candidates.
- `review/candidates.jsonl`: provenance and manual review state.
- `review/labels/`: provisional human-reviewed YOLO labels.
- `versions/`: immutable generated train/val/test revisions.

Prepare a review queue from the pinned public smoke download:

```powershell
python tools/vision/fc_bga_yolo/prepare_public_external_candidates.py --source-root data/external/fc_bga_public_smoke/downloads/bga-ram-chips-detection-t3cqn-v1 --source-manifest data/external/fc_bga_public_smoke/downloads/bga-ram-chips-detection-t3cqn-v1/source-manifest.json --source-registry data/external/fc_bga_public_external/sources.json --destination data/external/fc_bga_public_external/review --source-id roboflow-paween-bga-ram-v1
```

After manual review, audit and publish only when the applicable gate passes:

```powershell
python tools/vision/fc_bga_yolo/public_external_manifest.py --manifest data/external/fc_bga_public_external/review/candidates.jsonl --sources data/external/fc_bga_public_external/sources.json --json-report .test-tmp/public-external-audit.json
python tools/vision/fc_bga_yolo/public_external_revision.py --manifest data/external/fc_bga_public_external/review/candidates.jsonl --sources data/external/fc_bga_public_external/sources.json --output data/external/fc_bga_public_external/versions --version public-external-v0.1 --stage B0 --seed 42
```

Every new accepted set creates a new version and regenerates all source-group-safe splits. License changes quarantine the affected cache and block new use. Deletion occurs only when recorded terms, a confirmed contractual obligation, or an approved takedown process requires it.

## Current Gate Status (2026-08-17)

The pinned Roboflow source produced 73 source-image entries. Exact SHA-256 deduplication retained 56 unique review candidates and removed 17 duplicates. The candidate audit reports zero structural errors, but all 56 images remain `review_required`: no seven-class boxes have completed native-resolution human review.

Consequently, Stage B0 is `blocked_data` with zero accepted images and zero represented classes. `public-external-v0.1` has not been published and no B0 checkpoint exists. The local evidence files are `.test-tmp/public-external-candidate-audit.json` and `.test-tmp/public-external-b0-coverage-shortfall.json`.

Stage B1 is also unavailable on this host by design. Its resource report is `skipped_resource` because PyTorch is CPU-only and no CUDA device is visible. Do not weaken annotations, map `NG/OK` into the seven-class contract, or copy public single-RGB images into the formal four-light dataset to bypass these gates.
