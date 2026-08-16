# FC-BGA Official Pretrain and Public External Annotation Design

Date: 2026-08-16

## Objective

Use an official Ultralytics YOLOv8n pretrained checkpoint to replace the current scratch-only smoke experiment, and build a provenance-audited `public_external` annotation workflow for visible FC-BGA defects.

This work remains a portfolio/internal PoC. Public images and provisional annotations do not establish same-camera four-light performance, production precision, false-positive rate, escape rate, latency, or customer acceptance.

## Factual Boundaries

The repository has three separate evidence levels:

1. `public_smoke`: the existing public BGA RAM dataset with `NG` and `OK` classes. It proves download, training, validation, and checkpoint wiring only.
2. `public_external`: licensed public BGA images with provisional human-reviewed labels using the formal seven class names. It supports annotation rehearsal and non-production transfer-learning experiments only.
3. `fc_bga`: future same-camera PIS-IN R/G/B/RING captures, reviewed labels, group-safe train/validation/test splits, and blind-test evidence. Only this level may produce deployable model metadata after all formal gates pass.

Public single-RGB images must not be copied into the formal four-light dataset or represented as `rgb_grayscale_stack_v1` evidence. Simulated R/G/B/RING copies remain plumbing tests and cannot promote a dataset or model to the formal evidence level.

## Selected Approach

Use two isolated training tracks:

- Retrain the existing `NG`/`OK` public smoke dataset from official `yolov8n.pt` to verify transfer learning.
- Build and train a separate seven-class `public_external` dataset from licensed public images after provenance and annotation review.

Do not map `NG` to a formal defect class. Do not synthesize or guess labels to fill sparse classes. Synthetic data may be evaluated later in a separately named experiment, but it is outside this design.

## Data Layout and Provenance

Store public external artifacts under `data/external/fc_bga_public_external/`. Do not write them to `data/vision/fc_bga_defects/`.

The public external area must distinguish:

- source metadata and reproducible download instructions;
- downloaded raw images;
- accepted annotations;
- quarantined candidates and rejection reasons;
- generated train, validation, and test views.

Each candidate image must have a provenance record containing:

- stable sample identifier;
- source name and source URL;
- dataset version when available;
- declared license and attribution requirement;
- retrieval date;
- original file name;
- SHA-256 digest;
- source group identifier;
- review status;
- accepted defect classes;
- rejection or quarantine reason when not accepted.

Images with unknown reuse terms, unreadable defects, unresolved duplicates, conflicting labels, or insufficient context remain quarantined and never enter training.

Downloaded images, checkpoints, and run outputs remain ignored by Git. The repository may commit source metadata, licenses, annotation records, configuration, reports, and reproducible commands when their upstream terms permit it. API keys must be read only from environment variables and must never be printed, persisted, or committed.

## Annotation Contract

The public external dataset keeps the fixed formal class order:

1. `BALL_BRIDGE`
2. `MISSING_BALL`
3. `EXTRA_BALL`
4. `BALL_SIZE_ABNORMAL`
5. `BALL_OFFSET`
6. `BALL_SHAPE_ABNORMAL`
7. `FOREIGN_MATERIAL`

Annotation decisions use these rules:

- `BALL_BRIDGE`: two or more solder balls have a clearly visible continuous connection.
- `MISSING_BALL`: a missing position is auditable from an expected grid or equivalent product reference. A local dark area without a grid reference is not sufficient.
- `EXTRA_BALL`: a clearly visible additional solder ball lies outside the expected array pattern.
- `BALL_SIZE_ABNORMAL`: the same image provides an array-scale reference and the size difference is visually unambiguous.
- `BALL_OFFSET`: the ball center is clearly displaced relative to the expected array position.
- `BALL_SHAPE_ABNORMAL`: the visible contour is clearly non-circular, collapsed, or otherwise deformed.
- `FOREIGN_MATERIAL`: a visible attached object or contaminant is clearly not part of the solder-ball structure.

Only visible, spatially bounded evidence receives a YOLO box. Ambiguous boundaries, conflicting classes, inference based only on process knowledge, and defects outside visible-light scope receive `review_required` and stay out of training.

The first public external labels are `provisional_human_reviewed_poc`, not production expert truth. Classes with no accepted examples remain present in the class contract and are reported as having no evaluation evidence. They are not filled with synthetic or speculative examples.

`NORMAL` remains an empty label, not a model class. `UNKNOWN`, `MISSING_3D`, and `MISSING_LIGHT` remain decision or input states. Coplanarity remains a 3D rule, and solder void, head-in-pillow, and open defects remain outside visible-light scope.

## Input Profiles

Add an explicit `public_external` training profile for ordinary public RGB images. It uses the seven class names but cannot write deployable `model_metadata.json` and cannot be selected by the production inference adapter.

The existing formal `fc_bga` profile remains unchanged. Its `rgb_grayscale_stack_v1` contract converts same-camera R/G/B frames to grayscale channels in R/G/B order and retains RING as required provenance evidence.

This separation prevents a single-RGB public checkpoint from being mistaken for a validated four-light checkpoint.

## Source Selection

Only sources with clear redistribution or training permission and a stable version are eligible for download. The initial preferred sources are public BGA datasets with explicit licenses, including the documented Roboflow BGA sources when their current version and license remain verifiable.

Manufacturer case studies, research-paper figures, search-engine thumbnails, and images with unclear terms may inform visual understanding but do not enter the training dataset without explicit reuse permission.

The first acquisition pass targets 30 to 50 candidate images. Seven-class public external training starts only after at least 20 images pass review and at least two defect classes have accepted visible evidence.

## Quality Control and Splitting

Every accepted image must pass:

- supported format and successful decoding;
- width and height of at least 256 pixels, followed by visual confirmation that the proposed defect boundary remains distinguishable at native resolution;
- SHA-256 exact-duplicate detection;
- annotation syntax and normalized-coordinate validation;
- class identifier and fixed class-order validation;
- source and license completeness;
- review-state validation.

Perceptual or crop-related duplicates must be reviewed as a group. The same original image, derived crop, adjacent frame, or source product group may appear in only one split.

Train, validation, and test splits must all be nonempty. Exact duplicates, group leakage, unsupported files, invalid boxes, unlicensed accepted samples, and unreviewed accepted labels must each have a count of zero before training.

## Training Design

### Stage A: Official Pretrain Smoke Training

Download official `yolov8n.pt` through the existing model downloader. Require a file size of at least 1 MiB and record the official release URL, content length, retrieval date, and SHA-256 before use. That record becomes the local verification baseline for subsequent downloads.

Train the existing `public_smoke` dataset with:

- image size: 640;
- epochs: 30;
- patience: 10;
- batch: 4;
- device: CPU on the current host;
- workers: 0 on the current Windows environment;
- deterministic seed: 42.

This checkpoint remains `NG`/`OK` public smoke evidence and is not deployable.

### Stage B: Seven-Class Public External Training

After the public external data gate passes, train from official `yolov8n.pt` with:

- image size: 640;
- epochs: 50;
- patience: 10;
- batch: 4;
- device: CPU on the current host;
- workers: 0;
- deterministic seed: 42.

Report overall and per-class precision, recall, mAP50, and mAP50-95. A class with no test instances is reported as having no evidence, not as a measured zero or success. The checkpoint remains non-deployable regardless of its public-data metrics.

### Stage C: Future Formal Four-Light Training

Formal training remains gated on 100 to 200 same-camera sample groups, reviewed four-light labels, group-safe splits, and blind-test evidence. The existing PoC target remains 1280 pixels, 100 epochs, patience 20, and GPU execution with the selected official starting checkpoint. Formal parameters may be revised only after capture-resolution and GPU-memory measurements exist.

## Failure Handling

- Retry official downloads at most three times.
- Do not silently switch to an untrusted mirror.
- Treat a checkpoint smaller than 1 MiB, missing its recorded official-asset baseline, or differing from the recorded content length or SHA-256 as a hard failure.
- Quarantine sources with unknown licenses or unavailable attribution details.
- Quarantine ambiguous labels instead of guessing.
- Stop dataset publication when duplicate, split-leakage, class-order, coordinate, or provenance checks fail.
- Do not start Stage B when fewer than 20 images pass review or fewer than two classes have accepted evidence.
- Preserve failed-run logs separately and never overwrite a validated run.

## Outputs and Acceptance

Stage A is accepted when the official checkpoint is verified, 30 training epochs complete or valid early stopping occurs, the test split runs, and the run contains parameters, curves, confusion matrices, `best.pt`, and per-class metrics.

Stage B is accepted when provenance and quality gates pass, all three splits are nonempty and leakage-free, training and test evaluation complete, and the same artifact set is present. Acceptance proves a reproducible public-data PoC workflow, not model suitability for production.

No public checkpoint may generate formal deployable metadata, activate automatic PASS decisions, or support production performance claims.

## Verification

Implementation verification must include:

- focused tests for the new profile and its non-deployable metadata gate;
- provenance-schema and quarantine tests;
- class-order and annotation validation tests;
- exact-duplicate and source-group leakage tests;
- model download verification;
- public smoke dataset validation;
- public external preflight and dataset report;
- Stage A test evaluation;
- Stage B test evaluation when its data gate is satisfied;
- a repository secret scan and clean Git status before push.

## Repository Delivery

Commit implementation, tests, configs, provenance documentation, and non-sensitive reports to `main` after verification. Keep downloaded images, API keys, checkpoints, virtual environments, and training-run binaries outside Git.
