# FC-BGA Official Pretrain and Public External Annotation Design

Date: 2026-08-16

## Objective

Use an official Ultralytics YOLOv8n pretrained checkpoint to replace the current scratch-only smoke experiment, and build a provenance-audited `public_external` annotation workflow for visible FC-BGA defects.

This work remains a portfolio/internal PoC. Public images and provisional annotations do not establish same-camera four-light performance, production precision, false-positive rate, escape rate, latency, or customer acceptance.

**Resource boundary:** Stage B1 is a CUDA-dependent metric experiment. When no compatible CUDA device and CUDA-enabled PyTorch build are available, Stage B1 is skipped by design and produces a resource assessment report instead of waiting or attempting an unbounded CPU run. Stage A and Stage B0 retain the measured CPU fallback.

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

Downloaded images remain ignored by default. A source-specific, exact-deduplicated review set may be committed only after the user explicitly authorizes redistribution and the repository includes source/version attribution, a verified license snapshot and hash, and per-image SHA-256 provenance. This exception applies to the Roboflow `paween/bga-ram-chips-detection-t3cqn` version 1 CC BY 4.0 review candidates authorized on 2026-08-17; it does not promote them to accepted seven-class or four-light evidence. Checkpoints and run outputs remain ignored by Git. API keys must be read only from environment variables and must never be printed, persisted, or committed.

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

Acquire candidates in reviewed batches. The first batch targets 30 to 50 images. If fewer than 20 pass review, continue only through other approved, license-verifiable sources up to a maximum of 100 candidates. If the accepted set still has fewer than 20 images or fewer than two defect classes with visible evidence, publish a coverage-shortfall report and keep Stage B0 blocked. Do not weaken annotation rules to satisfy the count.

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

## Dataset Versioning and Split Regeneration

Every accepted-set change creates a new immutable dataset revision. Stage B0 begins with `public-external-v0.1`. Reaching the Stage B1 gate creates `public-external-v0.2`; later accepted additions increment the minor version again.

Each revision records the dataset version, complete accepted-sample manifest SHA-256, `split_seed=42`, `split_algorithm=group-stratified-v1`, source-group assignments, class counts, and creation time. Splitting operates on `source_group_id`, not individual images, and performs deterministic group-aware stratification over the full accepted set for that revision.

When `public-external-v0.2` is created, regenerate train, validation, and test assignments from all accepted images. The `v0.1` assignments become inactive for B1 but remain archived and hash-addressable with the B0 run. B0 and B1 metrics are not presented as a direct longitudinal improvement because their evaluation sets differ. A model-to-model comparison requires both checkpoints to be evaluated against a separately frozen common benchmark revision.

## Training Design

### Hardware Preflight

Before each training stage, record the PyTorch build, CUDA availability, visible device count, selected device, available GPU memory when applicable, CPU model, and calibration-run duration.

Use CUDA device `0` when `torch.cuda.is_available()` is true and the installed PyTorch/Ultralytics stack can complete a one-batch train and validation probe. Stage A and Stage B0 use CPU as an explicit fallback; Stage B1 does not. Keep `workers=0` as the reliable Windows baseline; a higher worker count may be selected only after a measured loader probe completes without process-spawn or DLL errors.

Run a three-epoch calibration with the target image size and batch before a longer CPU job. Estimate the full run from measured wall-clock time. If the projected local run exceeds two hours, stop after calibration and emit the verified GPU command instead of automatically starting the long run. When the same configuration continues locally, resume from the calibration checkpoint so those three epochs are not repeated.

The current host measurement is contextual evidence, not a permanent estimate: the CPU-only environment completed 20 epochs over 39 images at image size 320 in 330.451 seconds, or 16.52 seconds per epoch. Every new dataset and image size must be calibrated independently.

### Stage A: Official Pretrain Smoke Training

Download official `yolov8n.pt` through the existing model downloader. Require a file size of at least 1 MiB and record the official release URL, content length, retrieval date, and SHA-256 before use. That record becomes the local verification baseline for subsequent downloads.

Train the existing `public_smoke` dataset with:

- image size: 640;
- epochs: 30;
- patience: 10;
- batch: 4;
- device: hardware preflight selection, preferring CUDA device `0` and falling back to CPU;
- workers: 0 on the current Windows environment;
- deterministic seed: 42.

This checkpoint remains `NG`/`OK` public smoke evidence and is not deployable.

### Stage B0: Seven-Class Workflow Rehearsal

After at least 20 images pass review, at least two classes have accepted evidence, and all quality gates pass, train from official `yolov8n.pt` with:

- image size: 640;
- epochs: 10, including the first three calibration epochs;
- patience: 5;
- batch: 4;
- device: hardware preflight selection, preferring CUDA device `0` and falling back to CPU within the two-hour projected budget;
- workers: 0;
- deterministic seed: 42.

Stage B0 verifies only download, provenance, annotation, split, training, and evaluation wiring. Its report must begin with the exact warning: **INSUFFICIENT STATISTICAL EVIDENCE: workflow rehearsal metrics are not model-performance evidence.** Report mAP values as diagnostic output only. The checkpoint remains non-deployable.

### Stage B1: Seven-Class Metric Experiment

The 50-epoch public external experiment remains blocked until all of these additional gates pass:

- at least 100 accepted images;
- at least three represented defect classes;
- at least 30 training boxes for every represented class included in metric interpretation;
- at least 10 test boxes for every represented class included in metric interpretation;
- no source group, original image, derived crop, or adjacent-frame leakage across splits.

When the gates pass and the CUDA probe succeeds, train from official `yolov8n.pt` at image size 640, 50 epochs, patience 10, batch 4, workers 0, deterministic seed 42, and CUDA device `0`. CPU execution is not an automatic B1 fallback. If the CUDA probe fails, skip B1 and publish the resource assessment report as expected behavior.

Always report image and box denominators beside overall and per-class precision, recall, mAP50, and mAP50-95. If the test split contains at least 30 independent source groups, add a 95% grouped-bootstrap interval using 1,000 resamples. The resampling unit must be `source_group_id`, never `image_id`: a sampled group contributes all of its test images as one block, and an unsampled group contributes none. With fewer than 30 test source groups, omit the interval and retain the insufficient-statistical-evidence warning. Public-data metrics remain non-production evidence and the checkpoint remains non-deployable.

### Empty-Class Metric Policy

Ultralytics `8.4.120` computes detection AP over `np.unique(target_cls)`, so its native aggregate already excludes configured classes that have no ground-truth instances. Do not patch third-party `val.py`. Add a repository-owned evaluation wrapper that uses `nt_per_class` and `ap_class_index` to emit both the unchanged native aggregate and an explicit `observed_class_mAP` over classes with `total_gt > 0`. Report every configured empty class with metric value `null` and status `no_evidence`.

The report footnote must state: **mAP is computed over classes with nonzero ground-truth instances only.** A version-pinned regression test must use seven configured classes with GT in only two classes and prove that the five empty classes are excluded from `observed_class_mAP`. If a later Ultralytics release changes its native behavior, the wrapper remains the reporting authority and the native/wrapper difference is recorded.

### Stage C: Future Formal Four-Light Training

Formal training remains gated on 100 to 200 same-camera sample groups, reviewed four-light labels, group-safe splits, and blind-test evidence. The existing PoC target remains 1280 pixels, 100 epochs, patience 20, and GPU execution with the selected official starting checkpoint. Formal parameters may be revised only after capture-resolution and GPU-memory measurements exist.

## Failure Handling

- Prefer a verified local cache before network retrieval. Retry an official network source at most three times.
- Do not silently switch to an untrusted mirror.
- Treat a checkpoint smaller than 1 MiB, missing its recorded official-asset baseline, or differing from the recorded content length or SHA-256 as a hard failure.
- Store permitted offline artifacts outside Git with an artifact manifest containing source URL, source version, retrieval date, license snapshot or license URL, content length, and SHA-256. A cached artifact is usable only when its hash matches the manifest and its recorded license permits the retained copy.
- When an upstream source disappears, use a matching verified permitted cache. If no such cache exists, mark that source unavailable and select another already-approved source as a new dataset revision; never present the replacement as the missing version.
- Recheck the upstream license before publishing a new dataset revision or starting a new training run. Record the retrieved license text or file hash so the acquisition-time terms remain auditable.
- If current upstream terms become more restrictive, a takedown request appears, or provenance is disputed, immediately quarantine the affected cache and block new use pending license review. Do not automatically delete an artifact solely because a web page changed: delete it only when the recorded terms, a confirmed contractual obligation, or an approved takedown process requires deletion, and record the deletion event in the artifact manifest.
- Quarantine sources with unknown licenses or unavailable attribution details.
- Quarantine ambiguous labels instead of guessing.
- Stop dataset publication when duplicate, split-leakage, class-order, coordinate, or provenance checks fail.
- Do not start Stage B0 when fewer than 20 images pass review or fewer than two classes have accepted evidence.
- Do not start Stage B1 until its image, class, train-box, and test-box gates all pass.
- Treat a missing compatible CUDA environment as an expected B1 skip with a resource report, not a retry loop or pipeline failure.
- Preserve failed-run logs separately and never overwrite a validated run.

## Outputs and Acceptance

Stage A is accepted when the official checkpoint is verified, 30 training epochs complete or valid early stopping occurs, the test split runs, and the run contains parameters, curves, confusion matrices, `best.pt`, and per-class metrics.

Stage B0 is accepted when provenance and quality gates pass, all three splits are nonempty and leakage-free, calibration and workflow rehearsal complete within the resource gate, test evaluation completes, the warning is present, and the same artifact set is available. Acceptance proves only a reproducible public-data workflow.

Stage B1 has two valid terminal statuses. `executed` requires its stronger sample gates, a successful CUDA probe, completed training and test evaluation, and the required variability or confidence reporting. `skipped_resource` requires a failed CUDA probe and a completed resource assessment report. Neither status proves suitability for production.

No public checkpoint may generate formal deployable metadata, activate automatic PASS decisions, or support production performance claims.

## Verification

Implementation verification must include:

- focused tests for the new profile and its non-deployable metadata gate;
- provenance-schema and quarantine tests;
- class-order and annotation validation tests;
- exact-duplicate and source-group leakage tests;
- dataset-version, full-set split-regeneration, deterministic-seed, and archived-assignment tests;
- model download verification;
- public smoke dataset validation;
- public external preflight and dataset report;
- Stage A test evaluation;
- Stage B0 resource calibration and test evaluation when its workflow gate is satisfied;
- Stage B1 sample-gate audit and test evaluation only when its stronger metric gate is satisfied;
- a seven-class/two-GT-class regression test for empty-class `null` reporting and `observed_class_mAP`;
- a grouped-bootstrap test proving that all images from a sampled `source_group_id` move as one block;
- a CUDA-unavailable test proving Stage B1 exits as an expected resource skip;
- a license-change test proving the cache is quarantined before any deletion decision;
- a repository secret scan and clean Git status before push.

## Repository Delivery

Commit implementation, tests, configs, provenance documentation, and non-sensitive reports to `main` after verification. Keep downloaded images, API keys, checkpoints, virtual environments, and training-run binaries outside Git.
