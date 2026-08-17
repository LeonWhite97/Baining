# FC-BGA Public External Stage B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a provenance-audited, versioned public FC-BGA annotation workflow and run only the Stage B training level allowed by data, license, and CUDA gates.

**Architecture:** Keep public single-RGB evidence in a dedicated `public_external` contract, separate from formal R/G/B/RING data. Publish immutable dataset revisions through deterministic source-group splitting, add a repository-owned metric wrapper and source-group bootstrap collector, and reuse Plan 1's resource-gated stage runner.

**Tech Stack:** Python 3.13, Pillow, PyYAML, NumPy through Ultralytics, Ultralytics 8.4.120, PyTorch, pytest, JSON/JSONL, PowerShell on Windows.

## Global Constraints

- Plan 1 (`2026-08-16-fc-bga-stage-a-official-pretrain.md`) must be implemented first.
- `public_external` images are licensed public single-RGB evidence, not same-camera four-light evidence.
- Keep `data/external/fc_bga_public_external/` separate from `data/vision/fc_bga_defects/`.
- Fixed class order is `BALL_BRIDGE`, `MISSING_BALL`, `EXTRA_BALL`, `BALL_SIZE_ABNORMAL`, `BALL_OFFSET`, `BALL_SHAPE_ABNORMAL`, `FOREIGN_MATERIAL`.
- Never map `NG` to a formal defect class and never synthesize or guess a label to fill a sparse class.
- Public external labels are `provisional_human_reviewed_poc`; ambiguous evidence is `review_required` and excluded.
- B0 requires at least 20 accepted images and at least two represented defect classes; it is a 10-epoch workflow rehearsal.
- B1 requires at least 100 accepted images, three represented classes, 30 training boxes and 10 test boxes per interpreted class, and a successful CUDA probe; it is skipped as `skipped_resource` without CUDA.
- B0 reports must start with `INSUFFICIENT STATISTICAL EVIDENCE: workflow rehearsal metrics are not model-performance evidence.`
- Empty test classes are `null/no_evidence`; reported mAP is over classes with nonzero GT only.
- Bootstrap uses `source_group_id` blocks, never individual images, with 1,000 resamples only when the test split has at least 30 independent source groups.
- Any accepted-set change creates a new immutable revision; `public-external-v0.2` regenerates all splits from the full accepted set with `split_seed=42` and `split_algorithm=group-stratified-v1`, and does not overwrite `public-external-v0.1`.
- B0 and B1 metrics are not presented as a direct longitudinal comparison because their evaluation splits differ.
- License changes quarantine cached data before any deletion decision.
- Public checkpoints never emit deployable `model_metadata.json` and never activate automatic PASS decisions.

---

## File Map

- Create `data/external/fc_bga_public_external/README.md`: data boundary, annotation policy, and commands.
- Create `data/external/fc_bga_public_external/sources.json`: approved source registry starting with the pinned BGA RAM v1 source.
- Create `tools/vision/fc_bga_yolo/public_external_manifest.py`: candidate schema, provenance, license state, and audit.
- Create `tools/vision/fc_bga_yolo/prepare_public_external_candidates.py`: deduplicate licensed source images and build a review queue.
- Create `tools/vision/fc_bga_yolo/public_external_revision.py`: gate checks, deterministic full-set splitting, immutable publication, and YAML generation.
- Create `tools/vision/fc_bga_yolo/public_external_evaluation.py`: empty-class reporting, validation-stat collection, and grouped bootstrap.
- Create `tools/vision/fc_bga_yolo/configs/public_external.template.yaml`: seven-class single-RGB dataset template.
- Create `tools/vision/fc_bga_yolo/configs/train_public_external_b0.yaml`: 10-epoch workflow rehearsal.
- Create `tools/vision/fc_bga_yolo/configs/train_public_external_b1.yaml`: 50-epoch CUDA-only metric experiment.
- Create tests `test_public_external_manifest.py`, `test_public_external_candidates.py`, `test_public_external_revision.py`, and `test_public_external_evaluation.py`.
- Modify `.gitignore`, `tools/vision/fc_bga_yolo/train.py`, `run_training_stage.py`, `README.md`, `tests/test_training_commands.py`, and `tests/test_training_stage.py`.

---

### Task 1: Public External Provenance and License Contract

**Files:**
- Create: `tools/vision/fc_bga_yolo/public_external_manifest.py`
- Create: `tools/vision/fc_bga_yolo/tests/test_public_external_manifest.py`
- Create: `data/external/fc_bga_public_external/README.md`
- Create: `data/external/fc_bga_public_external/sources.json`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `CandidateRecord`, `SourceRecord`, `CandidateAudit`, `load_source_registry()`, `load_candidate_manifest()`, `audit_candidates()`, and `assess_license_state()`.
- Consumes later: candidate preparation, revision publication, and training preflight.

- [ ] **Step 1: Write failing strict-schema and license-change tests**

```python
def _accepted_candidate(tmp_path: Path) -> tuple[SourceRecord, CandidateRecord]:
    image = tmp_path / "images" / "sample-001.png"
    label = tmp_path / "labels" / "sample-001.txt"
    image.parent.mkdir()
    label.parent.mkdir()
    Image.new("RGB", (256, 256), "white").save(image)
    label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="ascii")
    source = SourceRecord(
        source_id="source-001",
        name="test source",
        url="https://example.test/dataset",
        version="1",
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/legalcode.txt",
        license_sha256="a" * 64,
        retrieved_at="2026-08-16T00:00:00Z",
        attribution="test source, CC BY 4.0",
    )
    record = CandidateRecord(
        sample_id="sample-001",
        source_group_id="group-001",
        source_id=source.source_id,
        original_filename="sample-001.png",
        image_path="images/sample-001.png",
        image_sha256=sha256_file(image),
        label_path="labels/sample-001.txt",
        review_status="accepted",
        annotation_status="provisional_human_reviewed_poc",
        accepted_classes=("BALL_BRIDGE",),
        quarantine_reason=None,
    )
    return source, record

def test_accepted_candidate_requires_reviewed_label_and_matching_hash(tmp_path: Path) -> None:
    source, record = _accepted_candidate(tmp_path)
    audit = audit_candidates((record,), sources=(source,), manifest_root=tmp_path)
    assert audit.errors == ()
    assert audit.accepted_images == 1

def test_license_change_quarantines_without_deleting(tmp_path: Path) -> None:
    cached = tmp_path / "image.png"
    Image.new("RGB", (256, 256), "white").save(cached)
    state = assess_license_state(recorded_sha="a" * 64, current_sha="b" * 64)
    assert state == "quarantined_license_change"
    assert cached.is_file()

@pytest.mark.parametrize(("changes", "error"), (
    ({"image_path": "../escape.png"}, "PATH_ESCAPE"),
    ({"image_sha256": "0" * 64}, "IMAGE_HASH_MISMATCH"),
    ({"label_path": None}, "ACCEPTED_LABEL_REQUIRED"),
    ({"review_status": "unknown"}, "REVIEW_STATUS_INVALID"),
    ({"accepted_classes": ("NOT_A_DEFECT",)}, "CLASS_INVALID"),
))
def test_candidate_schema_rejects_invalid_records(
    tmp_path: Path, changes: dict[str, object], error: str,
) -> None:
    source, record = _accepted_candidate(tmp_path)
    invalid = replace(record, **changes)
    audit = audit_candidates((invalid,), sources=(source,), manifest_root=tmp_path)
    assert any(error in item for item in audit.errors)

def test_candidate_rejects_small_image_and_missing_license_hash(tmp_path: Path) -> None:
    source, record = _accepted_candidate(tmp_path)
    image = tmp_path / record.image_path
    Image.new("RGB", (128, 256), "white").save(image)
    small = replace(record, image_sha256=sha256_file(image))
    assert any("IMAGE_TOO_SMALL" in item for item in audit_candidates(
        (small,), sources=(source,), manifest_root=tmp_path,
    ).errors)
    unlicensed = replace(source, license_sha256="")
    assert any("LICENSE_SNAPSHOT_REQUIRED" in item for item in audit_candidates(
        (record,), sources=(unlicensed,), manifest_root=tmp_path,
    ).errors)
```

- [ ] **Step 2: Run the focused tests and confirm red**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_public_external_manifest.py -q --basetemp .test-tmp/public-manifest-red`

Expected: FAIL because `public_external_manifest.py` does not exist.

- [ ] **Step 3: Implement exact records and audit output**

```python
ReviewStatus = Literal["review_required", "accepted", "quarantined"]

@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    name: str
    url: str
    version: str
    license_name: str
    license_url: str
    license_sha256: str
    retrieved_at: str
    attribution: str

@dataclass(frozen=True, slots=True)
class CandidateRecord:
    sample_id: str
    source_group_id: str
    source_id: str
    original_filename: str
    image_path: str
    image_sha256: str
    label_path: str | None
    review_status: ReviewStatus
    annotation_status: Literal["provisional_human_reviewed_poc"] | None
    accepted_classes: tuple[str, ...]
    quarantine_reason: str | None

@dataclass(frozen=True, slots=True)
class CandidateAudit:
    total_images: int
    accepted_images: int
    represented_classes: tuple[str, ...]
    class_boxes: Mapping[str, int]
    quarantine_counts: Mapping[str, int]
    errors: tuple[str, ...]
```

Implement these exact entry points: `load_source_registry(path: Path) -> tuple[SourceRecord, ...]`, `load_candidate_manifest(path: Path, sources: tuple[SourceRecord, ...]) -> tuple[CandidateRecord, ...]`, `audit_candidates(records: tuple[CandidateRecord, ...], *, sources: tuple[SourceRecord, ...], manifest_root: Path) -> CandidateAudit`, and `assess_license_state(*, recorded_sha: str, current_sha: str) -> Literal["verified", "quarantined_license_change"]`. Resolve all candidate paths inside the manifest directory. Verify image decoding, minimum dimensions, file hashes, strict YOLO rows, class order, accepted-class consistency, and source/license linkage. Accepted records require `annotation_status="provisional_human_reviewed_poc"`; other records require it to be null. `assess_license_state()` never deletes files.

- [ ] **Step 4: Create the approved initial source registry**

Add source ID `roboflow-paween-bga-ram-v1` with URL `https://universe.roboflow.com/paween/bga-ram-chips-detection-t3cqn`, version `1`, and license `CC BY 4.0`. The README states that its original `NG`/`OK` labels are ignored for seven-class annotation and that only visually auditable defects may receive new provisional boxes.

Update `.gitignore`:

```gitignore
data/external/fc_bga_public_external/cache/
data/external/fc_bga_public_external/versions/
data/external/fc_bga_public_external/review/images/
!data/external/fc_bga_public_external/README.md
!data/external/fc_bga_public_external/sources.json
```

- [ ] **Step 5: Run manifest tests and confirm green**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_public_external_manifest.py -q --basetemp .test-tmp/public-manifest-green`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

```powershell
git add .gitignore data/external/fc_bga_public_external/README.md data/external/fc_bga_public_external/sources.json tools/vision/fc_bga_yolo/public_external_manifest.py tools/vision/fc_bga_yolo/tests/test_public_external_manifest.py
git commit -m "feat: define public FC-BGA provenance contract"
```

---

### Task 2: Candidate Preparation and Review Queue

**Files:**
- Create: `tools/vision/fc_bga_yolo/prepare_public_external_candidates.py`
- Create: `tools/vision/fc_bga_yolo/tests/test_public_external_candidates.py`
- Modify: `data/external/fc_bga_public_external/README.md`

**Interfaces:**
- Produces: `prepare_candidates()` and CLI arguments `--source-root`, `--source-manifest`, `--source-registry`, `--destination`, and `--source-id`.
- Consumes: `SourceRecord`, file hashes from the pinned public smoke source, and the strict candidate JSONL format.

- [ ] **Step 1: Write failing duplicate and review-queue tests**

```python
def _public_source(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "source"
    paths = (
        root / "train/images/a.png",
        root / "val/images/a-copy.png",
        root / "test/images/b.png",
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), "white").save(paths[0])
    shutil.copy2(paths[0], paths[1])
    Image.new("RGB", (256, 256), "black").save(paths[2])
    manifest = root / "source-manifest.json"
    manifest.write_text(json.dumps({
        "accessed_on": "2026-08-16",
        "format": "yolov8",
        "license": "CC BY 4.0",
        "project": "test-project",
        "purpose": "public_smoke",
        "source_url": "https://example.test/dataset",
        "version": 1,
        "workspace": "test",
        "files": {path.relative_to(root).as_posix(): sha256_file(path) for path in paths},
    }, sort_keys=True), encoding="utf-8")
    registry = tmp_path / "sources.json"
    registry.write_text(json.dumps([{
        "source_id": SOURCE_ID,
        "name": "test source",
        "url": "https://example.test/dataset",
        "version": "1",
        "license_name": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode.txt",
        "license_sha256": "a" * 64,
        "retrieved_at": "2026-08-16T00:00:00Z",
        "attribution": "test source, CC BY 4.0",
    }]), encoding="utf-8")
    return root, manifest, registry

def test_prepare_candidates_deduplicates_by_sha_and_marks_review_required(tmp_path: Path) -> None:
    root, manifest, registry = _public_source(tmp_path)
    report = prepare_candidates(
        root, manifest, registry, tmp_path / "review", source_id=SOURCE_ID,
    )
    records = load_candidate_manifest(report.manifest, load_source_registry(registry))
    assert report.unique_images == 2
    assert report.exact_duplicates == 1
    assert {record.review_status for record in records} == {"review_required"}
    assert all(record.label_path is None for record in records)

def test_prepare_candidates_fails_closed_on_source_drift(tmp_path: Path) -> None:
    root, manifest, registry = _public_source(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    first_image = next(name for name in document["files"] if "/images/" in name)
    document["files"][first_image] = "0" * 64
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="SOURCE_HASH_MISMATCH"):
        prepare_candidates(root, manifest, registry, tmp_path / "review", source_id=SOURCE_ID)

def test_prepare_candidates_rejects_unregistered_source_and_extension(tmp_path: Path) -> None:
    root, manifest, registry = _public_source(tmp_path)
    with pytest.raises(ValueError, match="SOURCE_UNREGISTERED"):
        prepare_candidates(root, manifest, registry, tmp_path / "review-a", source_id="missing")
    unsupported = root / "train/images/unsupported.gif"
    Image.new("RGB", (256, 256), "red").save(unsupported)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["files"][unsupported.relative_to(root).as_posix()] = sha256_file(unsupported)
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="UNSUPPORTED_EXTENSION"):
        prepare_candidates(root, manifest, registry, tmp_path / "review-b", source_id=SOURCE_ID)

def test_prepare_candidates_refuses_changed_output_reuse(tmp_path: Path) -> None:
    root, manifest, registry = _public_source(tmp_path)
    destination = tmp_path / "review"
    prepare_candidates(root, manifest, registry, destination, source_id=SOURCE_ID)
    next((destination / "images").iterdir()).write_bytes(b"changed")
    with pytest.raises(ValueError, match="OUTPUT_CONTENT_CONFLICT"):
        prepare_candidates(root, manifest, registry, destination, source_id=SOURCE_ID)
```

- [ ] **Step 2: Run the focused tests and confirm red**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_public_external_candidates.py -q --basetemp .test-tmp/candidates-red`

Expected: FAIL because the preparation script does not exist.

- [ ] **Step 3: Implement deterministic candidate preparation**

```python
@dataclass(frozen=True, slots=True)
class CandidatePreparationReport:
    source_images: int
    unique_images: int
    exact_duplicates: int
    manifest: Path
    review_root: Path
```

Implement the exact signature `prepare_candidates(source_root: Path, source_manifest: Path, source_registry: Path, destination: Path, *, source_id: str) -> CandidatePreparationReport`. Load `source_registry`, require `source_id` to exist, and require its URL/version/license to match `source-manifest.json`. Sort inputs by normalized relative path, verify each source hash against `source-manifest.json`, deduplicate by SHA-256, copy unique images to `review/images/<sample_id>.<ext>`, and write `candidates.jsonl` atomically. Set `source_group_id` to the original source image identity so derived copies can never cross splits.

- [ ] **Step 4: Run candidate tests and confirm green**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_public_external_candidates.py -q --basetemp .test-tmp/candidates-green`

Expected: all tests pass.

- [ ] **Step 5: Prepare the real review queue from the pinned downloaded source**

```powershell
python tools\vision\fc_bga_yolo\prepare_public_external_candidates.py --source-root data\external\fc_bga_public_smoke\downloads\bga-ram-chips-detection-t3cqn-v1 --source-manifest data\external\fc_bga_public_smoke\downloads\bga-ram-chips-detection-t3cqn-v1\source-manifest.json --source-registry data\external\fc_bga_public_external\sources.json --destination data\external\fc_bga_public_external\review --source-id roboflow-paween-bga-ram-v1
```

Expected: test-derived duplicates are removed and every unique image starts as `review_required`. Do not carry over `NG`/`OK` class IDs.

- [ ] **Step 6: Commit Task 2**

```powershell
git add tools/vision/fc_bga_yolo/prepare_public_external_candidates.py tools/vision/fc_bga_yolo/tests/test_public_external_candidates.py data/external/fc_bga_public_external/README.md
git commit -m "feat: prepare public FC-BGA review candidates"
```

---

### Task 3: Immutable Revisions and Full-Set Group Splitting

**Files:**
- Create: `tools/vision/fc_bga_yolo/public_external_revision.py`
- Create: `tools/vision/fc_bga_yolo/tests/test_public_external_revision.py`
- Create: `tools/vision/fc_bga_yolo/configs/public_external.template.yaml`

**Interfaces:**
- Produces: `RevisionGate`, `PublishedRevision`, `evaluate_revision_gate()`, `assign_group_stratified_v1()`, and `publish_revision()`.
- Consumes later: public external training preflight and B0/B1 runner.

- [ ] **Step 1: Write failing B0/B1 gate and immutability tests**

```python
def _accepted_records(tmp_path: Path, count: int, classes: tuple[int, ...]) -> tuple[CandidateRecord, ...]:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir(exist_ok=True)
    labels.mkdir(exist_ok=True)
    result = []
    for index in range(count):
        sample_id = f"sample-{index:03d}"
        image = images / f"{sample_id}.png"
        label = labels / f"{sample_id}.txt"
        Image.new("RGB", (256, 256), (index % 256, 0, 0)).save(image)
        label.write_text(
            "".join(f"{class_id} 0.5 0.5 0.25 0.25\n" for class_id in classes),
            encoding="ascii",
        )
        result.append(CandidateRecord(
            sample_id=sample_id,
            source_group_id=f"group-{index:03d}",
            source_id="source-001",
            original_filename=image.name,
            image_path=image.relative_to(tmp_path).as_posix(),
            image_sha256=sha256_file(image),
            label_path=label.relative_to(tmp_path).as_posix(),
            review_status="accepted",
            annotation_status="provisional_human_reviewed_poc",
            accepted_classes=tuple(DEFECT_NAMES[class_id] for class_id in classes),
            quarantine_reason=None,
        ))
    return tuple(result)

def test_b0_gate_requires_twenty_images_and_two_classes(tmp_path: Path) -> None:
    blocked = _accepted_records(tmp_path, 19, (0, 1))
    assert evaluate_revision_gate(blocked, "B0", manifest_root=tmp_path).status == "blocked_data"
    ready = _accepted_records(tmp_path, 20, (0, 1))
    assert evaluate_revision_gate(ready, "B0", manifest_root=tmp_path).status == "ready"

def test_v02_rebuilds_splits_from_all_records_and_preserves_v01(tmp_path: Path) -> None:
    records_v01 = _accepted_records(tmp_path, 20, (0, 1))
    versions = tmp_path / "versions"
    v01 = publish_revision(records_v01, tmp_path, versions, version="public-external-v0.1", stage="B0")
    old_assignments = v01.assignments.read_bytes()
    records_v02 = _accepted_records(tmp_path, 100, (0, 1, 2))
    v02 = publish_revision(records_v02, tmp_path, versions, version="public-external-v0.2", stage="B1")
    v02_assignments = [
        json.loads(line) for line in v02.assignments.read_text(encoding="utf-8").splitlines()
    ]
    assert len(v02_assignments) == 100
    assert v01.assignments.read_bytes() == old_assignments
    assert v01.manifest_sha256 != v02.manifest_sha256

def test_group_split_is_seeded_nonempty_and_leakage_free(tmp_path: Path) -> None:
    records = list(_accepted_records(tmp_path, 20, (0, 1)))
    records[1] = replace(records[1], source_group_id=records[0].source_group_id)
    first = assign_group_stratified_v1(tuple(records), seed=42)
    second = assign_group_stratified_v1(tuple(reversed(records)), seed=42)
    assert first == second
    assert set(first.values()) == {"train", "val", "test"}
    assert first[records[0].sample_id] == first[records[1].sample_id]

def test_b1_gate_checks_images_classes_and_split_box_counts(tmp_path: Path) -> None:
    assert evaluate_revision_gate(
        _accepted_records(tmp_path, 99, (0, 1, 2)), "B1", manifest_root=tmp_path,
    ).status == "blocked_data"
    assert evaluate_revision_gate(
        _accepted_records(tmp_path, 100, (0, 1)), "B1", manifest_root=tmp_path,
    ).status == "blocked_data"
    sparse = list(_accepted_records(tmp_path, 100, (0, 1, 2)))
    for index, record in enumerate(sparse[5:], start=5):
        (tmp_path / str(record.label_path)).write_text(
            "0 0.5 0.5 0.25 0.25\n1 0.5 0.5 0.25 0.25\n", encoding="ascii",
        )
        sparse[index] = replace(record, accepted_classes=DEFECT_NAMES[:2])
    gate = evaluate_revision_gate(tuple(sparse), "B1", manifest_root=tmp_path)
    assert gate.status == "blocked_data"
    assert any("BOX_COUNT" in reason for reason in gate.reasons)

def test_revision_refuses_different_content_at_existing_version(tmp_path: Path) -> None:
    versions = tmp_path / "versions"
    original = _accepted_records(tmp_path, 20, (0, 1))
    publish_revision(original, tmp_path, versions, version="public-external-v0.1", stage="B0")
    changed = _accepted_records(tmp_path, 21, (0, 1))
    with pytest.raises(ValueError, match="REVISION_IMMUTABLE"):
        publish_revision(changed, tmp_path, versions, version="public-external-v0.1", stage="B0")
```

The B1 gate test imports no `torch` or Ultralytics module; CUDA remains a later resource decision and cannot alter these data-gate results.

- [ ] **Step 2: Run revision tests and confirm red**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_public_external_revision.py -q --basetemp .test-tmp/revision-red`

Expected: FAIL because the revision module does not exist.

- [ ] **Step 3: Implement explicit gates and group-level stratification**

```python
@dataclass(frozen=True, slots=True)
class RevisionGate:
    stage: Literal["B0", "B1"]
    status: Literal["ready", "blocked_data"]
    accepted_images: int
    represented_classes: tuple[str, ...]
    train_boxes: Mapping[str, int]
    test_boxes: Mapping[str, int]
    reasons: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class PublishedRevision:
    version: str
    root: Path
    manifest: Path
    assignments: Path
    data_yaml: Path
    manifest_sha256: str
```

Implement these exact public signatures: `assign_group_stratified_v1(records: tuple[CandidateRecord, ...], *, seed: int = 42) -> Mapping[str, Literal["train", "val", "test"]]`, `evaluate_revision_gate(records: tuple[CandidateRecord, ...], stage: Literal["B0", "B1"], *, manifest_root: Path, assignments: Mapping[str, Literal["train", "val", "test"]] | None = None) -> RevisionGate`, and `publish_revision(records: tuple[CandidateRecord, ...], manifest_root: Path, output_root: Path, *, version: str, stage: Literal["B0", "B1"], seed: int = 42) -> PublishedRevision`. If `assignments` is null, `evaluate_revision_gate()` calls `assign_group_stratified_v1(records, seed=42)`; it then reads strict labels below `manifest_root`, computes exact image/class/split-box counts, and fails closed on missing labels. Group by `source_group_id`, order groups by rare-class coverage then seeded tie-breaking, and greedily minimize per-split class/image deficits for target proportions 70/15/15. Never split a group. Evaluate B1's 30-train-box and 10-test-box thresholds only after these assignments exist. Publish through a same-volume staging directory, write `revision.json`, `manifest.jsonl`, `assignments.jsonl`, and `data.yaml`, then atomically rename. `revision.json` records `version`, full accepted-manifest SHA-256, `split_seed: 42`, `split_algorithm: group-stratified-v1`, source-group assignments SHA-256, class/image/box counts, and UTC creation time. Existing identical revisions are reusable; differing revisions fail with `REVISION_IMMUTABLE`.

- [ ] **Step 4: Run revision tests and confirm green**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_public_external_revision.py -q --basetemp .test-tmp/revision-green`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add tools/vision/fc_bga_yolo/public_external_revision.py tools/vision/fc_bga_yolo/tests/test_public_external_revision.py tools/vision/fc_bga_yolo/configs/public_external.template.yaml
git commit -m "feat: publish versioned public FC-BGA datasets"
```

---

### Task 4: Non-Deployable Public External Training Profiles

**Files:**
- Create: `tools/vision/fc_bga_yolo/configs/train_public_external_b0.yaml`
- Create: `tools/vision/fc_bga_yolo/configs/train_public_external_b1.yaml`
- Modify: `tools/vision/fc_bga_yolo/train.py`
- Modify: `tools/vision/fc_bga_yolo/run_training_stage.py`
- Modify: `tools/vision/fc_bga_yolo/tests/test_training_commands.py`
- Modify: `tools/vision/fc_bga_yolo/tests/test_training_stage.py`

**Interfaces:**
- Extends `TrainingSettings.profile` with `public_external`.
- Produces: `_public_external_dataset_root()` and profile-specific preflight.
- Consumes: `PublishedRevision`, B0/B1 gate report, and Plan 1's resource decisions.

- [ ] **Step 1: Write failing profile and metadata-gate tests**

```python
def test_public_external_b0_defaults_match_design() -> None:
    settings = load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_public_external_b0.yaml"))
    assert settings.profile == "public_external"
    assert (settings.imgsz, settings.epochs, settings.patience) == (640, 10, 5)
    assert settings.device == "auto"

def test_public_external_cannot_build_deployable_metadata(tmp_path: Path) -> None:
    settings = replace(
        load_training_settings(Path("tools/vision/fc_bga_yolo/configs/train_public_external_b0.yaml")),
        data=str(tmp_path / "data.yaml"),
    )
    with pytest.raises(ValueError, match="PUBLIC_EXTERNAL_MODEL_NOT_DEPLOYABLE"):
        build_training_metadata(settings, tmp_path / "best.pt", result_paths={}, runtime_versions={})
```

In the same test files, create `test_public_external_rejects_class_order_drift` by reversing `data.yaml.names` and asserting `DATA_CLASS_MISMATCH`; create `test_public_external_requires_revision_metadata` by deleting `revision.json` and asserting `PUBLIC_REVISION_UNAVAILABLE`; create `test_b1_rejects_v01_revision` with `version=public-external-v0.1` and assert `PUBLIC_REVISION_STAGE_MISMATCH`; and create `test_b1_without_cuda_is_skipped_before_ultralytics_import` with the Plan 1 `CPU_PROBE`, a `sys.modules["ultralytics"]` sentinel that fails on access, and assert `status == "skipped_resource"`.

- [ ] **Step 2: Run focused training tests and confirm red**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_training_commands.py tools/vision/fc_bga_yolo/tests/test_training_stage.py -q --basetemp .test-tmp/public-profile-red`

Expected: FAIL because the profile and configs do not exist.

- [ ] **Step 3: Implement profile-specific preflight and configs**

Allow `public_external` in `_validate_training_settings()`. Require fixed seven-class names, exact split paths, an immutable revision manifest, all nonempty splits, and a passing `RevisionGate`. Keep formal `fc_bga` validation unchanged. Return `PUBLIC_EXTERNAL_MODEL_NOT_DEPLOYABLE` from `build_training_metadata()` for this profile while preserving `PUBLIC_SMOKE_MODEL_NOT_DEPLOYABLE` for smoke.

Configure B0 as 640/10/patience 5/batch 4/device auto/workers 0/seed 42 and B1 as 640/50/patience 10/batch 4/device auto/workers 0/seed 42. The stage runner resolves B1 to `skipped_resource` without CUDA before importing Ultralytics.

- [ ] **Step 4: Run training tests and confirm green**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_training_commands.py tools/vision/fc_bga_yolo/tests/test_training_stage.py -q --basetemp .test-tmp/public-profile-green`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

```powershell
git add tools/vision/fc_bga_yolo/train.py tools/vision/fc_bga_yolo/run_training_stage.py tools/vision/fc_bga_yolo/configs/train_public_external_b0.yaml tools/vision/fc_bga_yolo/configs/train_public_external_b1.yaml tools/vision/fc_bga_yolo/tests/test_training_commands.py tools/vision/fc_bga_yolo/tests/test_training_stage.py
git commit -m "feat: gate public FC-BGA training profiles"
```

---

### Task 5: Empty-Class Reports and Source-Group Bootstrap

**Files:**
- Create: `tools/vision/fc_bga_yolo/public_external_evaluation.py`
- Create: `tools/vision/fc_bga_yolo/tests/test_public_external_evaluation.py`
- Modify: `tools/vision/fc_bga_yolo/train.py`

**Interfaces:**
- Produces: `ImageValidationStats`, `ValidationStatsCollector`, `build_observed_class_report()`, `grouped_bootstrap_map()`, and `write_public_evaluation_report()`.
- Consumes: Ultralytics `DetMetrics`, `ap_per_class`, dataset assignment records, and `source_group_id` mapping.

- [ ] **Step 1: Write the seven-class/two-GT-class failing test**

```python
def test_empty_classes_are_null_and_excluded_from_observed_map() -> None:
    report = build_observed_class_report(
        names=DEFECT_NAMES,
        nt_per_class=np.array([10, 5, 0, 0, 0, 0, 0]),
        ap_class_index=np.array([0, 1]),
        class_results=((0.8, 0.7, 0.6, 0.5), (0.4, 0.3, 0.2, 0.1)),
        native_results=(0.6, 0.5, 0.4, 0.3),
    )
    assert report["observed_class_mAP50"] == pytest.approx(0.4)
    assert report["classes"]["EXTRA_BALL"] == {"total_gt": 0, "status": "no_evidence", "metrics": None}
    assert report["footnote"] == "mAP is computed over classes with nonzero ground-truth instances only."
```

- [ ] **Step 2: Write the failing block-bootstrap test**

```python
def _validation_stats(groups: Mapping[str, int]) -> tuple[ImageValidationStats, ...]:
    result = []
    for group_id, image_count in groups.items():
        for index in range(image_count):
            result.append(ImageValidationStats(
                sample_id=f"{group_id}-{index}",
                source_group_id=group_id,
                tp=np.ones((1, 10), dtype=bool),
                conf=np.array([0.9]),
                pred_cls=np.array([0.0]),
                target_cls=np.array([0.0]),
            ))
    return tuple(result)

def test_grouped_bootstrap_moves_all_group_images_as_one_block() -> None:
    groups = {"g1": 5, "g2": 2, "g3": 1}
    groups.update({f"g{index}": 1 for index in range(4, 31)})
    stats = _validation_stats(groups)
    sampled = []
    grouped_bootstrap_map(stats, resamples=1, seed=42, observer=sampled.append)
    for resample in sampled:
        for group_id in set(item.source_group_id for item in resample):
            original_count = sum(item.source_group_id == group_id for item in stats)
            sampled_count = sum(item.source_group_id == group_id for item in resample)
            assert sampled_count % original_count == 0
```

- [ ] **Step 3: Run evaluation tests and confirm red**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_public_external_evaluation.py -q --basetemp .test-tmp/public-eval-red`

Expected: FAIL because the evaluation module does not exist.

- [ ] **Step 4: Implement raw per-image validation-stat collection**

```python
@dataclass(frozen=True, slots=True)
class ImageValidationStats:
    sample_id: str
    source_group_id: str
    tp: np.ndarray
    conf: np.ndarray
    pred_cls: np.ndarray
    target_cls: np.ndarray
```

Implement `ValidationStatsCollector.__init__(group_by_filename: Mapping[str, tuple[str, str]]) -> None`, `ValidationStatsCollector.on_val_batch_end(validator: object) -> None`, and `ValidationStatsCollector.records() -> tuple[ImageValidationStats, ...]`. The constructor normalizes keys with `Path(name).name`, rejects duplicate basenames or sample IDs, and initializes a zero cursor. Ultralytics 8.4.120 appends one entry per image to each list in `validator.metrics.stats`; `im_name` contains the basename. At `on_val_batch_end`, require the keys `tp`, `conf`, `pred_cls`, `target_cls`, and `im_name`, require equal list lengths, and copy entries from the cursor through the new length. Bind each `im_name` to `(sample_id, source_group_id)` from the immutable revision manifest, copy NumPy arrays before Ultralytics clears its buffers, then advance the cursor. `records()` returns the immutable accumulated tuple and rejects duplicate or missing sample IDs. Never use Ultralytics `target_img` as the source-group identifier. Pin this behavior with a fake-validator test and a version assertion for Ultralytics 8.4.120.

- [ ] **Step 5: Implement explicit observed-class reporting and grouped AP recomputation**

```python
EMPTY_CLASS_FOOTNOTE = "mAP is computed over classes with nonzero ground-truth instances only."
INSUFFICIENT_WARNING = (
    "INSUFFICIENT STATISTICAL EVIDENCE: workflow rehearsal metrics are not model-performance evidence."
)
```

Implement these exact public signatures: `build_observed_class_report(*, names: tuple[str, ...], nt_per_class: np.ndarray, ap_class_index: np.ndarray, class_results: tuple[tuple[float, float, float, float], ...], native_results: tuple[float, float, float, float]) -> dict[str, object]`, `grouped_bootstrap_map(records: tuple[ImageValidationStats, ...], *, resamples: int = 1000, seed: int = 42, observer: Callable[[tuple[ImageValidationStats, ...]], None] | None = None) -> Mapping[str, tuple[float, float]]`, and `write_public_evaluation_report(path: Path, report: Mapping[str, object]) -> Path`. `build_observed_class_report()` maps `ap_class_index` to result rows, writes `metrics=None/status=no_evidence` wherever `nt_per_class[class_id] == 0`, averages AP only over nonempty classes, and preserves the unchanged native tuple under `native_ultralytics`. For each bootstrap draw, sample unique `source_group_id` values with replacement, append all image-stat arrays for every selected group occurrence, call `observer(tuple(sampled_records))` when supplied, concatenate `tp/conf/pred_cls/target_cls`, and call the pinned Ultralytics `ap_per_class`. Return percentile 2.5/97.5 intervals for mAP50 and mAP50-95. Refuse bootstrap with fewer than 30 unique test groups; still emit denominators and the insufficient-evidence warning. `write_public_evaluation_report()` uses a sibling temporary file and `Path.replace()` for atomic JSON output.

- [ ] **Step 6: Integrate the collector into public external test evaluation**

Before `best_model.val()`, attach the exact callback with `best_model.add_callback("on_val_batch_end", collector.on_val_batch_end)`. After validation, build the native/observed report; run bootstrap only for B1 when the source-group threshold passes. Write `public_evaluation_report.json` beside the run outputs. Do not create formal metadata.

- [ ] **Step 7: Run evaluation and training tests**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests/test_public_external_evaluation.py tools/vision/fc_bga_yolo/tests/test_training_commands.py -q --basetemp .test-tmp/public-eval-green`

Expected: all tests pass, including the seven-class/two-GT-class and block-resampling regressions.

- [ ] **Step 8: Commit Task 5**

```powershell
git add tools/vision/fc_bga_yolo/public_external_evaluation.py tools/vision/fc_bga_yolo/train.py tools/vision/fc_bga_yolo/tests/test_public_external_evaluation.py tools/vision/fc_bga_yolo/tests/test_training_commands.py
git commit -m "feat: report public FC-BGA evaluation evidence"
```

---

### Task 6: Provisional Annotation, Revision Publication, and Allowed Training

**Files:**
- Runtime/manual: `data/external/fc_bga_public_external/review/candidates.jsonl`
- Runtime/manual: `data/external/fc_bga_public_external/review/labels/*.txt`
- Runtime/generated: `data/external/fc_bga_public_external/versions/public-external-v0.1/`
- Modify: `data/external/fc_bga_public_external/README.md`
- Modify: `tools/vision/fc_bga_yolo/README.md`

**Interfaces:**
- Consumes: review queue, annotation rules, revision publisher, official checkpoint baseline, and stage runner.
- Produces: either a coverage-shortfall report or an immutable v0.1 revision plus B0 run/report. B1 produces `skipped_resource` on the current CPU-only host unless a compatible CUDA environment is later supplied.

- [ ] **Step 1: Review every candidate at native resolution**

For each image, use the approved seven-class visual rules. Create normalized YOLO boxes only when the boundary and class are visually auditable. Set accepted records to `review_status="accepted"` and `annotation_status="provisional_human_reviewed_poc"`; set ambiguous or out-of-scope records to `review_status="quarantined"`, `annotation_status=null`, and a specific reason such as `GRID_REFERENCE_UNAVAILABLE`, `BOUNDARY_AMBIGUOUS`, `LICENSE_UNVERIFIED`, or `VISIBLE_LIGHT_SCOPE_MISMATCH`.

- [ ] **Step 2: Run the candidate audit**

```powershell
python tools\vision\fc_bga_yolo\public_external_manifest.py --manifest data\external\fc_bga_public_external\review\candidates.jsonl --sources data\external\fc_bga_public_external\sources.json --json-report .test-tmp\public-external-candidate-audit.json
```

Expected: zero structural errors. If fewer than 20 images or fewer than two classes are accepted after up to 100 licensed candidates, write the coverage-shortfall report and stop before publication; do not weaken labels.

- [ ] **Step 3: Publish immutable v0.1 when B0 gates pass**

```powershell
python tools\vision\fc_bga_yolo\public_external_revision.py --manifest data\external\fc_bga_public_external\review\candidates.jsonl --sources data\external\fc_bga_public_external\sources.json --output data\external\fc_bga_public_external\versions --version public-external-v0.1 --stage B0 --seed 42
```

Expected: nonempty train/val/test splits, no group leakage, fixed class order, and recorded manifest/assignment hashes.

- [ ] **Step 4: Run B0 preflight and training**

```powershell
.\.test-tmp\venv-yolo\Scripts\python.exe tools\vision\fc_bga_yolo\train.py --config tools\vision\fc_bga_yolo\configs\train_public_external_b0.yaml --check-only
.\.test-tmp\venv-yolo\Scripts\python.exe tools\vision\fc_bga_yolo\run_training_stage.py --stage B0 --config tools\vision\fc_bga_yolo\configs\train_public_external_b0.yaml --report .test-tmp\training-runs\stage-b0-resource-report.json
```

Expected: the report begins with the exact insufficient-statistical-evidence warning, empty classes are `null/no_evidence`, and no deployable metadata is created.

- [ ] **Step 5: Audit B1 without bypassing its gates**

Run the revision gate against all accepted candidates. On the current host, a data-ready B1 still returns `skipped_resource` because PyTorch is CPU-only. Record the resource report; do not force CPU execution and do not describe the stage as failed or hung.

- [ ] **Step 6: Update both READMEs with actual outcomes**

Document accepted/quarantined counts, represented classes, dataset revision/hash, Stage B0 status, output locations, and Stage B1 gate/resource status. State that no public-data metric is production evidence.

- [ ] **Step 7: Commit Task 6 documentation only**

```powershell
git add data/external/fc_bga_public_external/README.md tools/vision/fc_bga_yolo/README.md
git commit -m "docs: record public FC-BGA rehearsal evidence"
```

Do not add downloaded images by default. For the user-authorized Roboflow `paween/bga-ram-chips-detection-t3cqn` version 1 CC BY 4.0 exception dated 2026-08-17, add only the 56 exact-deduplicated `review/images/*.jpg` candidates, `review/candidates.jsonl`, attribution, and the verified license snapshot. Keep labels without verified redistribution permission, dataset versions, checkpoints, and run directories untracked.

---

### Task 7: Full Verification and GitHub Delivery

**Files:**
- Verify all files from Plans 1 and 2.
- No new production code is introduced in this task.

**Interfaces:**
- Produces: evidence-backed final status and a clean pushed `main` branch.

- [ ] **Step 1: Run the complete FC-BGA test suite**

Run: `python -m pytest tools/vision/fc_bga_yolo/tests -q --basetemp .test-tmp/public-external-final`

Expected: all tests pass.

- [ ] **Step 2: Run every CLI help/preflight path**

```powershell
python tools\vision\fc_bga_yolo\download_models.py --help
python tools\vision\fc_bga_yolo\prepare_public_external_candidates.py --help
python tools\vision\fc_bga_yolo\public_external_manifest.py --help
python tools\vision\fc_bga_yolo\public_external_revision.py --help
python tools\vision\fc_bga_yolo\run_training_stage.py --help
```

Expected: each command exits 0 without a network request.

- [ ] **Step 3: Verify model/data boundaries**

Assert Stage A/B0/B1 public run directories contain no `model_metadata.json`; formal `fc_bga` tests still generate metadata only after the four-light manifest gate. Verify `.gitignore` excludes caches, versions, weights, runs, `.pt`, `.onnx`, and `.engine` files.

- [ ] **Step 4: Run whitespace, secret, and artifact scans**

```powershell
git diff --check
rg -n "ROBOFLOW_API_KEY\s*=|api[_-]?key\s*[:=]" . -g "!.test-tmp/**" -g "!runs/**" -g "!data/external/fc_bga_public_smoke/downloads/**"
git status --short
```

Expected: no real key value, no tracked runtime artifact, and only intended source/doc changes.

- [ ] **Step 5: Review commit history and push**

```powershell
git log --oneline --decorate -12
git push origin main
```

Expected: all plan commits are present and `origin/main` advances without force-push.
