# PCB Image End-to-End Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the licensed PCB image fixtures to the PIS-IN import API and prove validated ingestion, immutable idempotency, evidence preview, review, alerting, and report creation through an automated end-to-end simulation.

**Architecture:** The existing file-path metadata contract remains the system boundary. A focused image-validation service resolves paths under `AOI_IMAGE_ROOT`, verifies the required light set, hashes and fully decodes images before the route opens an ingestion transaction; the simulator remains an HTTP-only client and writes a JSON evidence report.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2, Pillow, httpx, pytest, Docker Compose, SQLite for tests, PostgreSQL for composed runtime.

## Global Constraints

- Only files resolving under `AOI_IMAGE_ROOT` may be read; reject path traversal and symlink escape.
- Accept JPEG and PNG only, at most 50 MiB, 8192 px per side, and 40,000,000 total pixels.
- Require exactly one each of `R`, `G`, `B`, and `RING` before inference.
- Invalid identity or attachment input returns HTTP 202 with `status`, `quarantine_id`, `reason_code`, and a sanitized `reason`; it creates no event, attachment, or inference result.
- Exact Source Key replay returns HTTP 200 only when light IDs, canonical paths, and hashes match; any mismatch is `IDEMPOTENCY_CONFLICT` quarantine.
- The preview endpoint reads only persisted verified attachments and returns 409 if evidence disappears or changes.
- PCB decisions are synthetic. Regular PCB events use `REVIEW`; only the alert scenario explicitly uses two `DEFECT` and eighteen `NORMAL` labels and reports `synthetic_decision=true`.
- Do not add a database migration, upload endpoint, remote URL fetch, 3D synthesis, real model inference, or frontend controls.

---

## File Structure

- Create `apps/api/app/services/image_evidence.py`: canonical path, light-set, size, SHA-256, format, and decode validation.
- Create `apps/api/tests/image_fixtures.py`: small real JPEG/PNG fixture generation and PIS-IN payload construction.
- Create `apps/api/tests/test_image_evidence.py`: focused validation service tests.
- Modify `apps/api/app/main.py`: accept/store `image_root` and configure it from `AOI_IMAGE_ROOT`.
- Modify `apps/api/app/api/routes/operations.py`: quarantine helper, validated import, immutable replay, concurrent uniqueness recovery, and preview route.
- Modify `apps/api/tests/test_operations_api.py`: API import, isolation, idempotency, preview, review, alert, and report integration coverage.
- Modify `apps/api/pyproject.toml`: add bounded Pillow runtime dependency.
- Create `services/simulator/simulator/e2e.py`: one-shot HTTP orchestration and JSON report.
- Create `services/simulator/tests/test_e2e.py`: fake-transport tests for report behavior and exit status.
- Modify `services/simulator/simulator/main.py`: select continuous or one-shot mode by `SIM_MODE`.
- Modify `infra/docker-compose.yml`: read-only image fixture mounts and explicit roots/mode.
- Modify `.env.example`: document image root and simulator mode/report path.
- Modify `data/external/pcb_stability_samples/README.md`: add the end-to-end invocation and evidence boundary.

---

### Task 1: Controlled Image Evidence Validation

**Files:**
- Create: `apps/api/app/services/image_evidence.py`
- Create: `apps/api/tests/image_fixtures.py`
- Create: `apps/api/tests/test_image_evidence.py`
- Modify: `apps/api/pyproject.toml`

**Interfaces:**
- Produces: `EvidenceValidationError(reason_code: str, public_reason: str)`.
- Produces: `ValidatedImage(light_id: str, path: Path, sha256: str, width: int, height: int, media_type: str)`.
- Produces: `validate_image_set(attachments: tuple[NormalizedAttachment, ...], image_root: Path) -> tuple[ValidatedImage, ...]` sorted in `R`, `G`, `B`, `RING` order.

- [ ] **Step 1: Add real image fixture helpers**

Create helpers that use Pillow to save a small JPEG or PNG, calculate its SHA-256, and build four `NormalizedAttachment` values. Paths in payloads are absolute fixture paths so production validation exercises real canonicalization.

```python
def write_image(path: Path, *, format: str = "JPEG", size: tuple[int, int] = (64, 48)) -> str:
    Image.new("RGB", size, (30, 140, 80)).save(path, format=format)
    return hashlib.sha256(path.read_bytes()).hexdigest()

def make_attachments(path: Path, sha256: str) -> tuple[NormalizedAttachment, ...]:
    return tuple(NormalizedAttachment(light, str(path), sha256) for light in ("R", "G", "B", "RING"))
```

- [ ] **Step 2: Write failing validation tests**

Cover a valid JPEG, valid PNG, missing/duplicate/unknown light, outside-root path, symlink escape when supported, missing/empty/oversized file, wrong hash, truncated JPEG, unsupported GIF, oversized dimensions, and deterministic light ordering. Assert exact reason codes.

```python
def test_validates_and_orders_complete_light_set(tmp_path: Path) -> None:
    sha256 = write_image(tmp_path / "board.jpg")
    result = validate_image_set(tuple(reversed(make_attachments(tmp_path / "board.jpg", sha256))), tmp_path)
    assert [item.light_id for item in result] == ["R", "G", "B", "RING"]
    assert result[0].media_type == "image/jpeg"

def test_rejects_hash_mismatch(tmp_path: Path) -> None:
    write_image(tmp_path / "board.jpg")
    with pytest.raises(EvidenceValidationError) as error:
        validate_image_set(make_attachments(tmp_path / "board.jpg", "0" * 64), tmp_path)
    assert error.value.reason_code == "HASH_MISMATCH"
```

- [ ] **Step 3: Run tests and verify RED**

Run: `cd apps/api; python -m pytest tests/test_image_evidence.py -v`

Expected: collection fails because `app.services.image_evidence` does not exist.

- [ ] **Step 4: Implement the minimal validation service**

Use `Path.resolve(strict=False)` for root containment, `Path.is_file()`, `stat().st_size`, chunked hashing, and Pillow `Image.verify()` followed by reopen plus `load()` for full decode. Convert decompression-bomb warnings/errors into `IMAGE_DIMENSIONS_INVALID`; never include the absolute server path in `public_reason`.

```python
REQUIRED_LIGHTS = ("R", "G", "B", "RING")

@dataclass(frozen=True, slots=True)
class EvidenceValidationError(ValueError):
    reason_code: str
    public_reason: str

@dataclass(frozen=True, slots=True)
class ValidatedImage:
    light_id: str
    path: Path
    sha256: str
    width: int
    height: int
    media_type: str
```

- [ ] **Step 5: Add Pillow dependency and run focused tests**

Add `"pillow>=11.3,<12"` to API dependencies.

Run: `cd apps/api; python -m pytest tests/test_image_evidence.py -v`

Expected: all validation tests pass.

- [ ] **Step 6: Commit Task 1**

```powershell
git add -- apps/api/app/services/image_evidence.py apps/api/tests/image_fixtures.py apps/api/tests/test_image_evidence.py apps/api/pyproject.toml
git commit -m "feat: validate AOI image evidence"
```

---

### Task 2: Validated Import, Quarantine, and Immutable Idempotency

**Files:**
- Modify: `apps/api/app/main.py`
- Modify: `apps/api/app/api/routes/operations.py`
- Modify: `apps/api/tests/test_operations_api.py`

**Interfaces:**
- Consumes: `validate_image_set(...)`, `EvidenceValidationError`, and `ValidatedImage` from Task 1.
- Produces: `create_app(..., image_root: str | Path | None = None) -> FastAPI` with `app.state.image_root: Path`.
- Produces: `quarantine_import(session, raw, reason_code, reason) -> dict[str, str]`.

- [ ] **Step 1: Replace legacy fake-path import tests with real-file failing tests**

Create an unseeded app with `image_root=tmp_path`, send a four-light payload built from a real image, and assert first import `201`, replay `200`, same UUID, four attachments, and one inference result. Add tests for missing identity and every validation reason code; query database counts to prove quarantined requests created no event evidence.

```python
app = create_app(database_url="sqlite+pysqlite:///:memory:", mode="demo", image_root=tmp_path)
payload = make_pis_in_payload(tmp_path / "board.jpg", scenario="REVIEW")
first = client.post("/api/v1/inspections/import/pis-in", json=payload)
second = client.post("/api/v1/inspections/import/pis-in", json=payload)
assert (first.status_code, second.status_code) == (201, 200)
assert first.json()["event_uuid"] == second.json()["event_uuid"]
```

- [ ] **Step 2: Add failing conflict and ordering tests**

Replay with shuffled images and expect exact idempotency; then replay the same Source Key with one changed path/hash and expect HTTP 202 plus `IDEMPOTENCY_CONFLICT`. Verify persisted attachments still match the first request.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `cd apps/api; python -m pytest tests/test_operations_api.py -k "pis_in_import" -v`

Expected: failures because `create_app` lacks `image_root`, invalid files are accepted, and replays do not compare attachments.

- [ ] **Step 4: Configure the image root**

Add `image_root` to `create_app`; resolve `image_root` or `AOI_IMAGE_ROOT`. When neither is set, use an intentionally non-existent `Path("./aoi-images-disabled").resolve()` so imports fail closed rather than gaining access to the working directory.

```python
configured_image_root = image_root or os.getenv("AOI_IMAGE_ROOT", "./aoi-images-disabled")
app.state.image_root = Path(configured_image_root).resolve()
```

- [ ] **Step 5: Centralize quarantine responses**

Map `IdentityUnavailable` to `IDENTITY_MISSING`, map `EvidenceValidationError` directly, sanitize reasons, persist `reason_code: reason`, and return the fixed HTTP 202 payload. Do not commit partial events before quarantine.

- [ ] **Step 6: Validate before inference and make replays immutable**

Call `validate_image_set` before reading an existing event. Compare persisted and submitted sets as sorted `(light_id, canonical_path, sha256)` tuples. Exact match returns the old event; mismatch quarantines. Persist only `ValidatedImage` values and set `image_url` to `/api/v1/inspections/{event_uuid}/image?light_id=RING`.

- [ ] **Step 7: Recover identical concurrent creates**

Wrap the event/attachment/inference insert in one transaction. On `IntegrityError`, roll back, reload by Source Key, compare its stored attachments to the validated request, and return the exact replay response; re-raise only when no committed matching event exists.

- [ ] **Step 8: Run focused and full API tests**

Run: `cd apps/api; python -m pytest tests/test_operations_api.py -k "pis_in_import" -v`

Run: `cd apps/api; python -m pytest -q`

Expected: focused tests and the existing API suite pass.

- [ ] **Step 9: Commit Task 2**

```powershell
git add -- apps/api/app/main.py apps/api/app/api/routes/operations.py apps/api/tests/test_operations_api.py
git commit -m "feat: ingest validated AOI images idempotently"
```

---

### Task 3: Immutable Image Evidence Preview

**Files:**
- Modify: `apps/api/app/api/routes/operations.py`
- Modify: `apps/api/tests/test_operations_api.py`

**Interfaces:**
- Consumes: persisted `Attachment` values and Task 1 validation limits.
- Produces: `GET /api/v1/inspections/{event_uuid}/image?light_id=RING`.

- [ ] **Step 1: Write failing preview tests**

After a valid import, assert default and explicit-light responses contain exact file bytes and media types. Assert 404 for unknown event/light, 409 after deleting the file, and 409 after replacing it with another valid image whose hash differs.

```python
preview = client.get(f"/api/v1/inspections/{event_uuid}/image")
assert preview.status_code == 200
assert preview.headers["content-type"].startswith("image/jpeg")
assert preview.content == image_path.read_bytes()
```

- [ ] **Step 2: Run preview tests and verify RED**

Run: `cd apps/api; python -m pytest tests/test_operations_api.py -k "image_preview" -v`

Expected: 404 because the route does not exist.

- [ ] **Step 3: Implement database-backed preview**

Select the requested event and its persisted attachment. Re-resolve under `app.state.image_root`, require it still exists, stream-hash it, compare to `Attachment.file_hash`, determine `image/jpeg` or `image/png` from the verified suffix/content recorded at ingestion, and return `FileResponse(..., content_disposition_type="inline")`. Never accept a path query parameter.

- [ ] **Step 4: Run preview and full API tests**

Run: `cd apps/api; python -m pytest tests/test_operations_api.py -k "image_preview" -v`

Run: `cd apps/api; python -m pytest -q`

Expected: all pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add -- apps/api/app/api/routes/operations.py apps/api/tests/test_operations_api.py
git commit -m "feat: serve immutable AOI image evidence"
```

---

### Task 4: One-Shot HTTP End-to-End Simulator

**Files:**
- Create: `services/simulator/simulator/e2e.py`
- Create: `services/simulator/tests/test_e2e.py`
- Modify: `services/simulator/simulator/main.py`

**Interfaces:**
- Produces: `run_e2e(client: httpx.Client, *, api_url: str, manifest_path: Path, report_path: Path) -> dict[str, object]`.
- Produces: `main() -> int`, returning zero only when every named assertion passes.
- Report keys: `synthetic_decision`, `started_at`, `finished_at`, `counts`, `latency_ms`, `events`, `quarantines`, `alert_id`, `report_id`, and `assertions`.

- [ ] **Step 1: Write failing fake-transport tests**

Use `httpx.MockTransport` with a stateful handler that returns the expected create/replay/quarantine/review/alert/report sequence. Assert the report counts exact statuses, computes P50/P95 from recorded durations, includes `synthetic_decision=True`, and writes JSON even when one assertion fails.

```python
report = run_e2e(client, api_url="http://test/api/v1", manifest_path=manifest, report_path=report_path)
assert report["synthetic_decision"] is True
assert report["counts"]["quarantined"] == 4
assert all(item["passed"] for item in report["assertions"])
assert json.loads(report_path.read_text(encoding="utf-8"))["report_id"]
```

- [ ] **Step 2: Run simulator tests and verify RED**

Run: `cd services/simulator; python -m pytest tests/test_e2e.py -v`

Expected: collection fails because `simulator.e2e` does not exist.

- [ ] **Step 3: Implement payload and request helpers**

Read `manifest.json`, resolve `normalized_file` paths, calculate current hashes rather than trusting the manifest, and construct PIS-IN payloads with deterministic unique identities. Reuse one image for all four lights in baseline events; mutation scenarios use a second image or temporary corrupted file.

- [ ] **Step 4: Implement the exact workflow**

Run valid import, exact replay, shuffled replay, attachment conflict, missing light, bad hash, corrupt image, outside-root path, review event plus manual review, and 20-event `ST-PCB-ALERT` window. Then list alerts, acknowledge the matching OPEN alert, create its DRAFT report, and fetch it back.

- [ ] **Step 5: Implement durable reporting and exit status**

Record every HTTP duration with `time.perf_counter_ns()`. Calculate percentile values by nearest-rank over the sorted list. Always write UTF-8 JSON in `finally`; `main()` returns `1` when any assertion failed or request raised, otherwise `0`.

- [ ] **Step 6: Select simulator mode**

Keep existing continuous `run()` unchanged. When `SIM_MODE=e2e`, call the one-shot entry point using `SIM_MANIFEST_PATH` and `SIM_REPORT_PATH`; all other values continue current behavior.

- [ ] **Step 7: Run simulator tests**

Run: `cd services/simulator; python -m pytest -q`

Expected: existing scenario tests and new E2E tests pass.

- [ ] **Step 8: Commit Task 4**

```powershell
git add -- services/simulator/simulator/e2e.py services/simulator/simulator/main.py services/simulator/tests/test_e2e.py
git commit -m "test: add AOI image end-to-end simulator"
```

---

### Task 5: Composed Runtime and Evidence Documentation

**Files:**
- Modify: `infra/docker-compose.yml`
- Modify: `.env.example`
- Modify: `data/external/pcb_stability_samples/README.md`

**Interfaces:**
- Consumes: `AOI_IMAGE_ROOT`, `SIM_MODE`, `SIM_MANIFEST_PATH`, and `SIM_REPORT_PATH` from Tasks 2 and 4.
- Produces: read-only API/simulator mounts at `/aoi-images` and report output under `/sim-output`.

- [ ] **Step 1: Write a failing compose contract assertion**

Extend `infra/tests/compose_contract.ps1` to require the API environment `AOI_IMAGE_ROOT=/aoi-images/normalized_1920x1080`, read-only fixture mounts for API and simulator, and explicit E2E override documentation. Run it before changing Compose.

Run: `powershell -ExecutionPolicy Bypass -File .\infra\tests\compose_contract.ps1`

Expected: FAIL because image mounts and variables are absent.

- [ ] **Step 2: Add safe mounts and environment values**

Mount `../data/external/pcb_stability_samples:/aoi-images:ro` into API and simulator. Keep the default simulator in continuous mode so normal `docker compose up` behavior is unchanged; document an E2E override command that sets `SIM_MODE=e2e` and writes a report to a host-mounted output directory.

- [ ] **Step 3: Document exact local commands and limitations**

Add variables to `.env.example` and update the fixture README with:

```powershell
$env:AOI_IMAGE_ROOT = (Resolve-Path '.\data\external\pcb_stability_samples\normalized_1920x1080').Path
python -m uvicorn app.main:app --app-dir .\apps\api --port 8000
```

and the simulator command, report location, expected scenario counts, and the statement that all decisions are synthetic labels rather than image inference.

- [ ] **Step 4: Run compose contract and configuration render**

Run: `powershell -ExecutionPolicy Bypass -File .\infra\tests\compose_contract.ps1`

Run: `docker compose -f infra/docker-compose.yml config`

Expected: contract passes and Compose renders without errors.

- [ ] **Step 5: Commit Task 5**

```powershell
git add -- infra/docker-compose.yml infra/tests/compose_contract.ps1 .env.example data/external/pcb_stability_samples/README.md
git commit -m "chore: configure PCB image simulation runtime"
```

---

### Task 6: Full End-to-End Verification

**Files:**
- Create: `tmp/pcb-e2e-report.json` during verification only; do not commit.
- Modify only files required by evidence-backed failures found in this task.

**Interfaces:**
- Consumes every prior task.
- Produces fresh test output and `tmp/pcb-e2e-report.json` showing the completed workflow.

- [ ] **Step 1: Run all automated suites**

Run: `cd apps/api; python -m pytest -q`

Run: `cd services/simulator; python -m pytest -q`

Run: `powershell -ExecutionPolicy Bypass -File .\infra\tests\compose_contract.ps1`

Expected: zero failures.

- [ ] **Step 2: Start an isolated local API**

Use a temporary SQLite database, disable auto-seed, set the image root to the normalized fixture directory, and start Uvicorn on an unused localhost port. Poll `/api/v1/health` until ready; retain the process ID for guaranteed cleanup.

- [ ] **Step 3: Run the real HTTP simulator**

From `services/simulator`, set `API_URL` to the isolated API, `SIM_MODE=e2e`, `SIM_MANIFEST_PATH` to the absolute fixture manifest, and `SIM_REPORT_PATH` to the absolute `tmp/pcb-e2e-report.json`, then run `python -m simulator.main`.

Expected: exit code 0 and report assertions all `passed=true`.

- [ ] **Step 4: Independently verify persisted outcomes**

Query API endpoints and the JSON report. Confirm: valid create and replay share UUID; four named quarantine classes exist plus conflict; reviewed event is no longer in review queue; `ST-PCB-ALERT` has 10% rate and ACKNOWLEDGED status; DRAFT report references exactly two FAIL event UUIDs; preview bytes hash to the stored RING attachment hash.

- [ ] **Step 5: Stop the API and check the final diff**

Terminate only the process started in Step 2. Run `git diff --check` and `git status --short` scoped to this project. Do not remove user files or unrelated changes.

- [ ] **Step 6: Request code review**

Use the `requesting-code-review` skill against the implementation diff. Address only correctness, security, regression, and missing-test findings relevant to this plan, then rerun Steps 1-5.

- [ ] **Step 7: Commit verification fixes if any**

```powershell
git add -- <only files changed to resolve verified findings>
git commit -m "fix: harden PCB image simulation workflow"
```

If no fixes were required, create no empty commit.
