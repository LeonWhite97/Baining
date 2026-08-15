# PIS-IN AOI Runnable Demo Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or execute each task inline with test-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GPU-free but fully interactive AOI demo with generated inspection events, persistent FastAPI/PostgreSQL data, and eight React pages.

**Architecture:** `InspectionSourceAdapter` normalizes either deterministic simulator payloads or real PIS-IN exports before they enter `apps/api`. The API applies Source Key/idempotency/state rules, invokes a replaceable inference adapter, executes deterministic 2D/3D three-state decisions, and stores events, results, reviews, alerts, reports, and model releases in PostgreSQL. `apps/web` consumes versioned REST endpoints; real PIS-IN and TensorRT adapters can replace demo adapters without changing business APIs.

**Tech Stack:** Python 3.12, FastAPI, Pydantic 2, SQLAlchemy 2, Alembic, PostgreSQL 16, React 18, TypeScript 5, Vite 5, ECharts 5, Vitest, Pytest, Playwright, Docker Compose.

## Global Constraints

- Use ASCII identifiers and UTF-8 Chinese UI copy.
- Keep files focused; routers do HTTP mapping, services hold business rules, repositories own persistence.
- Use `DeviceID + DeviceSessionID + InspectionSequence + TrayID + SlotIndex + Surface` for the canonical Source Key input.
- Light ID is attachment metadata and never part of the event key.
- The demo must start and operate without CUDA, TensorRT, model weights, or image files.
- The frontend must not expose project budget data.
- All user-visible status values come from defined enums, not arbitrary strings.

## File Structure

```text
apps/api/
  pyproject.toml
  alembic.ini
  app/main.py
  app/config.py
  app/db.py
  app/domain/enums.py
  app/domain/source_key.py
  app/adapters/base.py
  app/adapters/demo.py
  app/adapters/pis_in.py
  app/models/inspection.py
  app/models/governance.py
  app/schemas/inspection.py
  app/schemas/dashboard.py
  app/repositories/inspection.py
  app/inference/base.py
  app/inference/demo.py
  app/inference/tensorrt.py
  app/services/ingestion.py
  app/services/orchestration.py
  app/services/decision.py
  app/services/query.py
  app/services/review.py
  app/services/alerts.py
  app/services/reports.py
  app/api/routes/health.py
  app/api/routes/inspection.py
  app/api/routes/dashboard.py
  app/api/routes/review.py
  app/api/routes/alerts.py
  app/api/routes/reports.py
  app/api/routes/models.py
  app/api/routes/demo.py
  migrations/versions/0001_initial.py
  tests/
services/simulator/
  pyproject.toml
  simulator/main.py
  simulator/scenarios.py
  tests/test_scenarios.py
apps/web/
  package.json
  vite.config.ts
  src/main.tsx
  src/app/App.tsx
  src/api/client.ts
  src/api/types.ts
  src/layout/AppShell.tsx
  src/pages/*.tsx
  src/components/*.tsx
  src/styles/tokens.css
  src/**/*.test.tsx
  tests/e2e/*.spec.ts
infra/docker-compose.yml
infra/nginx.conf
```

---

### Task 1: FastAPI Service Contract and Health Check

**Files:**
- Create: `apps/api/pyproject.toml`
- Create: `apps/api/app/__init__.py`
- Create: `apps/api/app/config.py`
- Create: `apps/api/app/main.py`
- Create: `apps/api/app/api/routes/health.py`
- Test: `apps/api/tests/test_health.py`

**Interfaces:**
- Produces: `create_app() -> FastAPI`
- Produces: `GET /api/v1/health -> {status, mode, version}`

- [ ] **Step 1: Write the failing health test**

```python
from fastapi.testclient import TestClient
from app.main import create_app

def test_health_reports_demo_mode():
    response = TestClient(create_app()).get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "demo", "version": "v3.5"}
```

- [ ] **Step 2: Run the test and verify the import fails**

Run: `cd apps/api && python -m pytest tests/test_health.py -v`
Expected: FAIL because `app.main` does not exist.

- [ ] **Step 3: Implement the minimal application factory**

```python
from fastapi import FastAPI
from app.api.routes.health import router as health_router

def create_app() -> FastAPI:
    app = FastAPI(title="PIS-IN AOI AI", version="3.5")
    app.include_router(health_router, prefix="/api/v1")
    return app

app = create_app()
```

- [ ] **Step 4: Run the health test**

Run: `cd apps/api && python -m pytest tests/test_health.py -v`
Expected: PASS.

- [ ] **Step 5: Commit the service skeleton**

```bash
git add apps/api
git commit -m "feat: scaffold AOI FastAPI service"
```

### Task 2: Canonical Source Key and Domain Enums

**Files:**
- Create: `apps/api/app/domain/enums.py`
- Create: `apps/api/app/domain/source_key.py`
- Test: `apps/api/tests/domain/test_source_key.py`
- Test: `apps/api/tests/domain/test_enums.py`

**Interfaces:**
- Produces: `SourceKeyParts` dataclass
- Produces: `generate_source_key_hash(parts: SourceKeyParts) -> str`
- Produces: `AssociationStatus`, `Decision`, `InferenceMode`, `Resolution` enums

- [ ] **Step 1: Write deterministic-key tests**

```python
def test_source_key_is_order_independent_and_excludes_light_id():
    parts = SourceKeyParts("PIS01", "BOOT77", "1042", "TRAY9", "A07", "TOP")
    first = generate_source_key_hash(parts)
    second = generate_source_key_hash(parts)
    assert first == second
    assert len(first) == 64
```

- [ ] **Step 2: Run the tests**

Run: `cd apps/api && python -m pytest tests/domain -v`
Expected: FAIL because domain types are missing.

- [ ] **Step 3: Implement canonical JSON plus SHA-256**

```python
canonical = json.dumps(asdict(parts), sort_keys=True, separators=(",", ":"))
return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Add enum transition assertions**

```python
assert AssociationStatus.RECEIVED.value == "RECEIVED"
assert Decision.REVIEW.value == "REVIEW"
assert InferenceMode.DEMO.value == "DEMO"
```

- [ ] **Step 5: Run domain tests and commit**

Run: `cd apps/api && python -m pytest tests/domain -v`
Expected: PASS.

```bash
git add apps/api/app/domain apps/api/tests/domain
git commit -m "feat: add deterministic AOI event identity"
```

### Task 3: Persistent Inspection and Governance Models

**Files:**
- Create: `apps/api/app/db.py`
- Create: `apps/api/app/models/base.py`
- Create: `apps/api/app/models/inspection.py`
- Create: `apps/api/app/models/governance.py`
- Create: `apps/api/migrations/env.py`
- Create: `apps/api/migrations/versions/0001_initial.py`
- Test: `apps/api/tests/db/test_schema.py`

**Interfaces:**
- Produces SQLAlchemy models: `InspectionEvent`, `Attachment`, `InferenceResult`, `QuarantineEvent`, `ReviewRecord`, `StationAlert`, `AnomalyReport`, `ModelRelease`
- Enforces unique keys: `source_key_hash`; `(event_uuid, model_version, policy_version, input_fingerprint)`; `(event_uuid, data_type, light_id, file_hash)`

- [ ] **Step 1: Write schema constraint tests**

```python
def test_duplicate_source_key_is_rejected(session, event_factory):
    session.add(event_factory(source_key_hash="a" * 64))
    session.commit()
    session.add(event_factory(source_key_hash="a" * 64))
    with pytest.raises(IntegrityError):
        session.commit()
```

- [ ] **Step 2: Run the schema test**

Run: `cd apps/api && python -m pytest tests/db/test_schema.py -v`
Expected: FAIL because models and fixtures do not exist.

- [ ] **Step 3: Implement models and migration**

Define explicit UTC timestamps, enum columns, JSON evidence fields, foreign keys, cascade behavior, and indexes for device/time, batch/product, status, alert state, and model version.

- [ ] **Step 4: Apply migration to a disposable database**

Run: `cd apps/api && alembic upgrade head`
Expected: all eight tables and indexes are created.

- [ ] **Step 5: Run schema tests and commit**

Run: `cd apps/api && python -m pytest tests/db -v`
Expected: PASS.

```bash
git add apps/api/app/db.py apps/api/app/models apps/api/migrations apps/api/tests/db
git commit -m "feat: add AOI event and governance schema"
```

### Task 4: Ingestion State Machine and Idempotency

**Files:**
- Create: `apps/api/app/schemas/inspection.py`
- Create: `apps/api/app/repositories/inspection.py`
- Create: `apps/api/app/services/ingestion.py`
- Create: `apps/api/app/api/routes/inspection.py`
- Test: `apps/api/tests/services/test_ingestion.py`
- Test: `apps/api/tests/api/test_ingestion_api.py`

**Interfaces:**
- Consumes: `generate_source_key_hash`, SQLAlchemy models
- Produces: `ingest_event(payload: InspectionIn) -> InspectionOut`
- Produces: `POST /api/v1/inspections`

- [ ] **Step 1: Write failing tests for duplicate and incomplete inputs**

```python
def test_duplicate_payload_returns_same_event_uuid(service, payload):
    first = service.ingest_event(payload)
    second = service.ingest_event(payload)
    assert first.event_uuid == second.event_uuid

def test_missing_required_light_stays_collecting(service, payload):
    payload.attachments = [a for a in payload.attachments if a.light_id != "RING"]
    result = service.ingest_event(payload)
    assert result.association_status == "COLLECTING"
```

- [ ] **Step 2: Run ingestion tests**

Run: `cd apps/api && python -m pytest tests/services/test_ingestion.py -v`
Expected: FAIL because the service is absent.

- [ ] **Step 3: Implement transactional upsert and readiness checks**

The service must lock or upsert by `source_key_hash`, deduplicate attachments by their unique tuple, recompute `received_light_set`, and enter `READY` only when all required lights and required 3D/AOI inputs are present.

- [ ] **Step 4: Expose and test the API**

Run: `cd apps/api && python -m pytest tests/services/test_ingestion.py tests/api/test_ingestion_api.py -v`
Expected: PASS for new, duplicate, collecting, ready, and quarantined cases.

- [ ] **Step 5: Commit ingestion behavior**

```bash
git add apps/api/app/schemas apps/api/app/repositories apps/api/app/services/ingestion.py apps/api/app/api/routes/inspection.py apps/api/tests
git commit -m "feat: ingest AOI events idempotently"
```

### Task 5: Inference Adapter and Three-State Decision Service

**Files:**
- Create: `apps/api/app/inference/base.py`
- Create: `apps/api/app/inference/demo.py`
- Create: `apps/api/app/inference/tensorrt.py`
- Create: `apps/api/app/services/decision.py`
- Create: `apps/api/app/services/orchestration.py`
- Modify: `apps/api/app/services/ingestion.py`
- Test: `apps/api/tests/inference/test_adapters.py`
- Test: `apps/api/tests/services/test_decision.py`
- Test: `apps/api/tests/services/test_orchestration.py`

**Interfaces:**
- Produces: `InferenceAdapter.predict(request: InferenceRequest) -> InferenceEvidence`
- Produces: `DemoInferenceAdapter`
- Produces: `TensorRtInferenceAdapter(engine_path: Path)` that fails closed when the engine is unavailable
- Produces: `decide(evidence: InferenceEvidence, policy: DecisionPolicy) -> DecisionResult`
- Produces: `process_ready_event(event_uuid: UUID) -> InspectionOut`; it persists one versioned `InferenceResult` and advances `VALIDATED -> INFERRED` in one transaction

- [ ] **Step 1: Write adapter and safety tests**

```python
def test_missing_tensorrt_engine_fails_closed(tmp_path):
    adapter = TensorRtInferenceAdapter(tmp_path / "missing.engine")
    with pytest.raises(InferenceUnavailable):
        adapter.predict(sample_request())

def test_incomplete_input_is_review_even_with_high_normal_confidence():
    result = decide(evidence(normal_confidence=0.999, input_complete=False), default_policy())
    assert result.decision == Decision.REVIEW
```

- [ ] **Step 2: Run the tests**

Run: `cd apps/api && python -m pytest tests/inference tests/services/test_decision.py tests/services/test_orchestration.py -v`
Expected: FAIL because adapters and decision service are absent.

- [ ] **Step 3: Implement the adapter protocol and deterministic demo inference**

`DemoInferenceAdapter` converts simulator evidence into normalized boxes, defect scores, 3D measures, model version, latency, and input fingerprint. `TensorRtInferenceAdapter` owns engine loading and response normalization but does not fabricate output when CUDA, TensorRT, or the engine file is missing. Add `orchestration.process_ready_event()` so a READY event is validated, inferred, decided, and persisted; inference unavailability must persist REVIEW with reason `INFERENCE_UNAVAILABLE`, never PASS.

- [ ] **Step 4: Implement ordered three-state rules**

Apply rules in this order: identity/input anomaly -> REVIEW; 3D hard-rule or strong defect evidence -> FAIL; complete, calibrated, policy-allowed normal evidence -> PASS; all remaining cases -> REVIEW.

- [ ] **Step 5: Run tests and commit**

Run: `cd apps/api && python -m pytest tests/inference tests/services/test_decision.py tests/services/test_orchestration.py -v`
Expected: PASS.

```bash
git add apps/api/app/inference apps/api/app/services/decision.py apps/api/app/services/orchestration.py apps/api/app/services/ingestion.py apps/api/tests/inference apps/api/tests/services/test_decision.py apps/api/tests/services/test_orchestration.py
git commit -m "feat: add inference adapter and safe decision service"
```

### Task 6: PIS-IN Input Adapter and AOI Simulator Service

**Files:**
- Create: `apps/api/app/adapters/base.py`
- Create: `apps/api/app/adapters/demo.py`
- Create: `apps/api/app/adapters/pis_in.py`
- Modify: `apps/api/app/api/routes/inspection.py`
- Test: `apps/api/tests/adapters/test_pis_in.py`
- Test: `apps/api/tests/api/test_pis_in_import.py`
- Create: `services/simulator/pyproject.toml`
- Create: `services/simulator/simulator/scenarios.py`
- Create: `services/simulator/simulator/main.py`
- Test: `services/simulator/tests/test_scenarios.py`

**Interfaces:**
- Produces: `InspectionSourceAdapter.normalize(raw: Mapping[str, object]) -> InspectionIn`
- Produces: `DemoSourceAdapter` and `PisInSourceAdapter`
- Produces internal real-mode endpoint: `POST /api/v1/inspections/import/pis-in`; return 404 in demo mode
- Produces: `build_scenario(seed: int, scenario: ScenarioKind) -> dict`
- Posts payloads to: `POST /api/v1/inspections`
- Scenarios: `NORMAL`, `DEFECT`, `REVIEW`, `MISSING_3D`, `MISSING_LIGHT`, `STATION_SPIKE`, `DUPLICATE`, `OUT_OF_ORDER`

- [ ] **Step 1: Write deterministic scenario tests**

```python
def test_same_seed_builds_same_source_identity():
    assert build_scenario(42, ScenarioKind.DEFECT)["inspection_sequence"] == build_scenario(42, ScenarioKind.DEFECT)["inspection_sequence"]

def test_pis_in_adapter_keeps_light_id_out_of_source_identity(raw_pis_in_record):
    normalized = PisInSourceAdapter().normalize(raw_pis_in_record)
    assert normalized.device_session_id
    assert {item.light_id for item in normalized.attachments} == {"R", "G", "B", "RING"}
```

- [ ] **Step 2: Run adapter and simulator tests**

Run:

```bash
cd apps/api && python -m pytest tests/adapters tests/api/test_pis_in_import.py -v
cd ../../services/simulator && python -m pytest -v
```

Expected: FAIL because source adapters and scenarios are missing.

- [ ] **Step 3: Implement seeded payload generation**

Implement one normalized `InspectionIn` contract. `PisInSourceAdapter` maps configured filename/manifest fields, rejects missing identity into quarantine, derives `device_session_id` from the native boot/session identifier, and treats light ID only as attachment metadata. The internal import route must require `APP_MODE=real` plus the configured adapter token before normalization and ingestion. Generate stable device/session/tray/slot fields, selected R/G/B/RING attachments, 3D values, defect boxes, model confidence, and reason codes. Do not create actual image binaries; use deterministic placeholder image URLs served by the API.

- [ ] **Step 4: Add interval and burst controls**

Support `SIM_INTERVAL_MS`, `SIM_SEED`, `SIM_SCENARIO_MIX`, and `SIM_BURST_SIZE` environment variables.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
cd apps/api && python -m pytest tests/adapters tests/api/test_pis_in_import.py -v
cd ../../services/simulator && python -m pytest -v
```

Expected: PASS.

```bash
git add apps/api/app/adapters apps/api/app/api/routes/inspection.py apps/api/tests/adapters apps/api/tests/api/test_pis_in_import.py services/simulator
git commit -m "feat: add PIS-IN adapter and AOI simulator"
```

### Task 7: Dashboard, Tray, Review, Alert, Report, and Model APIs

**Files:**
- Create: `apps/api/app/schemas/dashboard.py`
- Create: `apps/api/app/services/query.py`
- Create: `apps/api/app/services/review.py`
- Create: `apps/api/app/services/alerts.py`
- Create: `apps/api/app/services/reports.py`
- Create: `apps/api/app/api/routes/dashboard.py`
- Create: `apps/api/app/api/routes/review.py`
- Create: `apps/api/app/api/routes/alerts.py`
- Create: `apps/api/app/api/routes/reports.py`
- Create: `apps/api/app/api/routes/models.py`
- Create: `apps/api/app/api/routes/demo.py`
- Test: `apps/api/tests/api/test_business_api.py`

**Interfaces:**
- Produces: `GET /api/v1/dashboard/summary`, `GET /api/v1/inspections`, `GET /api/v1/inspections/{event_uuid}`, `GET /api/v1/trays/{tray_id}`
- Produces: `GET /api/v1/reviews`, `POST /api/v1/reviews`
- Produces: `GET /api/v1/alerts`, `POST /api/v1/alerts/{alert_id}/acknowledge`, `POST /api/v1/alerts/{alert_id}/close`
- Produces: `GET /api/v1/reports`, `POST /api/v1/reports`, `GET /api/v1/reports/{report_id}`
- Produces: `GET /api/v1/model-releases`, `GET /api/v1/project-profile`
- Produces demo-only: `POST /api/v1/demo/reset` with `{seed: int}`; return 404 when `APP_MODE != demo`
- Review command: `ReviewDecisionIn(event_uuid, decision, defect_code, comment, reviewer)`

- [ ] **Step 1: Write endpoint contract tests**

```python
def test_review_changes_queue_and_persists(client, seeded_review_event):
    response = client.post("/api/v1/reviews", json={
        "event_uuid": seeded_review_event,
        "decision": "FAIL",
        "defect_code": "BALL_BRIDGE",
        "comment": "confirmed",
        "reviewer": "qa_demo",
    })
    assert response.status_code == 201
    assert response.json()["golden_status"] == "CONFIRMED"

def test_acknowledged_alert_can_create_report(client, seeded_alert):
    client.post(f"/api/v1/alerts/{seeded_alert}/acknowledge", json={"operator": "qa_demo"})
    response = client.post("/api/v1/reports", json={"alert_id": seeded_alert})
    assert response.status_code == 201
    assert response.json()["status"] == "DRAFT"
```

- [ ] **Step 2: Run the contract tests**

Run: `cd apps/api && python -m pytest tests/api/test_business_api.py -v`
Expected: FAIL because endpoints are absent.

- [ ] **Step 3: Implement filterable query and command services**

Filters must include time range, product, batch, device, station, defect code, decision, and model version. Alert rules require a minimum sample count, rolling window, consecutive abnormal windows, acknowledgement metadata, and recovery state. Report creation must persist the alert snapshot, representative event IDs, metric window, model/policy versions, observed facts, and open questions. The demo reset endpoint must truncate only demo-owned business rows and reseed deterministic scenarios in one transaction.

- [ ] **Step 4: Run API tests**

Run: `cd apps/api && python -m pytest tests/api -v`
Expected: PASS.

- [ ] **Step 5: Commit business APIs**

```bash
git add apps/api/app/schemas apps/api/app/services apps/api/app/api/routes apps/api/tests/api
git commit -m "feat: expose AOI operations APIs"
```

### Task 8: React Shell, API Types, and Visual Tokens

**Files:**
- Create: `apps/web/package.json`
- Create: `apps/web/vite.config.ts`
- Create: `apps/web/src/main.tsx`
- Create: `apps/web/src/app/App.tsx`
- Create: `apps/web/src/api/client.ts`
- Create: `apps/web/src/api/types.ts`
- Create: `apps/web/src/layout/AppShell.tsx`
- Create: `apps/web/src/styles/tokens.css`
- Test: `apps/web/src/layout/AppShell.test.tsx`

**Interfaces:**
- Produces typed API client methods matching Task 7 endpoints
- Produces stable desktop/mobile navigation for eight pages

- [ ] **Step 1: Write the failing navigation test**

```tsx
it("renders all eight operational destinations", () => {
  render(<AppShell />)
  for (const label of ["总览", "实时检测", "Tray Map", "人工复核", "缺陷报表", "预警与报告", "模型治理", "项目说明"]) {
    expect(screen.getByText(label)).toBeInTheDocument()
  }
})
```

- [ ] **Step 2: Run the frontend test**

Run: `cd apps/web && npm test -- --run`
Expected: FAIL before the shell exists.

- [ ] **Step 3: Implement the shell and tokens**

Use a restrained industrial palette, 8px-or-less radii, Lucide icons, fixed toolbar dimensions, responsive grid tracks, and zero decorative gradient orbs. Do not display budget fields in routes, types, fixtures, or copy.

- [ ] **Step 4: Run unit tests and type checking**

Run: `cd apps/web && npm test -- --run && npm run typecheck`
Expected: PASS.

- [ ] **Step 5: Commit the frontend foundation**

```bash
git add apps/web
git commit -m "feat: scaffold AOI operations frontend"
```

### Task 9: Operational Pages and Interactions

**Files:**
- Create: `apps/web/src/pages/DashboardPage.tsx`
- Create: `apps/web/src/pages/InspectionPage.tsx`
- Create: `apps/web/src/pages/TrayMapPage.tsx`
- Create: `apps/web/src/pages/ReviewPage.tsx`
- Create: `apps/web/src/pages/ReportsPage.tsx`
- Create: `apps/web/src/pages/AlertsPage.tsx`
- Create: `apps/web/src/pages/ModelsPage.tsx`
- Create: `apps/web/src/pages/ProjectPage.tsx`
- Create: `apps/web/src/components/StatusBadge.tsx`
- Create: `apps/web/src/components/TrayGrid.tsx`
- Create: `apps/web/src/components/EventEvidence.tsx`
- Test: `apps/web/src/pages/ReviewPage.test.tsx`
- Test: `apps/web/src/pages/TrayMapPage.test.tsx`

**Interfaces:**
- Consumes: typed API client from Task 8
- Produces: clickable tray slots, event evidence detail, persisted review command, alert acknowledgement/closure, report creation/view, model shadow comparison

- [ ] **Step 1: Write review and Tray Map interaction tests**

```tsx
it("submits a review decision and removes the event from the queue", async () => {
  render(<ReviewPage />)
  await user.click(await screen.findByRole("button", {name: "确认缺陷"}))
  expect(api.createReview).toHaveBeenCalledWith(expect.objectContaining({decision: "FAIL"}))
})
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `cd apps/web && npm test -- --run`
Expected: FAIL because page components are absent.

- [ ] **Step 3: Implement all eight pages**

Use ECharts for time series and defect distribution. Use CSS grid for Tray Map so cells have stable dimensions. Use real API loading, empty, error, stale, and retry states. The project page may show team, period, architecture, compute estimates, and personal contributions, but not budget.

- [ ] **Step 4: Run frontend tests and type checking**

Run: `cd apps/web && npm test -- --run && npm run typecheck`
Expected: PASS.

- [ ] **Step 5: Commit operational pages**

```bash
git add apps/web/src
git commit -m "feat: add AOI dashboard and review workflows"
```

### Task 10: GPU-Free Docker Demo and End-to-End Tests

**Files:**
- Create: `apps/api/Dockerfile`
- Create: `services/simulator/Dockerfile`
- Create: `apps/web/Dockerfile`
- Create: `infra/docker-compose.yml`
- Create: `infra/nginx.conf`
- Create: `apps/web/playwright.config.ts`
- Create: `apps/web/tests/e2e/operations.spec.ts`
- Create: `.env.example`

**Interfaces:**
- Produces: `docker compose -f infra/docker-compose.yml up --build`
- Exposes: frontend `http://localhost:8080`, API `http://localhost:8080/api/v1`

- [ ] **Step 1: Write the failing Playwright journey**

```ts
test("simulated event can be reviewed, traced, and reported", async ({page}) => {
  await page.goto("http://localhost:8080")
  await expect(page.getByText("实时检测")).toBeVisible()
  await page.getByRole("link", {name: "人工复核"}).click()
  await page.getByRole("button", {name: "确认缺陷"}).first().click()
  await page.getByRole("link", {name: "Tray Map"}).click()
  await expect(page.locator("[data-status='FAIL']").first()).toBeVisible()
  await page.getByRole("link", {name: "预警与报告"}).click()
  await page.getByRole("button", {name: "确认预警"}).first().click()
  await page.getByRole("button", {name: "生成异常报告"}).first().click()
  await expect(page.getByText("DRAFT")).toBeVisible()
})
```

- [ ] **Step 2: Run Compose and observe the initial failure**

Run: `docker compose -f infra/docker-compose.yml up --build`
Expected: FAIL until Dockerfiles and service wiring are complete.

- [ ] **Step 3: Implement Compose, health checks, and Nginx routing**

Configure PostgreSQL health checks, API migration-before-start, simulator dependency on API health, frontend static serving, named database volume, and restart policies. Do not include a GPU reservation in this foundation compose file.

- [ ] **Step 4: Run all foundation verification**

Run:

```bash
cd apps/api && python -m pytest -q
cd ../../services/simulator && python -m pytest -q
cd ../../apps/web && npm test -- --run && npm run typecheck
docker compose -f ../../infra/docker-compose.yml up -d --build
npx playwright test
```

Expected: all tests PASS; dashboard receives events; review persists across API restart.

- [ ] **Step 5: Commit the runnable demo foundation**

```bash
git add apps services/simulator infra .env.example
git commit -m "feat: deliver runnable AOI demo foundation"
```
