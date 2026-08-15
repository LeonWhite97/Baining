# Handler AOI Phase 1 Safety Foundation Implementation Plan

> **项目决策更新（2026-08-10）：** 自动化 Handler 集成暂停，现阶段采用 AOI 单机操作。本计划已完成的代码作为可选扩展保留，默认通过 `HANDLER_INTEGRATION_ENABLED=false` 关闭，不再作为当前上线前置条件。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the database, runtime-mode, strict START ingestion, single-Site idempotency, and capture-identity foundation required by Handler-AOI V1.4 without claiming real camera or TensorRT integration.

**Architecture:** Keep the existing FastAPI modular monolith and PostgreSQL boundary. Introduce focused Handler cycle models and domain services; the future native TCP Gateway will translate protocol messages into the internal AOI START API. Camera and TensorRT remain injected adapter boundaries and fail closed until field dependencies are supplied.

**Tech Stack:** Python 3.12+, FastAPI, Pydantic v2, SQLAlchemy 2, Alembic, PostgreSQL/SQLite tests, pytest.

## Implementation Status (2026-08-10)

Phase 1 software foundation is implemented and verified. The API suite passes 62 tests; PostgreSQL Compose is migrated to `0004_active_cycle_guard`; API, PostgreSQL, frontend, Agent/RAG, and simulator services are running; the frontend returns HTTP 200 at `http://localhost:8080`.

Implemented scope includes strict shadow/controlled START ingestion, durable idempotency, global TraceID uniqueness, database-enforced single-Site active-cycle exclusion, capture identity and stale-frame validation, RESULT/BIN consistency validation, Handler/MES Outbox persistence, runtime-mode isolation, and configurable Compose `APP_MODE` with demo as the default.

Implementation also corrected two deployment issues discovered during PostgreSQL verification: Alembic revision identifiers exceeded the default 32-character version column, and application-only active-cycle checks did not close the concurrent START race. Regression tests now cover both conditions.

The following remain external integration gates and are not claimed complete: PIS-IN camera SDK and driver, SDK-issued trigger sequence behavior, physical R/G/B/RING timing, TensorRT Engine/CUDA/GPU runtime, Handler TCP endpoint and SESSION_SYNC/HMAC capability, final Handler BIN acceptance table, MES endpoint contract, and field fault-injection/throughput evidence.

## Global Constraints

- Preserve the current FastAPI, PostgreSQL, React, Docker Compose, YOLOv8, TensorRT, and Agent/RAG architecture.
- Runtime modes are exactly `demo`, `shadow`, and `controlled`; automatic PASS defaults off.
- Production START payloads forbid unknown fields and never accept `Scenario` or `scenario`.
- A TraceID is globally unique; one Handler has at most one active AOI cycle.
- START must be durable before it is acknowledged.
- Camera SDK, TensorRT Engine, CUDA runtime, and field throughput evidence are external dependencies and must not be simulated as production evidence.

---

### Task 1: Extend the persistence model for Handler cycles and Outbox state

**Files:**
- Modify: `apps/api/app/models/inspection.py`
- Modify: `apps/api/app/models/__init__.py`
- Create: `apps/api/migrations/versions/0002_handler_aoi_safety_foundation.py`
- Test: `apps/api/tests/test_handler_models.py`

**Interfaces:**
- Produces: nullable START-stage `InspectionEvent` fields, `HandlerResultOutbox`, and `MesOutbox` ORM models.
- Produces: global `UNIQUE(trace_id)` and `UNIQUE(handler_id, handler_session_id, cycle_id)` constraints.

- [ ] **Step 1: Write a failing model test**

```python
def test_start_stage_event_allows_missing_inference_fields(session):
    event = InspectionEvent(
        event_uuid="evt-start",
        source_key_hash="a" * 64,
        trace_id="TRACE-1",
        handler_id="HANDLER-1",
        handler_session_id="SESSION-1",
        cycle_id="CYCLE-1",
        device_id="HANDLER-1",
        device_session_id="SESSION-1",
        inspection_sequence="CYCLE-1",
        product_id="BGA-256",
        batch_id="LOT-1",
        tray_id="TRAY-1",
        slot_index="01",
        station="AOI",
        surface="TOP",
        association_status="START_RECEIVED",
    )
    session.add(event)
    session.commit()
    assert event.ai_decision is None
```

- [ ] **Step 2: Run the model test and verify it fails because the new fields and nullable columns do not exist**

Run: `pytest tests/test_handler_models.py -v` from `apps/api`.

- [ ] **Step 3: Add the ORM fields and Outbox tables**

```python
trace_id: Mapped[str | None] = mapped_column(String(128), unique=True, index=True)
handler_id: Mapped[str | None] = mapped_column(String(64), index=True)
handler_session_id: Mapped[str | None] = mapped_column(String(64))
cycle_id: Mapped[str | None] = mapped_column(String(64))
capture_id: Mapped[str | None] = mapped_column(String(36), unique=True)
aoi_bin: Mapped[int | None] = mapped_column(Integer)
result_category: Mapped[str | None] = mapped_column(String(16))
handler_publish_status: Mapped[str] = mapped_column(String(32), default="NOT_READY")
mes_publish_status: Mapped[str] = mapped_column(String(32), default="NOT_READY")
```

Make `ai_decision`, `ai_confidence`, `reason_code`, and `image_url` nullable. Add a deterministic Alembic migration rather than calling metadata creation inside the migration.

- [ ] **Step 4: Run the model test and the existing API suite**

Run: `pytest tests/test_handler_models.py tests/test_operations_api.py -v`.

### Task 2: Enforce runtime modes and strict production START schemas

**Files:**
- Create: `apps/api/app/config.py`
- Create: `apps/api/app/schemas/handler.py`
- Modify: `apps/api/app/main.py`
- Test: `apps/api/tests/test_runtime_mode.py`
- Test: `apps/api/tests/test_handler_start_api.py`

**Interfaces:**
- Produces: `RuntimeMode`, `RuntimeSettings`, and `AOIStartRequest` with `extra="forbid"`.
- Produces: `POST /api/v1/inspections/aoi/start` available only in shadow/controlled mode.

- [ ] **Step 1: Write failing mode and schema tests**

```python
def test_shadow_start_rejects_scenario_field(shadow_client):
    payload = valid_start_payload()
    payload["scenario"] = "NORMAL"
    assert shadow_client.post("/api/v1/inspections/aoi/start", json=payload).status_code == 422

def test_unknown_runtime_mode_fails_app_creation():
    with pytest.raises(ValueError, match="Unsupported APP_MODE"):
        create_app(database_url="sqlite+pysqlite:///:memory:", mode="production")
```

- [ ] **Step 2: Run the tests and verify they fail because the endpoint and runtime settings do not exist**

Run: `pytest tests/test_runtime_mode.py tests/test_handler_start_api.py -v`.

- [ ] **Step 3: Implement strict settings and request schema**

```python
class AOIStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trace_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,128}$")
    handler_id: str = Field(min_length=1, max_length=64)
    handler_session_id: str = Field(min_length=1, max_length=64)
    cycle_id: str = Field(min_length=1, max_length=64)
    station_code: Literal["AOI"] = "AOI"
    product_id: str
    batch_id: str
    tray_id: str
    slot_index: str
    surface: Literal["TOP", "BOTTOM"] = "TOP"
```

Set `AUTO_PASS_ENABLED` to false by default. Demo routes remain available only in demo mode.

- [ ] **Step 4: Run the new tests and the existing health/demo tests**

Run: `pytest tests/test_runtime_mode.py tests/test_handler_start_api.py tests/test_health.py tests/test_operations_api.py -v`.

### Task 3: Implement single-Site durable and idempotent START handling

**Files:**
- Create: `apps/api/app/domain/handler_cycle.py`
- Create: `apps/api/app/services/handler_cycles.py`
- Create: `apps/api/app/api/routes/handler.py`
- Modify: `apps/api/app/main.py`
- Test: `apps/api/tests/test_handler_cycles.py`
- Test: `apps/api/tests/test_handler_start_api.py`

**Interfaces:**
- Produces: `create_or_get_cycle(session, request) -> CycleStartResult`.
- Produces: HTTP 201 for a new cycle, 200 for an identical retry, and 409 for evidence conflict or an active different TraceID.

- [ ] **Step 1: Write failing idempotency and single-Site tests**

```python
def test_identical_start_is_idempotent(shadow_client):
    first = shadow_client.post(PATH, json=valid_start_payload())
    second = shadow_client.post(PATH, json=valid_start_payload())
    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json()["event_uuid"] == second.json()["event_uuid"]

def test_different_trace_is_rejected_while_cycle_active(shadow_client):
    shadow_client.post(PATH, json=valid_start_payload(trace_id="TRACE-1"))
    response = shadow_client.post(PATH, json=valid_start_payload(trace_id="TRACE-2", cycle_id="CYCLE-2"))
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "ACTIVE_CYCLE_EXISTS"
```

- [ ] **Step 2: Run tests and confirm both fail for the missing service**

Run: `pytest tests/test_handler_cycles.py tests/test_handler_start_api.py -v`.

- [ ] **Step 3: Implement canonical SourceKey and durable creation**

```python
def handler_source_key(payload: AOIStartRequest) -> str:
    identity = {
        "cycle_id": payload.cycle_id,
        "handler_id": payload.handler_id,
        "handler_session_id": payload.handler_session_id,
        "station_code": payload.station_code,
        "surface": payload.surface,
        "trace_id": payload.trace_id,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

Persist the cycle and commit before returning the response. Use database unique constraints as the final concurrency authority and translate integrity conflicts into stable 409 responses.

- [ ] **Step 4: Run the focused and full API suites**

Run: `pytest tests/test_handler_cycles.py tests/test_handler_start_api.py tests/test_operations_api.py -v`.

### Task 4: Add capture identity and stale-frame validation

**Files:**
- Create: `apps/api/app/domain/capture.py`
- Test: `apps/api/tests/test_capture_identity.py`

**Interfaces:**
- Produces: `CaptureAttachment`, `CaptureBatch`, and `validate_capture_batch(batch, required_lights)`.
- Rejects mismatched capture IDs, trigger sequences, stale monotonic times, duplicate lights, and incomplete light sets.

- [ ] **Step 1: Write failing stale-frame and mismatch tests**

```python
def test_capture_rejects_frame_started_before_start_event():
    batch = make_batch(start_received_monotonic_ns=200, capture_started_monotonic_ns=199)
    with pytest.raises(CaptureValidationError, match="STALE_CAPTURE"):
        validate_capture_batch(batch, {"R", "G", "B", "RING"})

def test_capture_rejects_mixed_trigger_sequences():
    batch = make_batch(trigger_sequences=[10, 10, 9, 10])
    with pytest.raises(CaptureValidationError, match="TRIGGER_SEQUENCE_MISMATCH"):
        validate_capture_batch(batch, {"R", "G", "B", "RING"})
```

- [ ] **Step 2: Run and verify failure due to missing capture domain module**

Run: `pytest tests/test_capture_identity.py -v`.

- [ ] **Step 3: Implement immutable capture value objects and validation**

```python
@dataclass(frozen=True, slots=True)
class CaptureAttachment:
    light_id: str
    capture_id: str
    camera_trigger_sequence: int
    file_path: str
    file_hash: str

def validate_capture_batch(batch: CaptureBatch, required_lights: set[str]) -> None:
    if batch.capture_started_monotonic_ns < batch.start_received_monotonic_ns:
        raise CaptureValidationError("STALE_CAPTURE")
```

- [ ] **Step 4: Run capture tests**

Run: `pytest tests/test_capture_identity.py -v`.

### Task 5: Add result categories and Outbox-ready transitions

**Files:**
- Modify: `apps/api/app/domain/enums.py`
- Create: `apps/api/app/domain/handler_result.py`
- Test: `apps/api/tests/test_handler_result.py`

**Interfaces:**
- Produces: `ResultCategory`, `PublishStatus`, and `validate_handler_result()`.
- Enforces PASS=BIN100/QUALITY, defect FAIL=BIN201-205/QUALITY, review=BIN280/QUALITY, and system review=BIN290-293/299/SYSTEM.

- [ ] **Step 1: Write failing consistency tests**

```python
def test_system_bin_cannot_be_pass():
    with pytest.raises(ValueError, match="SYSTEM_BIN_REQUIRES_REVIEW"):
        validate_handler_result("PASS", "SYSTEM", 292, False, None)

def test_pass_is_only_bin_100():
    validate_handler_result("PASS", "QUALITY", 100, False, None)
```

- [ ] **Step 2: Run and verify failure due to missing validator**

Run: `pytest tests/test_handler_result.py -v`.

- [ ] **Step 3: Implement the minimal result matrix**

Implement exact enum and BIN validation without adding field-device behavior.

- [ ] **Step 4: Run the domain suite**

Run: `pytest tests/test_handler_result.py tests/test_decision.py tests/test_domain.py -v`.

### Task 6: Regression verification and Phase 1 handoff

**Files:**
- Modify: `.env.example`
- Modify: `docs/Handler上位机-AOI系统接口规范_V1.4_联调基线版.md` only if implementation reveals a contradiction.

**Interfaces:**
- Produces: verified Phase 1 foundation while keeping demo behavior isolated and functional.

- [ ] **Step 1: Run the complete API test suite**

Run: `pytest -q` from `apps/api`.

- [ ] **Step 2: Run Alembic migration against PostgreSQL Compose**

Run: `docker compose -f infra/docker-compose.yml run --rm api alembic upgrade head`.

- [ ] **Step 3: Rebuild Compose and verify health**

Run: `docker compose -f infra/docker-compose.yml up -d --build`.

- [ ] **Step 4: Verify the shadow START endpoint rejects Scenario and persists an empty-inference cycle**

Expected: Scenario request returns 422; valid request returns 201 with `association_status=START_RECEIVED`; no PASS is generated.

- [ ] **Step 5: Record external blockers**

Document the missing PIS-IN SDK, trigger-sequence support, TensorRT Engine/CUDA environment, Handler IP/port/timeouts, HMAC capability, and MES contract. Do not mark hardware integration complete without those artifacts.
