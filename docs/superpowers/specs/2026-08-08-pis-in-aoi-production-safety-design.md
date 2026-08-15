# PIS-IN AOI V3.5 Production Safety Hardening Design

**Date:** 2026-08-08  
**Status:** Awaiting written-spec approval  
**Scope:** Production safety gates, trusted ingestion, controlled rollout, and executable acceptance  
**Architecture constraint:** Keep FastAPI, PostgreSQL, React, Docker Compose, Agent/RAG, YOLOv8, and TensorRT. Agent/RAG remains outside the real-time automatic PASS path.

## 1. Goal

Eliminate paths that can produce an automatic PASS from caller-controlled or incomplete evidence. Deliver two independently verifiable stages:

1. Stage 1: trusted data association, fail-closed decision gates, API authentication, immutable AI/human decisions, and shadow-mode deployment.
2. Stage 2: fault injection, blind-test evidence ingestion, release gates, and controlled automatic PASS activation.

This work makes the software capable of production acceptance. It does not claim that real factory blind testing or continuous-run targets have passed without real factory data and continuous-run evidence.

## 2. Considered Approaches

### 2.1 Recommended: harden the existing modular monolith

Keep the current FastAPI and PostgreSQL service boundary. Add focused ingestion, authorization, decision-gate, and release-gate modules. Use PostgreSQL transactions and unique constraints for deterministic association.

Advantages: minimal architecture change, testable in the current Compose environment, and compatible with the existing TensorRT adapter boundary. Limitation: sustained high throughput will eventually require an external durable queue, but that is not required for this safety baseline.

### 2.2 Alternative: introduce a message broker now

Add Kafka or RabbitMQ between ingestion and inference. This improves buffering and replay, but changes the approved stack and materially expands deployment and failure modes. It is deferred until measured throughput proves PostgreSQL-backed coordination insufficient.

### 2.3 Alternative: keep demo ingestion and only raise confidence thresholds

This does not solve caller-controlled scenarios, duplicate-light acceptance, late-file loss, or identity conflicts. It is rejected because model thresholds cannot repair untrusted evidence.

## 3. Safety Invariants

The implementation must enforce all of the following:

- `Scenario` is never accepted as production inference evidence.
- Automatic PASS is impossible unless the event is `VALIDATED`.
- `VALIDATED` requires complete identity, exact required light set, unique light IDs, accepted file types, valid SHA256 format, and a stable input fingerprint.
- Missing, duplicate, conflicting, unknown, expired, or inference-unavailable inputs resolve to REVIEW or quarantine, never PASS.
- Agent/RAG output cannot change the real-time PASS/FAIL/REVIEW decision.
- A Source Key identifies one physical inspection. Repeated identical evidence is idempotent; new non-conflicting attachments are merged; conflicting evidence is quarantined.
- AI decisions are immutable inference evidence. Human review produces separate human and final decisions.
- Automatic PASS remains disabled until a release gate has passed and an authorized administrator explicitly enables controlled rollout.

## 4. Stage 1 Design

### 4.1 Runtime modes

Add explicit modes:

- `demo`: simulator and deterministic demo inference are allowed; no production claims.
- `shadow`: trusted PIS-IN ingestion and real adapter boundary run, but final output is REVIEW for otherwise-passable parts.
- `controlled`: automatic PASS is allowed only when the active release gate is approved.

`AUTO_PASS_ENABLED` defaults to `false`. `APP_MODE=controlled` without an approved active release must still fail closed.

### 4.2 Authentication and authorization

Use two independent credentials without adding a new identity platform:

- Device ingestion: `X-Device-Token`, compared with `PIS_IN_DEVICE_TOKEN` using constant-time comparison.
- Human/API commands: bearer token mapped to roles by environment configuration.

Roles:

- `viewer`: read-only dashboards and evidence.
- `reviewer`: create human review records.
- `quality_manager`: acknowledge/close alerts and create reports.
- `admin`: change controlled-rollout state and release approvals.

Health checks remain unauthenticated. Demo reset requires demo mode plus admin authentication.

### 4.3 Trusted attachment validation

The adapter validates metadata only; a dedicated validator applies the production contract:

- Allowed lights: `R`, `G`, `B`, `RING`, `IR`, `UV`.
- Required lights are configured per product/station; the default is exactly `R,G,B,RING`.
- Each required light appears exactly once for readiness.
- SHA256 must be 64 lowercase hexadecimal characters.
- File paths must use an allowed scheme/prefix.
- The canonical input fingerprint is SHA256 over sorted `(data_type, light_id, file_hash)` tuples.

Because this workspace has no real shared AOI image directory, physical file existence, stable-write probing, image decoding, and hash recomputation are represented by a pluggable `AttachmentVerifier`. The production verifier must be configured before controlled mode; metadata-only verification is permitted only in demo/shadow mode and cannot authorize PASS.

### 4.4 Transactional association and late arrivals

Ingestion uses one transaction:

1. Normalize identity and compute `source_key_hash`.
2. Lock the existing event by Source Key or create it with `ON CONFLICT` recovery.
3. Merge new attachments idempotently.
4. Detect conflicting hashes for the same `(event_uuid, data_type, light_id)` and quarantine the new evidence.
5. Recompute received lights and input fingerprint.
6. Set `COLLECTING`, `READY`, `VALIDATED`, `REVIEW_REQUIRED`, or `QUARANTINED`.
7. Run inference only for `VALIDATED` events.

Late attachments update an existing `COLLECTING` event. Existing events are never returned early before merge and conflict checks.

### 4.5 Decision and traceability

Replace production use of `DemoInferenceAdapter` with an adapter selected from configuration. `demo` may use the demo adapter; `shadow` and `controlled` use the TensorRT adapter boundary and fail closed when it is unavailable.

Persist separately:

- `ai_decision`: immutable result for one model/policy/input fingerprint.
- `human_decision`: optional review outcome.
- `final_decision`: externally consumed result.
- `decision_source`: `AI`, `HUMAN`, or `SAFETY_GATE`.

In shadow mode, an AI PASS is stored as AI evidence while the final decision remains REVIEW with reason `SHADOW_MODE`.

### 4.6 Query safety

- Event detail selects an explicit active inference result ordered by creation time and model policy.
- List endpoints use bounded pagination.
- Integrity and database exceptions return stable 409/422 responses instead of unhandled 500 responses.

## 5. Stage 2 Design

### 5.1 Release gate

Add a persisted release assessment containing:

- model version and policy version;
- dataset identifier and immutable dataset fingerprint;
- sample count;
- silent mismatch count and rate;
- incomplete-input PASS count;
- unknown-defect PASS count;
- critical-defect recall;
- false-positive rate;
- P95 latency;
- fault-injection pass/fail summary;
- approver and approval timestamp.

Controlled PASS requires:

- silent mismatch count = 0;
- incomplete-input PASS count = 0;
- unknown-defect PASS count = 0;
- all mandatory fault injections pass;
- dataset fingerprint is present;
- approval by an admin;
- the approved model/policy pair matches the active runtime pair.

The existing quality targets remain reporting targets, not substitutes for the zero-tolerance safety invariants.

### 5.2 Executable fault injection

Provide an automated acceptance runner covering:

- duplicate light;
- missing light followed by late arrival;
- invalid SHA256;
- same Source Key with conflicting evidence;
- concurrent duplicate submissions;
- unknown scenario/data type;
- TensorRT unavailable;
- Agent/RAG unavailable;
- API restart and PostgreSQL persistence;
- unauthorized ingestion/review/alert actions.

The runner emits machine-readable JSON with case status, evidence, and aggregate gate status. It never modifies controlled-rollout approval.

### 5.3 Blind-test evidence

The system accepts a blind-test result manifest produced outside the model-training workflow. The manifest includes dataset ID, dataset fingerprint, labels fingerprint, model/policy versions, metrics, and evaluator identity. The API validates structure and persists it as evidence; it does not fabricate factory metrics.

## 6. Data Model Changes

Extend existing tables rather than replacing the architecture:

- `inspection_events`: `input_fingerprint`, `human_decision`, `final_decision`, `decision_source`, `association_reason`, `updated_at`.
- `data_attachment_links`: enforce one active evidence item per event/data type/light; retain file hash history through quarantine records.
- `inference_results`: retain immutable model/policy/input fingerprint results.
- `release_assessments`: persisted gate metrics and approval.
- `security_audit_events`: actor, role, action, resource, outcome, timestamp, and request correlation ID.

Alembic migration must preserve existing demo data and backfill `final_decision` from the current decision field.

## 7. API Changes

- `POST /api/v1/inspections/import/pis-in`: device-authenticated trusted ingestion; ignores/rejects caller `Scenario` outside demo mode.
- Existing read endpoints: viewer role and bounded pagination.
- `POST /api/v1/reviews`: reviewer role; stores human/final decision without overwriting AI evidence.
- Alert/report commands: quality-manager role.
- `POST /api/v1/release-assessments`: admin role.
- `POST /api/v1/release-assessments/{id}/approve`: admin role, subject to hard gate validation.
- `GET /api/v1/safety/status`: current mode, automatic PASS state, active model/policy, and gate reasons.

## 8. Error Handling

- Authentication failure: 401.
- Authorization failure: 403.
- Invalid attachment contract: 422 and REVIEW/QUARANTINED persistence where identity is known.
- Source evidence conflict: 409 plus quarantine record.
- Inference unavailable/timeout: persisted REVIEW with `INFERENCE_UNAVAILABLE`.
- Database transaction conflict: bounded retry, then 409; no partial event/attachment state.
- Agent/RAG failure: report remains DRAFT with `agent_status=UNAVAILABLE`; real-time inspection is unaffected.

## 9. Test Strategy

All behavior changes follow red-green TDD. Required suites:

- Unit: attachment validation, state transitions, decision gate, role checks, release gate.
- API: authentication, late merge, conflict quarantine, immutable AI decision, pagination.
- Concurrency: many identical submissions produce one event and no unhandled error.
- PostgreSQL integration: foreign keys, upsert behavior, migration, restart persistence.
- Compose acceptance: all containers healthy, shadow mode default, demo reset protected.
- Fault injection: JSON report with every mandatory case passing.

## 10. Acceptance Boundaries

Software implementation is complete when all automated tests, PostgreSQL integration tests, Compose health checks, and simulated fault injections pass.

Production acceptance remains pending until the customer supplies:

- a real PIS-IN field export contract;
- accessible image/3D files for physical verification;
- TensorRT engine and calibration artifacts;
- an independently held-out blind dataset;
- continuous-run duration and throughput target;
- named quality and release approvers.

Until those external conditions are satisfied, the delivered default is `shadow` and automatic PASS remains disabled.
