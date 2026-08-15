# PIS-IN AOI Agent/RAG and Production Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or execute each task inline with test-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated offline Agent/RAG service, production deployment profiles, and operational failure handling without changing the real-time inspection decision path.

**Architecture:** `services/agent-rag` owns pgvector knowledge ingestion, retrieval, and three LangGraph workflows. `apps/api` calls it asynchronously for data-quality summaries, report drafts, and model-release recommendations; inspection ingestion and PASS/FAIL/REVIEW continue when the service is unavailable. Docker Compose profiles separate GPU-free demo, local LLM, and TensorRT deployment modes.

**Tech Stack:** Python 3.12, FastAPI, LangGraph, PostgreSQL 16, pgvector, SQLAlchemy 2, OpenAI-compatible LLM client, Pytest, Docker Compose, NVIDIA Container Runtime, Nginx.

## Global Constraints

- Agent/RAG cannot create an automatic PASS or change a production model without approval.
- Every generated statement must include evidence references or be labelled as an unverified suggestion.
- Use PostgreSQL/pgvector; do not introduce Milvus or another vector database.
- The service must support a deterministic no-LLM demo provider.
- Inspection APIs must remain healthy when Agent/RAG is stopped.
- Production profiles may request GPU resources; the base demo profile must not.

## File Structure

```text
services/agent-rag/
  pyproject.toml
  agent_rag/main.py
  agent_rag/config.py
  agent_rag/db.py
  agent_rag/models.py
  agent_rag/schemas.py
  agent_rag/embeddings.py
  agent_rag/retrieval.py
  agent_rag/llm.py
  agent_rag/workflows/data_quality.py
  agent_rag/workflows/report.py
  agent_rag/workflows/model_governance.py
  agent_rag/api/routes.py
  migrations/versions/0001_knowledge.py
  tests/
apps/api/app/clients/agent_rag.py
apps/api/app/services/agent_jobs.py
infra/docker-compose.yml
infra/docker-compose.gpu.yml
infra/nginx.conf
docs/runbooks/agent-rag-failure.md
```

---

### Task 1: Knowledge Schema, Chunking, and Retrieval

**Files:**
- Create: `services/agent-rag/pyproject.toml`
- Create: `services/agent-rag/alembic.ini`
- Create: `services/agent-rag/agent_rag/config.py`
- Create: `services/agent-rag/agent_rag/db.py`
- Create: `services/agent-rag/agent_rag/models.py`
- Create: `services/agent-rag/agent_rag/embeddings.py`
- Create: `services/agent-rag/agent_rag/retrieval.py`
- Create: `services/agent-rag/migrations/versions/0001_knowledge.py`
- Create: `services/agent-rag/migrations/env.py`
- Test: `services/agent-rag/tests/test_retrieval.py`

**Interfaces:**
- Produces: `KnowledgeChunk(id, document_id, title, text, category, metadata, embedding)`
- Produces: `retrieve(query: str, categories: list[str], limit: int = 5) -> list[Evidence]`
- Categories: `DEFECT_DICTIONARY`, `SOP`, `DEVICE_MANUAL`, `INCIDENT_REPORT`, `MODEL_RELEASE`

- [ ] **Step 1: Write the failing ranked-retrieval test**

```python
def test_retrieve_filters_category_and_preserves_citation(repository):
    results = repository.retrieve("球高超限", ["SOP"], limit=3)
    assert results[0].category == "SOP"
    assert results[0].document_id
    assert results[0].chunk_id
```

- [ ] **Step 2: Run the test**

Run: `cd services/agent-rag && python -m pytest tests/test_retrieval.py -v`
Expected: FAIL because retrieval is missing.

- [ ] **Step 3: Implement pgvector retrieval and deterministic fallback embeddings**

Use cosine distance for production embeddings. In demo mode, generate a stable normalized vector from SHA-256 token buckets so tests and the GPU-free demo do not call an external model.

- [ ] **Step 4: Apply migration and run tests**

Run: `cd services/agent-rag && alembic upgrade head && python -m pytest tests/test_retrieval.py -v`
Expected: PASS.

- [ ] **Step 5: Commit retrieval foundation**

```bash
git add services/agent-rag
git commit -m "feat: add AOI knowledge retrieval"
```

### Task 2: Data Quality Agent Workflow

**Files:**
- Create: `services/agent-rag/agent_rag/schemas.py`
- Create: `services/agent-rag/agent_rag/workflows/data_quality.py`
- Test: `services/agent-rag/tests/workflows/test_data_quality.py`

**Interfaces:**
- Consumes: event/attachment aggregates from `apps/api`
- Produces: `DataQualityAssessment(risk_level, findings, evidence_refs, recommended_actions)`

- [ ] **Step 1: Write the guardrail tests**

```python
def test_missing_identity_never_recommends_pass():
    result = run_data_quality_workflow({"identity_complete": False, "attachments": []})
    assert result.risk_level == "CRITICAL"
    assert "AUTO_PASS" not in result.recommended_actions
```

- [ ] **Step 2: Run the workflow test**

Run: `cd services/agent-rag && python -m pytest tests/workflows/test_data_quality.py -v`
Expected: FAIL before the graph exists.

- [ ] **Step 3: Implement the LangGraph states and deterministic rules**

Nodes: `validate_identity`, `check_completeness`, `check_time_delta`, `check_duplicates`, `check_drift`, `retrieve_sop`, `compose_assessment`. Route identity failure directly to critical assessment.

- [ ] **Step 4: Run tests and commit**

Run: `cd services/agent-rag && python -m pytest tests/workflows/test_data_quality.py -v`
Expected: PASS.

```bash
git add services/agent-rag/agent_rag/schemas.py services/agent-rag/agent_rag/workflows/data_quality.py services/agent-rag/tests/workflows/test_data_quality.py
git commit -m "feat: add data quality agent"
```

### Task 3: Review and Anomaly Report Agent

**Files:**
- Create: `services/agent-rag/agent_rag/llm.py`
- Create: `services/agent-rag/agent_rag/workflows/report.py`
- Test: `services/agent-rag/tests/workflows/test_report.py`

**Interfaces:**
- Produces: `ReportDraft(summary, observed_facts, similar_incidents, open_questions, evidence_refs)`
- Requires: human confirmation before `status=APPROVED`

- [ ] **Step 1: Write evidence and approval tests**

```python
def test_report_draft_separates_facts_from_hypotheses():
    draft = run_report_workflow(sample_incident())
    assert draft.observed_facts
    assert draft.open_questions
    assert all(ref.document_id for ref in draft.evidence_refs)
    assert draft.status == "DRAFT"
```

- [ ] **Step 2: Run the report test**

Run: `cd services/agent-rag && python -m pytest tests/workflows/test_report.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement provider abstraction and report graph**

`DemoLlmProvider` must create deterministic Chinese text from structured facts. `OpenAICompatibleProvider` must use configured base URL, model, timeout, and API key. Nodes: `collect_evidence`, `retrieve_similar`, `draft`, `validate_citations`, `require_human_confirmation`.

- [ ] **Step 4: Run tests and commit**

Run: `cd services/agent-rag && python -m pytest tests/workflows/test_report.py -v`
Expected: PASS.

```bash
git add services/agent-rag/agent_rag/llm.py services/agent-rag/agent_rag/workflows/report.py services/agent-rag/tests/workflows/test_report.py
git commit -m "feat: add anomaly report agent"
```

### Task 4: Model Governance Agent

**Files:**
- Create: `services/agent-rag/agent_rag/workflows/model_governance.py`
- Test: `services/agent-rag/tests/workflows/test_model_governance.py`

**Interfaces:**
- Produces: `ReleaseRecommendation(action, failed_gates, evidence_refs, approval_required=True)`
- Inputs: recall, auto-pass escape rate, false-positive rate, review ratio, P95, throughput, backlog, mismatch rate, shadow difference rate

- [ ] **Step 1: Write production-gate tests**

```python
def test_nonzero_silent_mismatch_blocks_release():
    result = evaluate_release(metrics(silent_mismatch_rate=0.0001))
    assert result.action == "BLOCK"
    assert "SILENT_MISMATCH" in result.failed_gates
    assert result.approval_required is True
```

- [ ] **Step 2: Run the test**

Run: `cd services/agent-rag && python -m pytest tests/workflows/test_model_governance.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement explicit release gates**

The workflow must block on nonzero verified silent mismatch, missing blind-test evidence, failed rollback drill, critical-defect escape, sustained backlog growth, or absent approval metadata. It may return `SHADOW_MORE`, `ROLLBACK`, or `READY_FOR_APPROVAL`; it cannot execute release.

- [ ] **Step 4: Run tests and commit**

Run: `cd services/agent-rag && python -m pytest tests/workflows/test_model_governance.py -v`
Expected: PASS.

```bash
git add services/agent-rag/agent_rag/workflows/model_governance.py services/agent-rag/tests/workflows/test_model_governance.py
git commit -m "feat: add model governance agent"
```

### Task 5: Agent API and Failure-Isolated Main API Client

**Files:**
- Create: `services/agent-rag/agent_rag/main.py`
- Create: `services/agent-rag/agent_rag/api/routes.py`
- Create: `apps/api/app/clients/agent_rag.py`
- Create: `apps/api/app/services/agent_jobs.py`
- Modify: `apps/api/app/models/governance.py`
- Create: `apps/api/migrations/versions/0002_agent_jobs.py`
- Modify: `apps/api/app/api/routes/alerts.py`
- Modify: `apps/api/app/api/routes/models.py`
- Test: `apps/api/tests/services/test_agent_failure_isolation.py`
- Test: `services/agent-rag/tests/test_api.py`

**Interfaces:**
- Agent endpoints: `/agent-api/v1/health`, `/agent-api/v1/assess-data-quality`, `/agent-api/v1/draft-report`, `/agent-api/v1/recommend-model-release`, `/agent-api/v1/knowledge/search`
- Main API client returns `AgentJobResult(status="UNAVAILABLE")` on timeout instead of raising into inspection routes
- Produces persisted `AgentJob(id, job_type, subject_id, request_fingerprint, status, result, error, created_at, updated_at)` with `QUEUED/RUNNING/SUCCEEDED/FAILED/UNAVAILABLE`

- [ ] **Step 1: Write failure-isolation tests**

```python
async def test_agent_timeout_does_not_break_inspection_dashboard(client, agent_stub):
    agent_stub.raise_timeout()
    response = client.get("/api/v1/dashboard/summary")
    assert response.status_code == 200
```

- [ ] **Step 2: Run API tests**

Run: `cd apps/api && python -m pytest tests/services/test_agent_failure_isolation.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement timeouts, circuit state, and persisted job status**

Use a 500 ms connect timeout and configurable 5 s total timeout, no automatic retry for non-idempotent report creation, and a persistent job row with `QUEUED/RUNNING/SUCCEEDED/FAILED/UNAVAILABLE` states. Migration `0002_agent_jobs.py` must add an index on `(job_type, subject_id, created_at)` and a unique request fingerprint for idempotent jobs.

- [ ] **Step 4: Run both service test suites**

Run:

```bash
cd services/agent-rag && python -m pytest -q
cd ../../apps/api && python -m pytest -q
```

Expected: PASS.

- [ ] **Step 5: Commit integration**

```bash
git add services/agent-rag apps/api/app/clients apps/api/app/services/agent_jobs.py apps/api/app/models/governance.py apps/api/migrations/versions/0002_agent_jobs.py apps/api/app/api/routes apps/api/tests/services
git commit -m "feat: integrate isolated Agent RAG service"
```

### Task 6: Production and GPU Deployment Profiles

**Files:**
- Create: `services/agent-rag/Dockerfile`
- Modify: `infra/docker-compose.yml`
- Create: `infra/docker-compose.gpu.yml`
- Modify: `infra/nginx.conf`
- Create: `docs/runbooks/agent-rag-failure.md`
- Create: `docs/runbooks/model-rollback.md`
- Test: `infra/tests/compose_contract.ps1`

**Interfaces:**
- Base: GPU-free demo including Agent/RAG with deterministic embeddings and `DemoLlmProvider`
- Profile `local-llm`: switches Agent/RAG to an OpenAI-compatible local model service with one L4-class GPU
- Profile `tensorrt`: TensorRT service with edge GPU reservation

- [ ] **Step 1: Write the deployment contract check**

```powershell
$base = docker compose -f infra/docker-compose.yml config | Out-String
if ($base -match 'capabilities: \[gpu\]') { throw 'Base demo must not require GPU' }
$gpu = docker compose -f infra/docker-compose.yml -f infra/docker-compose.gpu.yml --profile local-llm config | Out-String
if ($gpu -notmatch '(?s)capabilities:.*gpu') { throw 'Local LLM profile must request GPU' }
```

- [ ] **Step 2: Run the contract and observe failure**

Run: `powershell -File infra/tests/compose_contract.ps1`
Expected: FAIL before GPU overlay exists.

- [ ] **Step 3: Implement profiles, health checks, resource limits, and routing**

Keep the deterministic Agent/RAG service in the base compose without GPU reservations. Add CPU/memory limits, named volumes, read-only application filesystems where practical, secret/env injection, health checks, restart policies, and an internal-only Docker network for `/agent-api/`; Nginx must not publish Agent endpoints. The GPU overlay changes provider settings and adds GPU reservations only for `local-llm` and `tensorrt` profiles.

- [ ] **Step 4: Verify failure isolation and profiles**

Run:

```bash
docker compose -f infra/docker-compose.yml up -d --build
docker compose -f infra/docker-compose.yml stop agent-rag
curl http://localhost:8080/api/v1/health
powershell -File infra/tests/compose_contract.ps1
```

Expected: API health remains 200; compose contract PASS.

- [ ] **Step 5: Commit deployment profiles**

```bash
git add services/agent-rag/Dockerfile infra docs/runbooks
git commit -m "feat: add Agent RAG and GPU deployment profiles"
```
