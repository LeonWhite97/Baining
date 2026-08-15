# PIS-IN AOI AI V3.5 Delivery Enhancement Program

> **For agentic workers:** Execute the linked plans in order. Each plan produces an independently testable deliverable and ends with its own verification gate.

**Goal:** Deliver a runnable AOI quality-inspection demo, offline Agent/RAG governance, Docker deployment, and a consistent V3.5 documentation/PDF package.

**Architecture:** The real-time path remains deterministic: adapter -> FastAPI -> YOLOv8/TensorRT adapter -> 3D/rule fusion -> PASS/FAIL/REVIEW. The demo uses generated AOI events when real images and model weights are unavailable. LangGraph and RAG remain in a separate offline service and cannot control real-time automatic PASS.

**Tech Stack:** React, TypeScript, Vite, ECharts, FastAPI, Pydantic, SQLAlchemy, Alembic, PostgreSQL/pgvector, LangGraph, Docker Compose, Nginx, Pytest, Vitest, Playwright, ReportLab.

## Global Constraints

- Preserve the PIS-IN AOI sidecar architecture and V3.5 Source Key/data-association rules.
- The frontend must not display the RMB 5 million project budget or budget breakdown.
- Real project facts: 2024-09 to 2025-01, eight staged participants, RMB 5 million total budget.
- False-positive targets use the original AOI NG candidate pool: baseline 12%, PoC <=6%, controlled rollout <=3%, mature target <=1.5%.
- If all inspected products are the denominator, report a separate full-inspection false-positive target <=0.5%.
- The demo must run without an NVIDIA GPU and without real AOI images or YOLO weights.
- Unknown, incomplete, mismatched, and low-confidence inputs cannot automatically PASS.
- Agent/RAG failures must not affect the real-time inspection API.
- All numerical claims must be labelled as real fact, historical result, standard estimate, or field-validation target.

## Execution Order

1. [Runnable Demo Foundation](2026-08-07-pis-in-aoi-demo-foundation.md)
   - Produces the database, FastAPI API, AOI simulator, React frontend, business interactions, and GPU-free Docker demo.
2. [Agent/RAG and Production Deployment](2026-08-07-pis-in-aoi-agent-rag-deployment.md)
   - Adds pgvector knowledge retrieval, three LangGraph workflows, failure isolation, Nginx, GPU profiles, and operations checks.
3. [Documentation and PDF Refresh](2026-08-07-pis-in-aoi-docs-and-pdf-refresh.md)
   - Updates all metric wording, writes the project/deployment documents, captures the runnable UI, and regenerates verified V3.5 PDFs.

## Program Completion Gate

- `docker compose up --build` starts the GPU-free demo.
- The simulator creates events that appear on the dashboard and Tray Map, and the same normalized contract is covered by a real PIS-IN adapter test.
- Review actions persist after service restart.
- Alerts can be acknowledged, closed, and used to produce a persisted anomaly-report draft.
- Agent/RAG can be disabled without breaking inspection pages.
- Playwright passes the dashboard, event-detail, review, Tray Map, alert, and report journeys.
- Final PDFs render without clipped text, overlap, broken glyphs, or stale V3.0/old metric wording.
