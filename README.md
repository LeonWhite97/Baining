# PIS-IN AOI AI Intelligent Inspection

An internal proof of concept for traceable AOI inspection workflows. The repository connects PIS-IN image evidence, deterministic validation, three-state decisions (`PASS`, `FAIL`, and `REVIEW`), model governance, operator review, alerts, reports, and a web console.

## Current Boundary

The checked-in project proves the software workflow, deployment contracts, and fail-closed safety controls. It does not include production FC-BGA images, formal annotations, trained YOLO weights, TensorRT engines, blind-test evidence, production accuracy, false-positive or escape rates, throughput, or site P95 latency. Public PCB samples are stability fixtures only and are not FC-BGA accuracy evidence.

The FC-BGA YOLOv8 toolkit is designed for a portfolio/internal PoC. Ultralytics licensing must be reviewed before any closed-source or commercial distribution.

## Repository Map

- `apps/api`: FastAPI inspection API and inference adapters.
- `apps/web`: React/Vite operations console.
- `services/agent-rag`: failure-isolated analysis service.
- `services/simulator`: repeatable end-to-end workflow simulator.
- `tools/vision/fc_bga_yolo`: FC-BGA data, training, prediction, validation, deduplication, and export tools.
- `docs/PIS-IN_AOI_AI智能质检_V3.5_项目总说明书.md`: factual system boundary and delivery index.
- `docs/superpowers/specs/2026-08-15-fc-bga-yolo-training-design.md`: approved FC-BGA design.

## Local Verification

```powershell
python -m pytest apps/api/tests -q
python -m pytest services/agent-rag/tests -q
python -m pytest services/simulator/tests -q
npm.cmd --prefix apps/web test -- --run
npm.cmd --prefix apps/web run build
```

Run Python tests from each package directory if another installed package shadows the local `app`, `agent_rag`, or `simulator` module. Deployment examples and environment contracts are in `infra/` and the deployment guide under `docs/`.
