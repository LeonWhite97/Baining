# PIS-IN AOI AI 智能质检 V3.5 部署运维手册

## 1. 服务清单

当前交付采用 AOI 工控机单机独立运行。操作员或本机 PIS-IN AOI 软件触发采集，Handler 自动握手与实时 BIN 回传默认关闭，仅保留未来扩展代码。

| 服务 | 端口 | 说明 |
|---|---:|---|
| `frontend` | 8080 | Nginx 托管 React，代理 `/api/` |
| `api` | 8000 (内网) | FastAPI 业务 API |
| `postgres` | 5432 (内网) | PostgreSQL 16 + pgvector |
| `agent-rag` | 8013 (内网) | 3 Agent、RAG、确定性 Demo Provider |
| `simulator` | 无 | 生成确定性 AOI 事件 |

## 2. GPU-free 演示

```powershell
Copy-Item .env.example .env
docker compose -f infra/docker-compose.yml config
docker compose -f infra/docker-compose.yml build api
docker compose -f infra/docker-compose.yml build agent-rag
docker compose -f infra/docker-compose.yml build simulator
docker compose -f infra/docker-compose.yml build frontend
docker compose -f infra/docker-compose.yml up -d --no-build
Invoke-WebRequest http://localhost:8080/api/v1/health
```

`.env` 必须保持 `HANDLER_INTEGRATION_ENABLED=false`。基础模式不请求 GPU，且 `APP_MODE=demo`、`AOI_INFERENCE_BACKEND=demo` 时由 `DemoInferenceAdapter` 提供确定性结果，Agent/RAG 使用确定性 embedding 和 Provider。Demo 后端不能用于 shadow/controlled；非 Demo 模式未配置真实模型时必须记录 `MODEL_UNAVAILABLE` 并进入 REVIEW。

当前验证环境为 Docker Desktop 4.84 / Compose 5.3.1。该版本在中文工作区同时构建多个 Compose 服务时可能触发多目标 Bake gRPC 会话异常，因此手册采用逐服务构建；镜像构建完成后再使用 `up -d --no-build` 启动。若现场版本已验证不存在该问题，可简化为 `docker compose -f infra/docker-compose.yml up -d --build`。

单机正式联调建议使用 `APP_MODE=shadow`，此时 AI 只记录建议，最终结果由人工复核确认。只有真实数据盲测、回滚演练和质量审批全部通过后，才可使用 `APP_MODE=controlled` 与 `AUTO_PASS_ENABLED=true`。Handler 路由即使处于 `shadow/controlled`，也只有显式设置 `HANDLER_INTEGRATION_ENABLED=true` 才会注册。

Ultralytics shadow 联调需在 API 环境安装 `apps/api[vision]`，并设置 `AOI_INFERENCE_BACKEND=ultralytics`、`AOI_MODEL_PATH`、`AOI_MODEL_METADATA_PATH`、`AOI_MODEL_DEVICE`、`AOI_MODEL_IMGSZ` 和 `AOI_MODEL_CONF`。模型元数据、固定 7 类顺序、`rgb_grayscale_stack_v1` 和权重 SHA-256 任一不匹配均 fail-closed。详细训练与导出命令见 `tools/vision/fc_bga_yolo/README.md`。

首版真实适配器把 `normal_confidence` 固定为 0；无框或低分输出只能 REVIEW，不能自动 PASS。R/G/B 灰度三通道堆叠是待现场对照实验验证的输入方案，当前没有同机位现场样本证明其优于单光源或其他融合方式。

## 3. GPU overlay

```powershell
docker compose -f infra/docker-compose.yml -f infra/docker-compose.gpu.yml --profile local-llm up -d --build
docker compose -f infra/docker-compose.yml -f infra/docker-compose.gpu.yml --profile tensorrt up -d
```

`local-llm` 只为 Agent/RAG 请求 L4 类 GPU；`tensorrt` 为边缘推理服务请求 GPU。模型 engine 缺失或 runtime 不可用时必须 fail-closed，结果进入 REVIEW。

## 4. 数据迁移与备份

API 容器启动前执行 `alembic upgrade head`。生产环境使用 PostgreSQL，不使用 SQLite。备份至少包含 PostgreSQL 数据、原始文件索引、模型版本/策略版本和 Agent 知识库版本。

## 5. 常见检查

- API 健康：`/api/v1/health`
- Agent 健康：容器内访问 `/agent-api/v1/health`
- 数据库：检查 `inspection_events.source_key_hash` 唯一约束和 `quarantine_events` 增长。
- 告警：窗口样本数达到最小值后才计算工站缺陷率；告警需 ACK 后生成报告。
- 前端：Nginx 必须保留 SPA fallback；直接刷新 `/alerts` 等深链接不能返回 404。
- Handler 开关：默认访问 Handler START 路由应返回 404；只有未来独立联调时才临时开启。
