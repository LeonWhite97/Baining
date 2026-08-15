# Agent/RAG 故障处置手册

## 触发条件

Agent 健康检查失败、检索超时、引用校验失败、LLM endpoint 不可用或报告生成超过总超时。

## 处置步骤

1. 确认 `/api/v1/health` 和 `/agent-api/v1/health`；先保证实时 API 正常。
2. 查看 `agent_status`。`UNAVAILABLE` 代表报告降级为确定性 DRAFT，不代表检测失败。
3. 检查 PostgreSQL/pgvector 连接、知识库迁移和容器日志；禁止为了恢复服务直接重放非幂等报告请求。
4. 需要临时停用时：`docker compose -f infra/docker-compose.yml stop agent-rag`，验证检测、看板、复核 API 仍返回 200。
5. 恢复后先执行知识检索、数据质量和报告草稿接口测试，再开放给质量人员。

## 安全边界

Agent/RAG 不得创建 PASS、修改 `inspection_events` 身份、执行模型发布或替代人工批准。未知、缺失、错配和低置信度仍进入 REVIEW/QUARANTINED。
