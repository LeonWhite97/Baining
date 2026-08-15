# 模型发布与回滚手册

## 发布门禁

模型治理 Agent 只输出 `BLOCK`、`SHADOW_MORE` 或 `READY_FOR_APPROVAL`。静默错配率必须为零，盲测、关键缺陷漏放、P95、持续吞吐和回滚演练均有证据，且存在审批元数据。

## 影子运行

新旧模型结果写入 `inference_results`，由 `event_uuid + model_version + policy_version + input_fingerprint` 区分。比较召回、漏放、误报、REVIEW 比例、P95 和差异事件，不能覆盖生产结果。

## 回滚步骤

1. 停止新版本流量或将其标记为 `SHADOW`。
2. 恢复最近一个已批准的 `PRODUCTION` 版本和策略版本。
3. 保留新版本结果、差异报告和审批记录，不删除证据。
4. 验证健康检查、决策三态、告警和复核队列。
5. 由质量/工艺负责人确认后关闭事件，并创建复盘报告。

