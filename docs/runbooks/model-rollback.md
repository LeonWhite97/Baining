# 模型发布与回滚手册

## 发布门禁

模型治理 Agent 只输出 `BLOCK`、`SHADOW_MORE` 或 `READY_FOR_APPROVAL`。静默错配率必须为零，盲测、关键缺陷漏放、P95、持续吞吐和回滚演练均有证据，且存在审批元数据。

## 影子运行

新旧模型结果写入 `inference_results`，由 `event_uuid + model_version + policy_version + input_fingerprint` 区分。比较召回、漏放、误报、REVIEW 比例、P95 和差异事件，不能覆盖生产结果。

## 回滚步骤

1. 将 `APP_MODE` 置为 `shadow`，必要时清空 `AOI_INFERENCE_BACKEND`，使模型不可用时安全进入 `MODEL_UNAVAILABLE`/REVIEW。
2. 核对最近一个已批准版本的 `best.pt`、`model_metadata.json`、模型 SHA-256、7 类顺序、`rgb_grayscale_stack_v1` 和策略版本。
3. 原子切换 `AOI_MODEL_PATH` 与 `AOI_MODEL_METADATA_PATH` 到该批准模型包，保持 `AOI_MODEL_IMGSZ` 与元数据一致，然后重启 API。
4. 用已批准的四光源黄金样本验证健康检查、模型版本、哈希门禁、结构化框、决策三态、告警和复核队列；未完成前不得恢复受控自动决策。
5. 保留新版本结果、差异报告、失败模型包哈希和审批记录，不删除图片或推理证据。
6. 由质量/工艺负责人确认后关闭事件，并创建复盘报告。
