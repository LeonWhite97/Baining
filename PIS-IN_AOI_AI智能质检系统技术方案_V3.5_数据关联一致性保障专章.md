# PIS-IN AOI 智能质检系统技术方案 V3.5

## 数据关联一致性保障专章（PoC 实施与生产上线准备版）

**文档版本**：V3.5
**编制日期**：2026 年 8 月
**适用范围**：PIS-IN AOI 旁路 AI 复判系统的数据适配层、YOLOv8/TensorRT 推理服务、FastAPI 接口层、离线 LangGraph 数据治理流程。
**架构边界**：本专章不改变既定的 PIS-IN、FastAPI、YOLOv8、TensorRT 和独立 Agent/RAG 技术栈。当前采用 AOI 工控机单机独立运行，数据关联、审计和上线治理围绕本机采集、PostgreSQL 与可选 MES 异步同步实施；Handler 自动握手与实时 BIN 回传不属于当前主链路。

---

## 1. 目标与安全原则

系统需要将同一组件的 2D 多光源图像、3D 量测数据、原 AOI 判定、Tray Map 和设备元数据准确关联。任何无法证明属于同一物理检测事件的数据都不得融合，更不得触发 AI 自动 PASS。

核心原则：

1. **确定性关联优先**：使用设备原生 Inspection ID；设备未提供时，使用包含设备会话标识的确定性组合键。
2. **Light ID 不是事件主键**：R/G/B/RING 等光源是同一事件的附件属性，不能将同一组件拆分为多个事件。
3. **时间戳用于验证，不用于猜测归属**：等待文件到达的超时窗口与同事件采集时间偏差必须分开处理。
4. **数据异常默认保守**：身份缺失、关联失败、文件不完整、时间源不可信时进入 REVIEW 或 QUARANTINED，禁止自动 PASS。
5. **结果可复现**：每个推理结果必须关联输入指纹、模型版本、决策策略版本、3D 规则版本和理由码。
6. **AI 不阻塞设备**：任何数据适配、推理、数据库或报告服务故障均不得阻塞 PIS-IN 原 AOI 判定与机械流程。

---

## 2. 确定性事件标识

### 2.1 事件 Source Key

一个物理检测事件的关联键为：

```text
Source Key = DeviceID
           + DeviceSessionID
           + InspectionSequence
           + TrayID
           + SlotIndex
           + Surface
```

字段说明：

| 字段 | 作用 | 来源 |
|---|---|---|
| `device_id` | 区分设备 | 设备配置或设备输出 |
| `device_session_id` | 防止设备重启后计数器归零碰撞 | 原生会话 ID、启动日志或受控会话记录 |
| `inspection_sequence` | 区分同设备会话中的检测事件 | 原生 Inspection ID 优先，内部计数器次之 |
| `tray_id` | 区分 Tray | 文件名、日志或 Manifest |
| `slot_index` | 区分 Tray 内组件 | 文件名、Tray Map 或 Manifest |
| `surface` | 区分正背面 | 设备字段或受控检测序列 |

`light_id` 仅存在于附件记录中：

```text
R / G / B / RING / IR / UV / NONE
```

若不同曝光确实代表不同物理检测事件，必须增加 `exposure_sequence` 并将其纳入 Source Key；不能复用 `light_id`。

### 2.2 Source Key 生成

使用规范化 JSON 再计算 SHA256，避免字符串拼接歧义：

```python
import hashlib
import json


def generate_source_key_hash(
    device_id: str,
    device_session_id: str,
    inspection_sequence: str,
    tray_id: str,
    slot_index: str,
    surface: str,
) -> tuple[str, dict]:
    payload = {
        "device_id": device_id,
        "device_session_id": device_session_id,
        "inspection_sequence": inspection_sequence,
        "tray_id": tray_id,
        "slot_index": slot_index,
        "surface": surface,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), payload
```

`event_uuid` 由 FastAPI 适配层在创建事件时生成，仅用于内部追踪，不作为跨数据源关联依据。

---

## 3. 数据汇聚、校验与状态机

### 3.1 并发安全的汇聚流程

文件监听服务不直接以“先查询再创建”的方式处理事件，必须在数据库事务中进行原子 Upsert：

```text
文件到达
  -> 文件完整性检查
  -> 解析确定性 Source Key
  -> 原子创建或取得 inspection_events 行
  -> 写入附件记录（重复文件幂等忽略）
  -> 锁定事件行并重算附件集合、时间偏差和状态
  -> 提交事务
  -> 仅对 VALIDATED 事件发送推理队列
```

关键数据库操作：

```sql
INSERT INTO inspection_events (
    event_uuid,
    source_key_hash,
    source_key_payload,
    device_id,
    device_session_id,
    inspection_id_source,
    tray_id,
    slot_index,
    surface,
    inspection_sequence,
    association_status
)
VALUES (
    :event_uuid,
    :source_key_hash,
    CAST(:source_key_payload AS JSONB),
    :device_id,
    :device_session_id,
    :inspection_id_source,
    :tray_id,
    :slot_index,
    :surface,
    :inspection_sequence,
    'RECEIVED'
)
ON CONFLICT (source_key_hash)
DO UPDATE SET updated_at = NOW()
RETURNING event_uuid;
```

随后使用 `SELECT ... FOR UPDATE` 锁定该事件，再追加附件、计算完整性和更新状态。唯一约束负责最终幂等，缓存仅可作为性能优化，不得作为事实来源。

### 3.2 状态机

| 状态 | 含义 | 可转换状态 |
|---|---|---|
| `RECEIVED` | 收到第一份身份可解析的文件 | `COLLECTING`、`QUARANTINED` |
| `COLLECTING` | 等待预期附件集合 | `READY`、`EXPIRED`、`QUARANTINED` |
| `READY` | 附件已到齐且文件级校验通过 | `VALIDATED`、`INVALID` |
| `VALIDATED` | L1-L5 校验通过 | `INFERRED`、`REVIEW_REQUIRED` |
| `INFERRED` | 推理结果已写入 | `ARCHIVED`、`REVIEW_REQUIRED` |
| `ARCHIVED` | 已归档，最终态 | 无 |
| `EXPIRED` | 等待超时，附件未到齐，最终态 | 无 |
| `QUARANTINED` | 身份依据缺失，最终态 | 无 |
| `REVIEW_REQUIRED` | 数据或判决需要人工复核 | `ARCHIVED` |
| `INVALID` | 数据错位、格式错误或校验失败，最终态 | 无 |

应用层必须校验合法状态迁移；数据库状态枚举仅用于阻止非法状态值。

### 3.3 五级校验

| 级别 | 校验内容 | 失败处理 |
|---|---|---|
| L1 文件完整性 | 文件已完成写入、大小稳定、Hash 可计算、非损坏 | 重试，超时后 `REVIEW_REQUIRED` |
| L2 结构完整性 | 图像尺寸、位深、光源编号；3D JSON Schema、字段单位和范围 | `INVALID` 或 `REVIEW_REQUIRED` |
| L3 关联一致性 | Source Key 的设备、会话、序号、Tray、Slot、正背面完全一致 | 禁止融合，`REVIEW_REQUIRED` |
| L4 采集时间一致性 | 每个附件的 `captured_at` 与同事件 2D/3D 时间窗口相符 | 禁止融合，`REVIEW_REQUIRED` |
| L5 幂等性 | 相同事件、输入指纹、模型和策略版本不重复执行 | 返回已有结果或安全重试 |

参数说明：

```text
assembly_timeout：允许等待其他附件到达的时间，PoC 起始值 2-5 秒。
capture_delta_threshold：同一事件的采集时间偏差阈值，按设备、产品、正背面和量测模式实测后配置。
```

`assembly_timeout` 不是数据匹配阈值。禁止在超时后使用“较新文件”或邻近时间戳文件补齐事件。

### 3.4 文件写入完整性

优先级如下：

1. 设备端临时文件写完后原子重命名；
2. 设备端完成标记文件；
3. 文件大小和修改时间连续两次稳定后读取；
4. 解析失败重试，最多 3 次；
5. 超过 `assembly_timeout` 后隔离，绝不错误拼接。

---

## 4. 降级与判决边界

| 数据状态 | 推理模式 | 自动 PASS | 处置 |
|---|---|---|---|
| 2D、3D、AOI 完整且关联验证通过 | `FULL` | 仅按全局决策策略允许 | 完整融合推理 |
| 2D 身份确认，3D 缺失 | `2D_ONLY` | 默认禁止 | 使用独立校准模型，进入 REVIEW 或受控策略 |
| 身份确认但部分附件缺失 | `PARTIAL` | 禁止 | 结果标记降级并进入 REVIEW |
| 身份无法确认 | `QUARANTINED` | 禁止 | 不执行正常推理，人工处置 |
| 文件错位、时间超差或结构错误 | `INVALID` | 禁止 | REVIEW 并告警 |

`FULL` 仅表示可进入完整三态决策，并不是自动 PASS 结论。自动 PASS 仍需同时满足：

```text
原 AOI 为 NG 或低置信度
AND 输入质量正常
AND 模型概率经盲测校准后达到 PASS 阈值
AND 3D 硬规则未触发
AND 不属于 UNKNOWN 或高风险缺陷
```

---

## 5. PostgreSQL 14+ 数据模型与约束

以下 DDL 为单份迁移脚本。正式部署应先在测试库执行 `BEGIN` / `ROLLBACK` 冒烟验证，再通过受控迁移工具提交。

```sql
BEGIN;

CREATE TABLE inspection_events (
    id                      BIGSERIAL PRIMARY KEY,
    event_uuid              VARCHAR(64) UNIQUE NOT NULL,
    source_key_hash         CHAR(64) UNIQUE NOT NULL,
    source_key_payload      JSONB NOT NULL,

    device_id               VARCHAR(32) NOT NULL,
    device_session_id       VARCHAR(64) NOT NULL,
    inspection_id_source    VARCHAR(16) NOT NULL,
    product_id              VARCHAR(32),
    batch_id                VARCHAR(32),
    tray_id                 VARCHAR(32) NOT NULL,
    slot_index              VARCHAR(8) NOT NULL,
    surface                 VARCHAR(8) NOT NULL DEFAULT 'TOP',
    inspection_sequence     VARCHAR(64) NOT NULL,

    t_2d_raw                TIMESTAMP WITH TIME ZONE,
    t_3d_raw                TIMESTAMP WITH TIME ZONE,
    t_aoi_raw               TIMESTAMP WITH TIME ZONE,
    t_ingest                TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    capture_delta_ms        INTEGER,
    delta_check_passed      BOOLEAN,

    has_2d                  BOOLEAN NOT NULL DEFAULT FALSE,
    has_3d                  BOOLEAN NOT NULL DEFAULT FALSE,
    has_aoi_result          BOOLEAN NOT NULL DEFAULT FALSE,
    data_completeness       VARCHAR(16) NOT NULL DEFAULT 'PENDING',
    required_light_set      JSONB NOT NULL DEFAULT '[]'::jsonb,
    received_light_set      JSONB NOT NULL DEFAULT '[]'::jsonb,

    association_status      VARCHAR(16) NOT NULL DEFAULT 'RECEIVED',
    association_reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
    validation_version      VARCHAR(32) NOT NULL,

    aoi_decision            VARCHAR(16),
    aoi_defect_code         VARCHAR(32),
    ball_height_max         DOUBLE PRECISION,
    coplanarity_max         DOUBLE PRECISION,
    warp_max                DOUBLE PRECISION,

    schema_version          VARCHAR(16) NOT NULL DEFAULT 'v3.5',
    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_ie_id_source CHECK (
        inspection_id_source IN ('NATIVE', 'SESSION_COUNTER', 'MANIFEST')
    ),
    CONSTRAINT chk_ie_surface CHECK (surface IN ('TOP', 'BOTTOM')),
    CONSTRAINT chk_ie_status CHECK (
        association_status IN (
            'RECEIVED', 'COLLECTING', 'READY', 'VALIDATED', 'INFERRED',
            'ARCHIVED', 'EXPIRED', 'QUARANTINED', 'REVIEW_REQUIRED', 'INVALID'
        )
    ),
    CONSTRAINT chk_ie_completeness CHECK (
        data_completeness IN ('PENDING', 'FULL', '2D_ONLY', 'PARTIAL', 'INVALID')
    ),
    CONSTRAINT chk_ie_light_sets CHECK (
        jsonb_typeof(required_light_set) = 'array'
        AND jsonb_typeof(received_light_set) = 'array'
    )
);

CREATE TABLE quarantine_events (
    id                      BIGSERIAL PRIMARY KEY,
    quarantine_id           VARCHAR(64) UNIQUE NOT NULL,
    source_file_path        VARCHAR(512) NOT NULL,
    file_hash               CHAR(64) NOT NULL,
    file_size_bytes         BIGINT,
    parse_error             VARCHAR(256) NOT NULL,
    extracted_fields        JSONB NOT NULL DEFAULT '{}'::jsonb,
    device_id               VARCHAR(32),
    resolution              VARCHAR(64),
    resolved_at             TIMESTAMP WITH TIME ZONE,
    resolved_by             VARCHAR(64),
    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_qe_resolution CHECK (
        resolution IS NULL OR resolution IN ('MANUAL_REVIEW', 'REPROCESSED', 'DISCARDED')
    )
);

CREATE TABLE data_attachment_links (
    id                      BIGSERIAL PRIMARY KEY,
    event_uuid              VARCHAR(64) REFERENCES inspection_events(event_uuid) ON DELETE RESTRICT,
    quarantine_id           VARCHAR(64) REFERENCES quarantine_events(quarantine_id) ON DELETE RESTRICT,
    captured_at             TIMESTAMP WITH TIME ZONE,
    source_timestamp_raw    VARCHAR(64),

    data_type               VARCHAR(16) NOT NULL,
    light_id                VARCHAR(8) NOT NULL,
    file_path               VARCHAR(512) NOT NULL,
    file_hash               CHAR(64) NOT NULL,
    file_size_bytes         BIGINT,
    file_stable_at          TIMESTAMP WITH TIME ZONE NOT NULL,
    checksum_passed         BOOLEAN NOT NULL DEFAULT TRUE,
    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_dal_owner CHECK (
        (event_uuid IS NOT NULL AND quarantine_id IS NULL)
        OR (event_uuid IS NULL AND quarantine_id IS NOT NULL)
    ),
    CONSTRAINT chk_dal_type_light CHECK (
        (data_type = '2D_IMAGE' AND light_id IN ('R', 'G', 'B', 'RING', 'IR', 'UV'))
        OR (data_type IN ('3D_JSON', 'AOI_LOG') AND light_id = 'NONE')
    )
);

CREATE TABLE inference_results (
    id                      BIGSERIAL PRIMARY KEY,
    event_uuid              VARCHAR(64) NOT NULL REFERENCES inspection_events(event_uuid) ON DELETE RESTRICT,
    model_version           VARCHAR(32) NOT NULL,
    policy_version          VARCHAR(32) NOT NULL,
    three_d_rule_version    VARCHAR(32),
    input_fingerprint       CHAR(64) NOT NULL,
    inference_mode          VARCHAR(16) NOT NULL,
    ai_decision             VARCHAR(16) NOT NULL,
    ai_confidence           DOUBLE PRECISION,
    defect_code             VARCHAR(32),
    defect_bbox             JSONB NOT NULL DEFAULT '[]'::jsonb,
    input_quality_status    VARCHAR(16) NOT NULL,
    decision_reason_codes   JSONB NOT NULL DEFAULT '[]'::jsonb,
    inference_latency_ms    INTEGER,
    run_scope               VARCHAR(16) NOT NULL DEFAULT 'SHADOW',
    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_ir_event_model_policy_input UNIQUE (
        event_uuid, model_version, policy_version, input_fingerprint
    ),
    CONSTRAINT chk_ir_mode CHECK (inference_mode IN ('FULL', '2D_ONLY', 'PARTIAL')),
    CONSTRAINT chk_ir_decision CHECK (ai_decision IN ('PASS', 'FAIL', 'REVIEW')),
    CONSTRAINT chk_ir_quality CHECK (input_quality_status IN ('OK', 'DEGRADED', 'INVALID')),
    CONSTRAINT chk_ir_scope CHECK (run_scope IN ('SHADOW', 'CONTROLLED', 'PRODUCTION'))
);

CREATE UNIQUE INDEX uq_dal_event_file
    ON data_attachment_links (event_uuid, data_type, light_id, file_hash)
    WHERE event_uuid IS NOT NULL;

CREATE UNIQUE INDEX uq_dal_quarantine_file
    ON data_attachment_links (quarantine_id, data_type, light_id, file_hash)
    WHERE quarantine_id IS NOT NULL;

CREATE INDEX idx_ie_device_tray ON inspection_events(device_id, tray_id, slot_index);
CREATE INDEX idx_ie_device_time ON inspection_events(device_id, created_at DESC);
CREATE INDEX idx_ie_batch_product ON inspection_events(batch_id, product_id);
CREATE INDEX idx_ie_status ON inspection_events(association_status);
CREATE INDEX idx_dal_captured_at ON data_attachment_links(captured_at);
CREATE INDEX idx_ir_model ON inference_results(model_version);
CREATE INDEX idx_qe_resolved ON quarantine_events(resolved_at);

COMMIT;
```

说明：

- `source_key_hash` 与 `event_uuid` 的唯一约束已经自带索引，不再创建重复索引。
- `UNIQUE(event_uuid, model_version, policy_version, input_fingerprint)` 允许同一事件在策略版本或已验证输入集合变化后保留结果，同时阻止相同输入、模型和策略的重复推理。
- 质量审计数据不使用级联删除。生产环境禁止物理删除检测事件，采用逻辑归档和保留策略。

---

## 6. 推理审计与回溯

每条 `inference_results` 至少保存：

```text
event_uuid
input_fingerprint
model_version
policy_version
three_d_rule_version
inference_mode
input_quality_status
ai_decision
decision_reason_codes
```

示例理由码：

```json
[
  "AOI_NG_TRIGGERED",
  "FULL_INPUT_VERIFIED",
  "YOLO_NO_DEFECT_HIGH_CONFIDENCE",
  "THREE_D_WITHIN_LIMIT",
  "PASS_POLICY_V1"
]
```

这样可以准确回答：某个组件为何被放行、使用了哪些输入、由哪个模型和规则版本产生、后续模型更新是否改变了结论。

---

## 7. 设备接口与 PoC 门禁

进入数据适配层开发前，必须确认 2D、3D 和 AOI 三类输出都能获取一致的：

```text
DeviceID + DeviceSessionID + InspectionSequence + TrayID + SlotIndex + Surface
```

PoC 验收样本不少于 1000 个事件，并覆盖：

- 设备重启前后；
- Tray 切换；
- 正面与背面；
- 多槽位同时成像；
- 实际使用的全部光源；
- 产品和批次切换；
- 文件乱序、重复、延迟、断网和部分写入。

验收条件：

```text
可验证事件的 Source Key 一致率 = 100%
故障注入场景的静默错配率 = 0
同一事件的多光源文件全部汇聚到同一 event_uuid
模型版本和策略版本可并存且不覆盖
异常数据均进入 REVIEW 或 QUARANTINED，不产生自动 PASS
```

若任意来源缺少必要身份字段，必须由设备厂商提供 Manifest 文件或修正导出格式；禁止以时间戳猜测归属。

---

## 8. 上线准入与回退

| 阶段 | 自动 PASS | 准入条件 |
|---|---|---|
| 影子运行 | 禁止 | 数据关联、推理、回写均记录，但不控制产线 |
| 受控上线 | 仅 `FULL` 的指定产品和缺陷类别 | 独立盲测、人工复核、P95、积压与回退演练通过 |
| 扩大范围 | 分设备、产品、类别逐步开放 | 持续运行稳定、差异样本闭环、质量共同批准 |

生产自动 PASS 的硬条件：

1. 静默错配率为 0，且有故障注入和人工抽检证据；
2. 自动 PASS 漏放率满足质量部门批准阈值，并报告 95% 置信区间；
3. 缺陷召回率、误报过滤率、REVIEW 比例按产品、批次、设备和班次分层达标；
4. 峰值负载下持续吞吐与端到端 P95 不造成队列持续增长；
5. 断网、服务退出、GPU 故障、数据库不可用和错误模型发布时，原 AOI 均可继续运行；
6. 模型、策略、阈值和本机自动 PASS 范围均经质量、设备与 AI 团队审批；如启用 MES 异步同步，再由 IT/MES 团队确认接口与权限。

默认安全策略：

```text
2D_ONLY / PARTIAL / QUARANTINED / INVALID / UNKNOWN
=> REVIEW

FastAPI、TensorRT、数据库或网络异常
=> 原 AOI 继续运行，AI 不阻塞设备
```

---

## 9. 实施清单

| 项目 | 验收标准 | 责任角色 |
|---|---|---|
| 设备身份字段确认 | 三类数据可解析一致 Source Key | 设备工程师 |
| 数据库冒烟 | PostgreSQL 14+ 测试库执行迁移并回滚成功 | DBA |
| 原子汇聚 | 并发到达时同一 Source Key 仅产生一个事件 | AI/后端工程师 |
| 附件约束 | 错误的 `data_type/light_id` 组合被数据库拒绝 | AI/后端工程师 |
| 多光源聚合 | `required_light_set` 与附件集合一致 | AI 工程师 |
| 故障注入 | 乱序、重复、延迟、断网、重启和部分写入无静默错配 | AI/设备工程师 |
| 审计回溯 | 可由结果追溯附件、模型、策略、规则和理由码 | 质量/AI 团队 |
| 影子运行 | 按设备、产品、班次统计差异样本并形成闭环 | 质量/设备团队 |

---

## 10. 结论

V3.5 将数据关联从“按时间猜测”改为“按确定性事件键汇聚”，并补齐了并发安全、附件约束、版本审计、设备重启防碰撞和生产上线门禁。该版本可作为 PoC 数据适配层、YOLO/TensorRT 联调、影子运行和后续缺陷报表/工站异常预警模块的正式设计基线。

生产自动 PASS 仍必须遵循受控上线原则：先影子、再分产品和缺陷类别开放、最后逐步扩大范围。
