# Handler上位机-AOI系统接口规范

> **状态变更（2026-08-10）：暂停启用 / 仅作未来扩展储备。** 当前项目采用 AOI 单机独立运行，不接入 Handler 自动握手与实时 BIN 回传。本文档保留接口预研和安全约束，但不属于当前交付主链路；代码侧默认 `HANDLER_INTEGRATION_ENABLED=false`。

**文档版本：** V1.4（联调基线版）
**线协议版本：** 1.4
**适用对象：** Handler上位机供应商、AOI系统开发团队、设备工程师、测试团队
**最后更新：** 2026-08-10
**状态：** 暂停启用；未来如恢复 Handler 自动化集成，需重新评审并完成硬件在环测试

## 1. 文档概述

### 1.1 目的

本规范定义半导体封测流程中 Handler 上位机与 PIS-IN AOI 系统之间的实时通信接口，包括职责边界、TCP/JSON 报文、检测周期状态机、幂等规则、AOI BIN、异常处理、结果可靠投递、MES 异步旁路及联调验收标准。

### 1.2 适用场景

生产流程为：

```text
上料 -> 烘烤 -> 测试1 -> 降温 -> 测试2 -> AOI -> 下料
```

Handler 采用单 Site 顺序检测。每个工站均可产生 NG 下料结果；AOI 作为独立工站完成外观检测，将 AOI BIN 直接发送给 Handler 上位机。MES 并行接收检测数据，用于良率统计、批次追溯和报表，不参与实时分选闭环。

### 1.3 核心原则

1. Handler 只负责单 Site 顺序调度、追溯上下文和最终物理分选，不传输图像。
2. PIS-IN AOI 负责 R/G/B/RING 采图、文件校验、YOLOv8/TensorRT 推理和 AOI BIN 输出。
3. START 不携带图像路径、图像 Hash 或附件数组。
4. Handler 实时主链路和 MES 异步旁路独立管理，MES 故障不得阻塞 Handler。
5. 缺图、错配、输入无效、模型不可用、超时或未知状态不得产生 PASS。
6. AOI BIN 是工站结果，不是整条 Handler 流程的最终下料 BIN。
7. Agent/RAG 仅用于异常分析和报告，不参与实时 PASS/FAIL/REVIEW 判定。

### 1.4 术语

| 术语 | 含义 |
|---|---|
| TraceID | Handler 为芯片生成的全局业务追溯标识 |
| CycleID | Handler 会话内一次物理 AOI 检测周期标识 |
| HandlerSessionID | Handler 每次启动生成的新会话标识 |
| EventUUID | AOI 数据库内部检测事件主键 |
| SourceKeyHash | AOI 根据规范化业务字段生成的幂等关联 Hash |
| AOI BIN | AOI 工站结果编码 |
| Final BIN | Handler 汇总测试1、测试2、AOI 后决定的最终下料编码 |

## 2. 总体架构与职责

### 2.1 系统拓扑

```text
Handler上位机 <==== 实时TCP/JSON主链路 ====> AOI Gateway
                                              |
                                              +-> CaptureController
                                              |    -> PIS-IN相机SDK
                                              |    -> R/G/B/RING图像
                                              |
                                              +-> AttachmentVerifier
                                              |    -> 稳定性/解码/SHA256
                                              |
                                              +-> PisInSourceAdapter
                                              |    -> 内部数据标准化
                                              |
                                              +-> InferenceOrchestrator
                                              |    -> YOLOv8/TensorRT
                                              |
                                              +-> BinPolicy
                                              |    -> AOI BIN
                                              |
                                              +-> ResultOutbox -> Handler上位机
                                              |
                                              +-> MesOutbox ----> MES（异步）
```

### 2.2 模块职责

| 模块 | 职责 |
|---|---|
| HandlerGateway | 建立 TCP 连接、解析协议、持久化周期、维护单 Site 状态机 |
| CaptureController | 调用实际 PIS-IN 相机 SDK，完成多光源采集 |
| AttachmentVerifier | 等待文件稳定、图像解码、计算实际 SHA256、校验光源集合 |
| PisInSourceAdapter | 将采集结果标准化为 AOI 内部数据对象，不直接控制相机 |
| InferenceOrchestrator | 调用 YOLOv8/TensorRT，保存不可变推理证据 |
| BinPolicy | 根据模型结果和安全规则生成版本化 AOI BIN |
| ResultOutbox | 持久化并可靠投递 RESULT，处理 ACK、重试和不确定状态 |
| MesOutbox | 异步发布 MES，失败不影响 Handler 主链路 |
| PostgreSQL | 保存检测周期、附件、推理结果、Outbox 和审计记录 |

## 3. TCP/JSON 线协议

### 3.1 协议参数

| 项目 | 规范 |
|---|---|
| 传输层 | TCP/IP |
| 应用层 | 私有 JSON 文本协议，不等同于 SECS/GEM |
| 字符编码 | UTF-8，无 BOM |
| 连接方式 | 长连接；Handler 为 Server，AOI Gateway 为 Client |
| 默认端口 | 9000，设备侧确认后固化 |
| 帧边界 | 单行 JSON，以 `\n` 结束 |
| 最大报文 | 256 KiB，超限立即拒绝 |
| 线协议版本 | `1.4` |

接收端必须按换行符处理 TCP 粘包和半包，并在缓冲区达到 256 KiB 仍未发现换行符时关闭连接、记录协议告警。JSON 字符串中的换行必须按 JSON 规则转义。

### 3.2 通用消息头

| 字段 | 类型 | 业务消息 | 连接级消息 | 说明 |
|---|---|---:|---:|---|
| protocol_version | string | 必填 | 必填 | 固定为 `1.4` |
| message_id | string | 必填 | 必填 | UUIDv7；同一逻辑消息重试时保持不变 |
| trace_id | string | 必填 | 禁止 | Handler 生成的芯片追溯标识 |
| timestamp | string | 必填 | 必填 | UTC ISO 8601，毫秒精度，使用 `Z` |
| sender | string | 必填 | 必填 | 发送设备标识 |
| receiver | string | 必填 | 必填 | 接收设备标识 |
| type | string | 必填 | 必填 | 消息类型 |
| payload | object | 必填 | 必填 | 消息负载 |
| auth | object | 生产必填 | 生产必填 | HMAC 认证信息；实验室模式可关闭 |

业务消息类型：

```text
START
START_ACK
RESULT
RESULT_ACK
CANCEL
CANCEL_ACK
```

连接级消息类型：

```text
HEARTBEAT
HEARTBEAT_ACK
SESSION_SYNC
SESSION_SYNC_ACK
```

`ERROR` 可以是业务级或连接级消息：能够识别原业务上下文时必须携带 `trace_id`；协议解析、版本协商或连接认证失败且无法识别业务上下文时必须省略 `trace_id`。

### 3.3 消息 ID 规则

每条逻辑消息必须具有独立的 `message_id`：

```text
START      = MSG-A
START_ACK  = MSG-B，acknowledged_message_id = MSG-A
RESULT     = MSG-C
RESULT_ACK = MSG-D，acknowledged_message_id = MSG-C
```

网络重试必须重发原始消息，保持相同 `message_id`、`trace_id`、`cycle_id` 和业务内容。禁止通过生成新消息 ID 的方式重试同一个 RESULT。

### 3.4 业务键与幂等键

数据库使用以下键：

| 键 | 用途 |
|---|---|
| event_uuid | AOI 内部主键 |
| trace_id | Handler 业务唯一键 |
| source_key_hash | 物理 AOI 周期幂等关联键 |

`source_key_hash` 使用规范化 JSON 和 SHA256 生成，字段为：

```json
{
  "handler_id": "HANDLER-01",
  "handler_session_id": "HS-20260810-01",
  "cycle_id": "CYCLE-00001234",
  "trace_id": "DEV-BGA256-20260810-S001-T1001-01",
  "station_code": "AOI",
  "surface": "TOP"
}
```

规范化规则：键名升序、UTF-8、无多余空格、空字段不得参与计算。

数据库唯一约束：

```text
UNIQUE(handler_id, handler_session_id, cycle_id)
UNIQUE(trace_id)
UNIQUE(source_key_hash)
```

幂等判定：

| 场景 | 处理 |
|---|---|
| 相同 message_id、相同内容 | 返回原响应，不重复执行业务 |
| 相同 trace_id、相同周期上下文 | 返回原 START_ACK 或原 RESULT |
| 相同 trace_id、不同产品/料盘/槽位/周期上下文 | 证据冲突，拒绝并告警 |
| 已完成 trace_id 再次提交 | 返回原状态或原结果，不重新推理 |

### 3.5 TraceID 规则

1. TraceID 由 Handler 生成并保证跨 Handler 全局唯一，AOI 将其视为不透明业务标识。
2. 长度为 1-128 个字符。
3. 允许字符为 `A-Z`、`a-z`、`0-9`、短横线、下划线和点。
4. AOI 不得根据时间戳或文件名推测 TraceID。
5. TraceID 不直接作为数据库主键。

### 3.6 超时与时钟

| 参数 | 默认值 | 生产要求 |
|---|---:|---|
| tcp_connect_timeout_ms | 5000 | 设备侧确认 |
| start_ack_timeout_ms | 500 | 设备侧确认 |
| capture_timeout_ms | 由产品配置 | 设备侧确认 |
| inference_timeout_ms | 由模型配置 | 真实 TensorRT 实测 |
| result_ack_timeout_ms | 由设备配置 | 设备侧确认 |
| handler_max_wait_time_ms | 无默认值 | Handler 供应商必须提供 |
| handler_result_max_retries | 3 | 双方确认 |
| mes_max_retries | 10 | MES 方确认 |

性能门禁：

```text
AOI总耗时 <= handler_max_wait_time_ms
AOI P95总耗时 <= handler_max_wait_time_ms * 80%
```

性能拆分为三个指标：`result_send_ms` 从 Gateway 收到完整 START 帧到 RESULT 完整写入 TCP；`result_ack_ms` 从收到 START 到收到 RESULT_ACK；`handler_receive_ms` 由 Handler 在 RESULT_ACK 的 `received_at` 中提供。Handler 时间戳只用于审计和跨系统统计，不参与 Gateway 本地超时计算。

Handler 与 AOI Gateway 使用 NTP 同步 UTC 时间，审计时间误差不得超过正负 1 秒。业务超时必须使用单调时钟，不得依赖可跳变的系统墙上时钟。

### 3.7 心跳与重连

| 参数 | 值 |
|---|---:|
| 心跳间隔 | 30 秒 |
| 心跳超时 | 90 秒 |
| 重连退避 | 1、2、4、8 秒，最大 60 秒 |

AOI Gateway 发送 HEARTBEAT，Handler 必须返回 HEARTBEAT_ACK。连续三次未收到匹配 ACK，Gateway 判定连接不可用并进入重连流程。

每次 TCP 连接建立后，包括首次连接和断线重连，Gateway 都必须先完成 SESSION_SYNC。同步完成前连接状态为 `SYNCING`，双方禁止处理 START、RESULT 和积压补发。

### 3.8 安全认证

联调环境最低要求：

1. 独立生产 VLAN 或隔离网络；
2. 双方 IP 白名单和端口 ACL；
3. 每台设备独立预共享密钥；
4. HMAC-SHA256 消息签名；
5. 时间窗和 nonce 防重放。

`auth` 示例：

```json
{
  "key_id": "HANDLER-01-KEY-01",
  "nonce": "019cd123-4567-7000-8000-123456789000",
  "signature": "64位小写十六进制HMAC-SHA256"
}
```

签名输入为不含 `auth.signature` 的完整消息对象，并按 RFC 8785 JSON Canonicalization Scheme 规范化为 UTF-8 字节后计算 HMAC-SHA256。接收端必须校验 `key_id`、签名、时间窗和 nonce；同一 `key_id + nonce` 在 5 分钟窗口内只能接受一次。生产环境优先采用双向 TLS；若供应商无法支持 TLS/HMAC，必须形成书面风险接受记录。

## 4. 单 Site 周期管理

### 4.1 硬性约束

```text
max_inflight_cycles = 1
```

1. 同一 Handler 会话同时只允许一个活动检测周期。
2. 当前周期未进入 COMPLETED、EXPIRED 或 CANCELLED 前，不接受不同 TraceID 的 START。
3. 重复的相同 START 返回原 START_ACK。
4. 活动周期收到不同 TraceID 的 START，返回 `ACTIVE_CYCLE_EXISTS`。
5. Handler 必须等待当前周期明确终结后才能放入下一颗芯片。
6. Handler 重启必须生成新的 HandlerSessionID，旧会话中的未完成周期失效。
7. 迟到 RESULT 只能归档，不能驱动新周期的物理分选。

### 4.2 正常状态机

```text
IDLE
  -> START_RECEIVED
  -> START_ACKED
  -> COLLECTING
  -> VALIDATED
  -> INFERENCING
  -> RESULT_READY
  -> RESULT_SENT
  -> ACKED
  -> COMPLETED
  -> IDLE
```

### 4.3 异常结果状态机

采图超时：

```text
COLLECTING
  -> SYSTEM_RESULT_READY(BIN 292)
  -> RESULT_SENT
  -> ACKED
  -> COMPLETED
```

附件校验失败或 TraceID 冲突：

```text
COLLECTING/VALIDATED
  -> QUARANTINED
  -> SYSTEM_RESULT_READY(BIN 290或293)
  -> RESULT_SENT
  -> ACKED
  -> COMPLETED
```

模型不可用：

```text
INFERENCING
  -> SYSTEM_RESULT_READY(BIN 291)
  -> RESULT_SENT
  -> ACKED
  -> COMPLETED
```

RESULT_ACK 超时：

```text
RESULT_SENT
  -> RETRYING（重发完全相同的RESULT）
  -> ACKED
```

重试耗尽：

```text
RETRYING
  -> DELIVERY_UNCERTAIN
  -> 人工告警或安全停站
```

CANCEL 与 RESULT 竞争时，状态更新必须使用数据库行锁或 compare-and-set 原子转换：

| 当前状态 | 收到 CANCEL 后的唯一合法结果 |
|---|---|
| START_ACKED / COLLECTING / VALIDATED | CANCELLED，不发送 RESULT |
| INFERENCING | 记录取消意图，内部推理可安全结束，但不发送 RESULT，最终为 CANCELLED |
| RESULT_READY | CANCELLED，不发送 RESULT |
| RESULT_SENT / RETRYING | CANCEL_PENDING；等待 ACK 或超时 |
| ACKED / COMPLETED | 返回 ALREADY_COMPLETED，不改变状态 |

`CANCEL_PENDING` 收到匹配 RESULT_ACK 后进入 ACKED；达到结果投递截止时间仍未收到 ACK 时进入 CANCELLED。单个周期不得同时拥有 ACKED 和 CANCELLED 两个终态。

完全断线且 Handler 已结束当前周期时：

```text
任意非终态
  -> EXPIRED
```

### 4.4 持久化边界

1. Gateway 必须先持久化 START 和周期状态，再发送 START_ACK。
2. 推理结果、AOI BIN 和 Handler ResultOutbox 必须在同一数据库事务中写入。
3. MES Outbox 可在同一事务中写入，但 MES 投递不得阻塞 Handler RESULT。
4. AI 原始结果不可被人工复核覆盖；人工决策和最终决策必须使用独立字段。

### 4.5 数据模型要求

`inspection_events` 至少增加：

```text
trace_id                    VARCHAR(128) UNIQUE
handler_session_id          VARCHAR(64)
cycle_id                    VARCHAR(64)
capture_id                  UUID
camera_trigger_sequence     BIGINT
capture_started_at          TIMESTAMPTZ
capture_completed_at        TIMESTAMPTZ
cycle_deadline_at            TIMESTAMPTZ
aoi_bin                     INTEGER
result_category             VARCHAR(16)
final_decision              VARCHAR(16)
handler_publish_status      VARCHAR(32) DEFAULT 'NOT_READY'
mes_publish_status          VARCHAR(32) DEFAULT 'NOT_READY'
updated_at                  TIMESTAMPTZ
```

START 阶段尚无推理结果，因此 `ai_decision`、`ai_confidence`、`defect_code`、`reason_code` 和 `image_url` 必须允许 NULL。生成 RESULT 后由事务填充这些字段，并把 Outbox 状态从 NOT_READY 转为 PENDING。

另建 `handler_result_outbox`、`mes_outbox` 和 `handler_protocol_messages`，分别保存实时投递、MES 投递和协议幂等响应。不得把重试状态压入 `inspection_events` 的单一 JSON 字段。

## 5. 消息定义

### 5.1 START：Handler到AOI

```json
{
  "protocol_version": "1.4",
  "message_id": "019cd123-4567-7000-8000-123456789001",
  "trace_id": "DEV-BGA256-20260810-S001-T1001-01",
  "timestamp": "2026-08-10T10:05:20.000Z",
  "sender": "HANDLER-01",
  "receiver": "AOI-GW-01",
  "type": "START",
  "payload": {
    "handler_id": "HANDLER-01",
    "handler_session_id": "HS-20260810-01",
    "cycle_id": "CYCLE-00001234",
    "station_code": "AOI",
    "product_id": "BGA-256",
    "batch_id": "LOT-20260810",
    "tray_id": "TRAY-1001",
    "slot_index": "01",
    "surface": "TOP"
  },
  "auth": {
    "key_id": "HANDLER-01-KEY-01",
    "nonce": "019cd123-4567-7000-8000-123456789101",
    "signature": "<hmac-sha256>"
  }
}
```

START 不得携带 attachments、file_uri、file_hash 或 light_id。

### 5.2 START_ACK：AOI到Handler

```json
{
  "protocol_version": "1.4",
  "message_id": "019cd123-4567-7000-8000-123456789002",
  "trace_id": "DEV-BGA256-20260810-S001-T1001-01",
  "timestamp": "2026-08-10T10:05:20.050Z",
  "sender": "AOI-GW-01",
  "receiver": "HANDLER-01",
  "type": "START_ACK",
  "payload": {
    "acknowledged_message_id": "019cd123-4567-7000-8000-123456789001",
    "cycle_id": "CYCLE-00001234",
    "status": "ACCEPTED",
    "reject_code": null,
    "estimated_time_ms": 1500
  },
  "auth": {
    "key_id": "AOI-GW-01-KEY-01",
    "nonce": "019cd123-4567-7000-8000-123456789102",
    "signature": "<hmac-sha256>"
  }
}
```

START_ACK 状态为 `REJECTED` 时必须填写 `reject_code`：

```text
INVALID_MESSAGE
INVALID_TRACE_ID
ACTIVE_CYCLE_EXISTS
EVIDENCE_CONFLICT
STATION_UNAVAILABLE
PROTOCOL_VERSION_UNSUPPORTED
AUTHENTICATION_FAILED
```

### 5.3 RESULT：AOI到Handler

```json
{
  "protocol_version": "1.4",
  "message_id": "019cd123-4567-7000-8000-123456789003",
  "trace_id": "DEV-BGA256-20260810-S001-T1001-01",
  "timestamp": "2026-08-10T10:05:22.123Z",
  "sender": "AOI-GW-01",
  "receiver": "HANDLER-01",
  "type": "RESULT",
  "payload": {
    "cycle_id": "CYCLE-00001234",
    "event_uuid": "52a2492e-c0b0-4fb9-b13e-5f781ac1f121",
    "aoi_decision": "FAIL",
    "result_category": "QUALITY",
    "aoi_bin": 201,
    "defect_code": "BALL_BRIDGE",
    "confidence": 0.964,
    "model_version": "yolov8s-aoi-v1.2",
    "policy_version": "bin-policy-v1.0",
    "requires_review": false,
    "reason_code": "DEFECT_SCORE",
    "defect_positions": [
      {
        "light_id": "RING",
        "coordinate_unit": "PIXEL",
        "origin": "TOP_LEFT",
        "image_width": 1920,
        "image_height": 1080,
        "x": 128,
        "y": 256,
        "width": 32,
        "height": 32
      }
    ]
  },
  "auth": {
    "key_id": "AOI-GW-01-KEY-01",
    "nonce": "019cd123-4567-7000-8000-123456789103",
    "signature": "<hmac-sha256>"
  }
}
```

RESULT 一致性规则：

| aoi_decision | result_category | 允许 BIN | defect_code | requires_review |
|---|---|---|---|---:|
| PASS | QUALITY | 100 | 必须为空 | false |
| FAIL | QUALITY | 201-205 | 必填 | false |
| REVIEW | QUALITY | 280 | 可选 | true |
| REVIEW | SYSTEM | 290-293、299 | 可选 | true |

任何不符合该表的 RESULT 均为协议错误。SYSTEM BIN 使用 `aoi_decision=REVIEW`，以兼容当前项目的 PASS/FAIL/REVIEW 决策模型；系统故障率按 `result_category=SYSTEM` 统计，不得计入人工复核率或质量缺陷率。

### 5.4 RESULT_ACK：Handler到AOI

```json
{
  "protocol_version": "1.4",
  "message_id": "019cd123-4567-7000-8000-123456789004",
  "trace_id": "DEV-BGA256-20260810-S001-T1001-01",
  "timestamp": "2026-08-10T10:05:22.180Z",
  "sender": "HANDLER-01",
  "receiver": "AOI-GW-01",
  "type": "RESULT_ACK",
  "payload": {
    "cycle_id": "CYCLE-00001234",
    "acknowledged_message_id": "019cd123-4567-7000-8000-123456789003",
    "status": "ACCEPTED",
    "reject_code": null,
    "received_at": "2026-08-10T10:05:22.180Z",
    "processing_started_at": "2026-08-10T10:05:22.181Z",
    "sorting_completed_at": "2026-08-10T10:05:22.300Z"
  },
  "auth": {
    "key_id": "HANDLER-01-KEY-01",
    "nonce": "019cd123-4567-7000-8000-123456789104",
    "signature": "<hmac-sha256>"
  }
}
```

Handler 收到同一 RESULT `message_id` 时必须返回与首次处理一致的 RESULT_ACK，不得重复执行分选动作。`REJECTED` 时必须填写 `reject_code`，且不得用另一个 AOI BIN 替换原结果。Gateway 收到 `REJECTED` 后进入 `DELIVERY_UNCERTAIN`，触发人工告警或安全停站，不得自动开始下一周期。

`received_at` 为必填；`processing_started_at` 和 `sorting_completed_at` 可选。Handler 时间仅用于审计和跨系统延迟分析，Gateway 的超时判断仍使用本地单调时钟。

### 5.5 HEARTBEAT与HEARTBEAT_ACK

HEARTBEAT 为连接级消息，不得包含 `trace_id`：

```json
{
  "protocol_version": "1.4",
  "message_id": "019cd123-4567-7000-8000-123456789005",
  "timestamp": "2026-08-10T10:05:30.000Z",
  "sender": "AOI-GW-01",
  "receiver": "HANDLER-01",
  "type": "HEARTBEAT",
  "payload": {
    "gateway_status": "READY"
  },
  "auth": {
    "key_id": "AOI-GW-01-KEY-01",
    "nonce": "019cd123-4567-7000-8000-123456789105",
    "signature": "<hmac-sha256>"
  }
}
```

HEARTBEAT_ACK 的 payload 必须包含 `acknowledged_message_id` 和 Handler 当前状态。

### 5.6 SESSION_SYNC与SESSION_SYNC_ACK

重连成功后 AOI Gateway 发送 SESSION_SYNC：

```json
{
  "protocol_version": "1.4",
  "message_id": "019cd123-4567-7000-8000-123456789006",
  "timestamp": "2026-08-10T10:06:00.000Z",
  "sender": "AOI-GW-01",
  "receiver": "HANDLER-01",
  "type": "SESSION_SYNC",
  "payload": {
    "gateway_id": "AOI-GW-01",
    "last_known_handler_session_id": "HS-20260810-01",
    "pending_cycle_id": "CYCLE-00001234",
    "pending_trace_id": "DEV-BGA256-20260810-S001-T1001-01",
    "pending_result_message_id": "019cd123-4567-7000-8000-123456789003"
  },
  "auth": {
    "key_id": "AOI-GW-01-KEY-01",
    "nonce": "019cd123-4567-7000-8000-123456789106",
    "signature": "<hmac-sha256>"
  }
}
```

Handler 返回 SESSION_SYNC_ACK：

```json
{
  "protocol_version": "1.4",
  "message_id": "019cd123-4567-7000-8000-123456789007",
  "timestamp": "2026-08-10T10:06:00.050Z",
  "sender": "HANDLER-01",
  "receiver": "AOI-GW-01",
  "type": "SESSION_SYNC_ACK",
  "payload": {
    "acknowledged_message_id": "019cd123-4567-7000-8000-123456789006",
    "handler_session_id": "HS-20260810-01",
    "active_cycle_id": "CYCLE-00001234",
    "active_trace_id": "DEV-BGA256-20260810-S001-T1001-01",
    "cycle_status": "WAITING_RESULT"
  },
  "auth": {
    "key_id": "HANDLER-01-KEY-01",
    "nonce": "019cd123-4567-7000-8000-123456789107",
    "signature": "<hmac-sha256>"
  }
}
```

只有满足以下全部条件，Gateway 才允许补发 RESULT：

```text
handler_session_id与原会话相同
active_cycle_id与待发送周期相同
active_trace_id与待发送TraceID相同
cycle_status = WAITING_RESULT
```

其他情况只能将待发送结果归档为 EXPIRED，禁止驱动物理分选。

### 5.7 CANCEL与CANCEL_ACK

Handler 因急停、夹具异常、人工终止或周期超时时发送 CANCEL。payload 必须包含 `cycle_id` 和机器可读的 `cancel_reason`。

```json
{
  "protocol_version": "1.4",
  "message_id": "019cd123-4567-7000-8000-123456789008",
  "trace_id": "DEV-BGA256-20260810-S001-T1001-01",
  "timestamp": "2026-08-10T10:05:21.000Z",
  "sender": "HANDLER-01",
  "receiver": "AOI-GW-01",
  "type": "CANCEL",
  "payload": {
    "cycle_id": "CYCLE-00001234",
    "cancel_reason": "HANDLER_TIMEOUT"
  },
  "auth": {
    "key_id": "HANDLER-01-KEY-01",
    "nonce": "019cd123-4567-7000-8000-123456789108",
    "signature": "<hmac-sha256>"
  }
}
```

CANCEL 处理规则：

1. 未开始采图：立即终止并进入 CANCELLED。
2. 正在采图或推理：记录取消意图，允许内部操作安全结束，但禁止向 Handler 发布 RESULT。
3. RESULT 尚未发送：原子转换为 CANCELLED，不发送 RESULT。
4. RESULT 已发送：原子转换为 CANCEL_PENDING，不得生成第二个 RESULT。
5. RESULT 已 ACKED 或周期已 COMPLETED：CANCEL_ACK 返回 `ALREADY_COMPLETED`。
6. CANCEL_ACK 必须包含原 `message_id`、`cycle_id` 和当前状态。

### 5.8 ERROR

ERROR 用于协议错误、鉴权错误和不可恢复系统错误。能够识别原消息时，payload 应包含 `failed_message_id`。

| 错误码 | 含义 | 建议处理 |
|---|---|---|
| E1001 | 图像 Hash 校验失败 | BIN 290，隔离 |
| E1002 | 图像读取或解码失败 | 重试一次，仍失败则 BIN 290 |
| E1003 | TraceID 格式无效 | START_ACK REJECTED |
| E1004 | 相同 TraceID 证据冲突 | BIN 293，最高告警 |
| E1005 | 周期已过期 | 拒绝，不重新推理 |
| E1006 | 当前存在活动周期 | START_ACK REJECTED |
| E1101 | JSON/协议结构无效 | ERROR，记录协议告警 |
| E1102 | 协议版本不支持 | ERROR，关闭连接 |
| E1103 | 身份认证或签名失败 | ERROR 或立即关闭连接 |
| E2001 | TensorRT 推理超时 | BIN 292 |
| E2002 | 模型加载失败 | BIN 291，暂停工站 |
| E3001 | Handler 连接不可用 | Handler 执行本地安全动作 |
| E3002 | RESULT_ACK 超时 | 重发原 RESULT，不生成新 BIN |
| E9001 | 未知内部错误 | BIN 299，暂停工站 |

## 6. AOI采集与推理内部契约

### 6.1 图像采集

1. CaptureController 在 START_ACK 持久化并发送后触发采集。
2. 默认必需光源集合为 `R/G/B/RING`，实际集合可按产品和工站配置。
3. 每个必需光源只能出现一次。
4. 图像路径和 Hash 仅在 AOI 内部流通，不进入 Handler 线协议。
5. 图像文件必须通过稳定写入、可解码、尺寸合法和实际 SHA256 重算。
6. 缺失、重复、未知或冲突光源不得进入 TensorRT 推理。

每次采集必须先创建并持久化采集身份：

| 字段 | 类型 | 说明 |
|---|---|---|
| capture_id | UUID | AOI 为当前 CycleID 生成的采集标识 |
| camera_trigger_sequence | uint64 | 相机 SDK 返回的单调递增触发序号，禁止应用模拟自增 |
| start_received_monotonic_ns | uint64 | Gateway 收到 START 的本地单调时钟值 |
| capture_started_monotonic_ns | uint64 | 本地开始采集的单调时钟值 |
| capture_started_at | UTC timestamp | 审计时间，不用于本地先后顺序判断 |
| capture_completed_at | UTC timestamp | 采集完成审计时间 |
| trigger_source | enum | HANDLER_START / MANUAL / RETRY |

采集归属硬约束：

1. `capture_id` 必须在触发相机前与当前 `event_uuid + cycle_id` 原子绑定。
2. R/G/B/RING 必须属于同一个 `capture_id` 和同一个 `camera_trigger_sequence`。
3. `capture_started_monotonic_ns` 必须大于等于当前周期的 `start_received_monotonic_ns`。
4. 相机 SDK 必须确认触发序号；若 SDK 无法回传自定义 capture_id，则由触发序号与当前单 Site 活动周期建立唯一映射。
5. 触发前必须清空或显式跳过旧帧，禁止把相机缓存中的上一周期图像绑定到当前周期。
6. 任一归属校验失败均进入 QUARANTINED，并生成 BIN 293，禁止 PASS。

### 6.2 输入指纹

AOI 使用排序后的 `(data_type, light_id, file_hash)` 元组计算输入指纹。同一模型版本、策略版本和输入指纹的推理结果不可变。

### 6.3 推理安全门禁

模型 PASS 必须同时满足：

```text
检测周期状态 = VALIDATED
必需光源集合完整
文件校验全部通过
真实TensorRT Engine可用
活动模型和策略版本匹配
生产自动PASS开关已通过发布审批
```

任一条件不满足时，最终结果只能为 REVIEW 或 SYSTEM BIN。

### 6.4 运行模式与外部依赖

运行模式沿用项目既有定义：

| APP_MODE | 行为 |
|---|---|
| demo | 允许模拟器和 DemoInferenceAdapter，仅用于演示 |
| shadow | 使用可信采集和真实适配器，但 AI PASS 的最终结果保持 REVIEW |
| controlled | 仅在发布门禁获批且 AUTO_PASS_ENABLED=true 时允许自动 PASS |

生产部署不得使用 `APP_MODE=demo`。生产 START Schema 不定义 `scenario`，并配置 Pydantic `extra="forbid"`，任何包含 `Scenario/scenario` 的请求直接返回 422。不得通过读取未声明字段后再判断的方式实现隔离。

TensorRT Engine、CUDA Runtime、GPU、PIS-IN 相机 SDK 和设备驱动属于现场外部依赖。在这些依赖未提供前，软件只能完成适配器、模拟器和 fail-closed 验证，不得宣称真实推理或真实采集已通过。

## 7. AOI BIN规范

| BIN | 分类 | 含义 | Handler处理 |
|---:|---|---|---|
| 100 | PASS | AOI 检测通过 | 进入下一工站或良品下料 |
| 201 | FAIL | 桥连 Ball Bridge | AOI_NG 隔离 |
| 202 | FAIL | 缺球 Missing Ball | AOI_NG 隔离 |
| 203 | FAIL | 共面度异常 Coplanarity | AOI_NG 隔离 |
| 204 | FAIL | 锡球尺寸异常 | AOI_NG 隔离 |
| 205 | FAIL | 偏移 Offset | AOI_NG 隔离 |
| 280 | REVIEW | 待人工复核 | 复核料盒 |
| 290 | SYSTEM | 输入无效、格式/Hash/光源错误 | 隔离 |
| 291 | SYSTEM | TensorRT Engine 或模型不可用 | 隔离或暂停工站 |
| 292 | SYSTEM | AOI 采图或处理超时 | 隔离 |
| 293 | SYSTEM | TraceID 或证据关联冲突 | 隔离并触发最高告警 |
| 299 | SYSTEM | 未知系统错误 | 隔离并暂停工站 |

RESULT_ACK 超时不属于产品检测结果，不分配 BIN。Gateway 只能重发原 RESULT，禁止生成替代 BIN。

Final BIN 由 Handler 综合测试1、测试2和 AOI 结果决定，AOI 不得覆盖其他工站结果。

## 8. 结果发布与MES旁路

### 8.1 Handler ResultOutbox

状态：

```text
PENDING
SENDING
SENT
ACKED
RETRYING
DELIVERY_UNCERTAIN
EXPIRED
```

含义：

| 状态 | 含义 |
|---|---|
| PENDING | 已持久化，等待发送 |
| SENDING | 正在写入 TCP |
| SENT | 完整帧已写入 TCP，尚未收到 ACK |
| ACKED | 收到匹配且已接受的 RESULT_ACK |
| RETRYING | ACK 超时，重发完全相同 RESULT |
| DELIVERY_UNCERTAIN | 重试耗尽或 RESULT 被拒绝，无法确认 Handler 是否执行；必须人工处置或安全停站 |
| EXPIRED | Handler 周期已失效，禁止补发 |

Handler ResultOutbox 默认最多重试 3 次。重试耗尽后必须告警或安全停站，不得无限重试。

重试默认采用 100 ms、200 ms、400 ms 指数退避。每次重试前必须同时满足：

```text
retry_count < handler_result_max_retries
当前本地单调时间 + next_retry_delay < cycle_deadline
当前Handler会话和周期仍有效
```

任一条件不满足时进入 DELIVERY_UNCERTAIN，并记录 `delivery_uncertain_at`。`cycle_deadline_at` 使用 UTC 持久化用于重启恢复；进程存活期间使用本地单调时钟执行精确预算。重启后如果无法可靠证明周期仍有效，必须先 SESSION_SYNC，不能直接重试。

### 8.2 MesOutbox

MES 状态独立为：

```text
PENDING
SENDING
SENT
ACKED
RETRYING
DEAD_LETTER
```

MES 默认最多重试 10 次，重试耗尽进入 DEAD_LETTER，等待人工补偿。MES 不可用不得改变 Handler ResultOutbox 状态。

MES 报文至少包含：

```json
{
  "schema_version": "1.0",
  "message_id": "019cd123-4567-7000-8000-123456789009",
  "event_uuid": "52a2492e-c0b0-4fb9-b13e-5f781ac1f121",
  "event_revision": 1,
  "event_type": "AOI_RESULT",
  "trace_id": "DEV-BGA256-20260810-S001-T1001-01",
  "cycle_id": "CYCLE-00001234",
  "station_code": "AOI",
  "aoi_decision": "FAIL",
  "result_category": "QUALITY",
  "aoi_bin": 201,
  "defect_code": "BALL_BRIDGE",
  "confidence": 0.964,
  "model_version": "yolov8s-aoi-v1.2",
  "policy_version": "bin-policy-v1.0",
  "detected_at": "2026-08-10T10:05:22.123Z"
}
```

`message_id` 是 MES 消息幂等键；`event_uuid` 是业务聚合键；`event_uuid + event_revision` 标识事件版本。`event_type` 至少支持 `AOI_RESULT`、`REVIEW_CONFIRMED` 和 `RESULT_CORRECTED`，使人工复核或订正不会被 MES 当成重复消息丢弃。

### 8.3 机器可执行契约

双方必须共享并版本化以下 JSON Schema：

```text
schemas/handler_aoi/v1_4/common.schema.json
schemas/handler_aoi/v1_4/start.schema.json
schemas/handler_aoi/v1_4/start_ack.schema.json
schemas/handler_aoi/v1_4/result.schema.json
schemas/handler_aoi/v1_4/result_ack.schema.json
schemas/handler_aoi/v1_4/heartbeat.schema.json
schemas/handler_aoi/v1_4/session_sync.schema.json
schemas/handler_aoi/v1_4/cancel.schema.json
schemas/handler_aoi/v1_4/error.schema.json
```

Schema 必须定义 `additionalProperties=false`、字段长度、枚举、数值范围、条件必填和协议版本。Handler 模拟器、AOI Gateway 与契约测试共用同一套文件。

## 9. 故障处理矩阵

| 场景 | Handler行为 | AOI行为 | MES行为 |
|---|---|---|---|
| 采图超时 | 接收 BIN 292 并隔离 | 生成安全结果 | 记录异常 |
| 图像无效 | 接收 BIN 290 并隔离 | 隔离证据并生成安全结果 | 记录异常 |
| TraceID/证据冲突 | 接收 BIN 293 并隔离 | 最高告警 | 记录关联异常 |
| TensorRT 不可用 | 接收 BIN 291 | 隔离或暂停工站 | 记录模型异常 |
| Handler 断线 | 本地执行安全动作 | 持久化结果，完成 SESSION_SYNC 后决定是否补发 | 不受影响 |
| RESULT_ACK 丢失 | 对重复 RESULT 返回相同 ACK | 重发相同 RESULT | 不受影响 |
| MES 断线 | 不受影响 | Handler 主链路继续运行 | Outbox 重试 |
| Handler 重启 | 新建会话，旧周期失效 | 旧结果归档 EXPIRED | 记录重启事件 |
| 急停/取消 | 发送 CANCEL | 停止发布 RESULT | 记录取消原因 |
| Agent/RAG 不可用 | 不受影响 | 报告降级 | 记录服务不可用 |

## 10. 验收标准

### 10.1 协议测试

1. TCP 粘包、半包、断包和超长报文；
2. 非法 JSON、未知消息类型和不支持协议版本；
3. 签名错误、nonce 重放和时间窗超限；
4. START、RESULT、ACK 幂等重放；
5. ACK 丢失和断线重连；
6. SESSION_SYNC 后允许或禁止补发；
7. CANCEL 各阶段处理；
8. 单 Site 活动周期门禁。

### 10.2 AOI故障注入

1. 缺少 R/G/B/RING 中任一光源；
2. 重复光源；
3. 图像解码失败；
4. SHA256 不匹配；
5. 迟到文件；
6. TraceID 冲突；
7. TensorRT 超时、Engine 缺失和模型加载失败；
8. API、Gateway 和数据库重启；
9. Handler 与 MES 分别断网；
10. 迟到 RESULT 不得驱动下一颗芯片。

### 10.3 一票否决门禁

| 指标 | 目标 |
|---|---:|
| TraceID 静默错配率 | 0 |
| 不完整输入产生 PASS 数 | 0 |
| 推理不可用产生 PASS 数 | 0 |
| 同一 TraceID 矛盾 BIN 数 | 0 |
| Handler 结果丢失数 | 0 |
| 重复消息导致错误分选数 | 0 |
| 迟到 RESULT 驱动错误分选数 | 0 |
| MES 故障影响 Handler 主链路次数 | 0 |

验收统计必须同时满足连续运行不少于 24 小时和有效样本不少于 10,000 颗；若现场 24 小时产量不足 10,000，则继续运行至样本量达标。所有门禁按完整窗口计算，不得只报告抽样片段。

### 10.4 运维门禁

| 项目 | 基线要求 |
|---|---|
| 原始图像保留 | `IMAGE_RETENTION_DAYS=30`，到期归档或删除，操作留审计 |
| 磁盘告警 | 使用率达到 80% 告警，达到 90% 禁止接收新周期并安全停站 |
| PostgreSQL | 每日基础备份、连续 WAL 归档、保留 30 天、按季度恢复演练 |
| 结构化日志 | 每条业务日志包含 trace_id、cycle_id、event_uuid、message_id |
| Prometheus | 暴露周期耗时、BIN 分布、系统故障率、Outbox 堆积和磁盘水位 |
| HMAC 轮换 | 当前密钥与上一密钥并存，默认重叠 24 小时，支持即时吊销 |
| 运行模式 | 生产部署只允许 shadow 或 controlled；检测到 demo 时拒绝启动 |

## 11. 联调前置参数

| 参数 | 提供方 | 状态 |
|---|---|---|
| Handler Server IP和端口 | Handler供应商 | 待确认 |
| handler_max_wait_time_ms | Handler供应商 | 待确认 |
| start_ack_timeout_ms | Handler供应商 | 待确认 |
| result_ack_timeout_ms | Handler供应商 | 待确认 |
| Handler安全动作和停站条件 | Handler供应商 | 待确认 |
| RESULT_ACK拒绝码 | Handler供应商 | 待确认 |
| 完整AOI BIN表 | 双方 | 待确认 |
| PIS-IN相机SDK和触发接口 | AOI设备方 | 待确认 |
| R/G/B/RING实际采集耗时 | AOI设备方 | 待实测 |
| TensorRT Engine和输入规格 | AOI算法团队 | 待提供 |
| MES接口、认证和重试约束 | MES团队 | 待确认 |
| HMAC或双向TLS支持能力 | 双方 | 待确认 |
| capture_id/触发序号回传能力 | AOI设备方 | 待确认 |

在上述参数落值、协议测试通过、硬件在环测试通过之前，本规范不构成生产自动 PASS 的批准。

## 12. 版本记录与签署

| 版本 | 日期 | 变更说明 |
|---|---|---|
| V1.0 | 2026-08-10 | 初稿 |
| V1.1 | 2026-08-10 | 增加 START_ACK、多光源附件、超时和状态机 |
| V1.2 | 2026-08-10 | START 移除附件，明确 AOI 本地采集、单 Site 和 CycleID |
| V1.3 | 2026-08-10 | 修正消息ID、SYSTEM BIN、断线同步、异常状态机；增加心跳确认、CANCEL、HMAC与Outbox终态 |
| V1.4 | 2026-08-10 | 增加采集身份、首次连接同步、全局TraceID、结果分类、性能拆解、取消竞争、MES版本事件、JSON Schema、重试预算和运维门禁 |

签署前必须完成设备侧参数评审。

| 角色 | 姓名/签名 | 日期 |
|---|---|---|
| Handler供应商代表 |  |  |
| AOI系统开发负责人 |  |  |
| 设备工程师 |  |  |
| 测试负责人 |  |  |
| 项目负责人 |  |  |
