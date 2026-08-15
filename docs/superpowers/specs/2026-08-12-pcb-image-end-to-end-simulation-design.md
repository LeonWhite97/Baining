# PCB 图片端到端模拟测试设计

## 1. 目标与事实边界

本设计把 `data/external/pcb_stability_samples` 中的公开 PCB 实拍图接入现有 PIS-IN 导入 API，验证文件落盘、图片验证、附件入库、Source Key 幂等、人工复核、工站告警和异常报告的端到端闭环。

仓库当前没有真实 PIS-IN R/G/B/RING 同机位图、3D 数据、YOLO 权重或 TensorRT Engine。PCB 网络图片属于非同域测试输入，测试只证明软件链路行为，不证明真实缺陷识别准确率、误报率、漏放率或现场节拍。

## 2. 选定方案

沿用设备侧“图片先落盘，API 接收路径与哈希元数据”的现有契约。模拟驱动器从测试素材构造四光源附件，调用 `/api/v1/inspections/import/pis-in`；API 只允许读取配置的素材根目录内文件，在推理前完成文件与附件集合验证。

不新增 multipart 上传接口，不让测试脚本直接写数据库，不改变 Source Key 组成，也不增加数据库迁移。

## 3. 配置与受控文件访问

应用增加 `AOI_IMAGE_ROOT` 配置，作为可读取图片的唯一根目录。测试创建应用时可显式传入临时目录；本地模拟默认指向项目的 `data/external/pcb_stability_samples/normalized_1920x1080`，容器使用显式只读挂载目录。

验证器按以下顺序处理每个附件：

1. 将请求中的路径解析为绝对规范路径，确认它位于 `AOI_IMAGE_ROOT` 内；拒绝 `..` 越界、根目录外绝对路径和符号链接逃逸。
2. 确认文件存在、是普通文件且大小大于零。
3. 流式计算实际 SHA-256，并与请求值做不区分大小写的精确比较。
4. 使用标准图片解码器读取文件，执行完整解码校验，并记录宽、高和格式；首期仅接受 JPEG 和 PNG。
5. 要求尺寸为正、单边不超过 8192 px、总像素不超过 40,000,000，文件大小不超过 50 MiB；当前测试基线为 `1920x1080`。

Python 标准库不能可靠完成 JPEG/PNG 全量解码校验，因此引入 Pillow 作为 API 直接依赖，不引入 OpenCV。Pillow 同时用于测试中生成损坏与边界样本，版本范围固定在项目依赖中。

## 4. 附件集合与失败策略

默认必需光源集合为 `R/G/B/RING`。进入推理前必须满足：

- 四个光源各出现一次；
- 不存在未知光源或重复光源；
- 每个附件均通过路径、文件、哈希和解码校验；
- 同一事件中的文件集合稳定且可追溯。

身份字段缺失、路径越界、文件不存在、空文件、文件或像素超过上限、哈希错误、解码失败、未知/重复/缺失光源均返回 HTTP 202，并写入 `quarantine_events`。响应固定包含 `status=QUARANTINED`、`quarantine_id`、`reason_code` 和不泄露主机绝对路径的 `reason`。隔离记录的 `parse_error` 保存 `原因码: 说明`，`extracted_fields` 保存已提取身份字段和请求中声明的来源路径；不创建 `inspection_events`、`attachments` 或 `inference_results`，不进入 Demo 推理。

原因码固定覆盖：`IDENTITY_MISSING`、`PATH_OUTSIDE_ROOT`、`FILE_MISSING`、`FILE_TOO_LARGE`、`HASH_MISMATCH`、`IMAGE_DECODE_FAILED`、`IMAGE_DIMENSIONS_INVALID`、`LIGHT_SET_INVALID` 和 `IDEMPOTENCY_CONFLICT`。

## 5. 入库与幂等语义

验证成功后继续使用现有 Source Key：

`DeviceID + DeviceSessionID + InspectionSequence + TrayID + SlotIndex + Surface`

首次请求在一个事务中创建检测事件、四条附件和一条推理结果。`image_url` 固定指向该事件的 `RING` 附件预览端点。

同 Source Key 重放时：

- 光源集合、规范化文件路径和 SHA-256 与已存附件完全一致：返回 HTTP 200、同一 `event_uuid`、同一附件数量，不重复推理和写入。
- 任一附件路径、光源或哈希与已存记录冲突：返回 HTTP 202 并写入隔离记录，原因码为 `IDEMPOTENCY_CONFLICT`；旧事件保持不可变。

数据库现有 `source_key_hash` 唯一约束和附件唯一约束继续作为并发兜底。并发收到相同有效请求时，API 捕获唯一约束竞争并重新读取已提交事件，最终对调用方呈现一次创建、其余幂等成功，不产生 500。

## 6. 图片预览接口

新增 `GET /api/v1/inspections/{event_uuid}/image`，只从数据库中已验证的附件记录解析文件，不接受调用方提供任意路径。默认返回 RING 图；可选 `light_id` 仅允许事件已保存的光源。

响应使用正确的媒体类型，并设置文件内联展示。事件或光源不存在返回 404；文件在入库后消失或哈希变化返回 409，避免静默展示被替换的证据。

## 7. 模拟驱动与场景

在 `services/simulator` 增加一次性端到端驱动模式，和现有无限循环 Demo 流量发生器分离。驱动器读取素材清单，通过 HTTP 只调用公开 API，并输出 JSON 报告。

基线执行包含：

1. **有效图片入库**：使用同一张归一化图片模拟 R/G/B/RING，创建事件并验证附件数、图片预览和详情查询。
2. **幂等重放**：原请求重放，验证 HTTP 200、事件 UUID 不变、事件/附件/推理记录数不增加。
3. **顺序扰动**：打乱四光源数组顺序，仍被识别为相同幂等请求。
4. **附件冲突**：同 Source Key 替换一个光源图片或哈希，验证隔离且旧事件不变。
5. **缺失光源**：仅发送 R/G/B，验证 `LIGHT_SET_INVALID` 隔离。
6. **错误哈希**：发送合法图片和错误 SHA-256，验证 `HASH_MISMATCH` 隔离。
7. **损坏图片**：在临时受控目录创建截断 JPEG，使用其真实哈希发送，验证 `IMAGE_DECODE_FAILED` 隔离。
8. **路径越界**：请求受控根目录外文件，验证 `PATH_OUTSIDE_ROOT` 隔离。
9. **人工复核**：非同域图片事件以 `REVIEW` 场景入库，调用复核 API 确认缺陷，验证 `review_records` 和事件结果回写。
10. **告警与报告**：在独立工站 `ST-PCB-ALERT` 创建恰好 20 个唯一事件，其中 2 个显式使用 Demo `DEFECT`、18 个使用 `NORMAL`，使 AI 初判缺陷率达到 10%；此工站不执行人工复核。验证 OPEN 告警、确认告警、生成 DRAFT 报告以及报告关联 2 个 FAIL 事件。

默认运行 10 张素材各一次，并支持通过参数增加循环次数和并发数。报告包含请求总数、成功/幂等/隔离/失败数量、P50/P95 请求耗时、事件 UUID、告警 ID、报告 ID及每个断言结果。

## 8. Demo 判定限制

文件验证通过不等于图片内容通过模型识别。由于当前 `DemoInferenceAdapter` 只读取 `Scenario` 标签：

- 普通 PCB 稳定性样本默认使用 `REVIEW`，原因是非同域输入；
- 告警场景可显式使用 `DEFECT`，但报告必须标记 `synthetic_decision=true`；
- 测试报告不得将 Demo confidence、defect code 或 bbox 解释为图片推理结果。

## 9. 自动化测试

API 单元/集成测试使用临时目录和真实 JPEG/PNG 文件，覆盖：

- 根目录约束、哈希、完整解码和尺寸验证；
- 完整/缺失/重复/未知光源集合；
- 首次入库与精确幂等重放；
- 同 Source Key 附件冲突隔离；
- 图片预览成功、未知光源、文件丢失和文件被替换；
- 独立工站上的复核回写，以及另一个独立工站上的 20 件窗口告警、确认和草稿报告；
- 模拟驱动报告汇总和失败退出码。

验收命令包括 API 全量 pytest、模拟器全量 pytest，以及一次使用 10 张素材的本地端到端运行。任何断言失败时驱动器退出非零，并保留 JSON 报告供排查。

## 10. 不在本次范围

- 真实 YOLO/TensorRT 图片推理或精度评估；
- R/G/B/RING 光学变换的合成；
- 3D 量测生成与物理规则精度；
- 大文件上传、对象存储或远程 URL 拉取；
- 生产自动 PASS 授权；
- 前端新增测试控制页面。
