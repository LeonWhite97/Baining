# FC-BGA YOLO 训练微调与推理接入设计

**日期：** 2026-08-15  
**状态：** 已确认，待实施计划  
**目标仓库：** `C:\Users\Windows\Desktop\PIS-IN_AOI_AI智能质检项目`  
**参考案例：** `D:\YOLO\01_yolo_tongue_detection`

## 1. 背景与事实边界

当前 AOI 仓库具备 FastAPI、数据关联、图片证据校验、三态决策、Demo 推理适配器和 TensorRT fail-closed 桩，但没有真实训练脚本、YOLO 数据集、正式权重或可运行的视觉模型适配器。

本设计补齐以下能力：

- FC-BGA 视觉数据转换、校验、去重、下载、训练、预测、评估和导出脚本；
- R/G/B 多光源灰度三通道输入契约；
- 7 类 2D 可见缺陷契约；
- Ultralytics YOLOv8 内部 PoC 推理适配器；
- 模型元数据、类别顺序和权重哈希校验；
- 模型缺失或异常时转 REVIEW 的安全失败路径。

交付后仍不得宣称已有现场精度、生产误报率、漏放率、现场 P95、持续吞吐或 TensorRT 实测结果。公开小数据只用于验证脚本链路和辅助实验，不能替代 PIS-IN 同机位四光源现场标注集。

## 2. 已确认决策

1. 项目定位为作品集/内部 PoC，训练与 PoC 推理采用 `ultralytics` YOLOv8。
2. README 明确 Ultralytics AGPL-3.0/企业许可边界；闭源商业交付需另行确认许可。
3. 公开 CC BY 4.0 数据只进入 `public_smoke`，不作为正式 7 类精度依据。
4. 首版模型输入为同一事件 R/G/B 三张图分别灰度化后，按 R、G、B 顺序堆叠为三通道图像。
5. RING 图首版不进入模型，但仍必须通过完整性校验并作为审计和前端证据保留。
6. 采用 7 类 2D 视觉缺陷；3D、X-ray 和系统状态不混入 YOLO 类别。
7. 训练工具包放在 `tools/vision/fc_bga_yolo/`；在线适配器放在 `apps/api/app/inference/`。
8. 首版检测器不允许用“没有检测框”推导高置信度正常，因而不开放自动 PASS。

## 3. 范围

### 3.1 本次包含

- 正式 7 类数据目录模板和公开 smoke 数据目录；
- 数据来源与许可清单；
- R/G/B 灰度堆叠转换；
- YOLO 标签和数据划分校验；
- SHA-256 内容去重，默认 audit-only；
- YOLOv8n smoke 和 YOLOv8s PoC 微调入口；
- 独立 test 评估；
- 批量预测、JSONL 结果和可视化输出；
- PT 到 ONNX 导出，以及在合适主机上导出 TensorRT Engine 的入口；
- 模型包元数据与完整性校验；
- Ultralytics 在线适配器、适配器工厂和环境配置；
- 推理异常转 REVIEW；
- 单元测试、API 回归测试和文档更新。

### 3.2 本次不包含

- 伪造或生成“现场真实”FC-BGA 缺陷图片；
- 把厂商网页图片抓取后直接用于训练；
- 现场相机标定、R/G/B 亚像素配准或光源消融结论；
- 3D 共面度算法；
- X-ray 空洞、枕头效应或开路检测；
- 正式 TensorRT 性能承诺；
- 自动 PASS 开放；
- 正式生产权重或精度验收结果。

## 4. 缺陷类别契约

类别编号固定并写入正式数据 YAML、模型元数据和 API 适配器校验逻辑。发布后不得在同一模型系列内重排编号。

| ID | 缺陷代码 | 标注对象 | 首版处置 |
|---:|---|---|---|
| 0 | `BALL_BRIDGE` | 包含发生连接的相邻焊球及桥连区域的最小外接框 | 高分进入 FAIL |
| 1 | `MISSING_BALL` | 由经质量确认的阵列基准确定的缺球格位框 | 高分进入 FAIL |
| 2 | `EXTRA_BALL` | 设计阵列以外的多余焊球实体框 | 高分进入 FAIL |
| 3 | `BALL_SIZE_ABNORMAL` | 尺寸明显超出质量规范的焊球实体框 | 高分进入 FAIL |
| 4 | `BALL_OFFSET` | 相对经确认阵列格位发生超限偏移的焊球实体框 | 高分进入 FAIL |
| 5 | `BALL_SHAPE_ABNORMAL` | 毛刺、非圆、塌陷或其他表面轮廓异常的焊球实体框 | 高分进入 FAIL |
| 6 | `FOREIGN_MATERIAL` | 焊球阵列或关键基板区域内可见异物框 | 高分进入 FAIL |

标注约束：

- 正常图像使用同名空 `.txt` 标签，不增加 `NORMAL` 类。
- `MISSING_BALL` 只有在预期格位可由产品图纸、阵列模板或人工金标准明确确定时才标注；无法确定格位的样本进入人工复核，不进入监督训练。
- 同一缺陷必须在 R/G/B 共注册坐标系上使用一份标签。三张图尺寸不同或未完成同机位配准时，转换脚本拒绝处理。
- 多缺陷图像允许多个不同类别框。
- 类别边界和疑难样本最终由质量/工艺人员确认，脚本不替代金标准流程。

以下概念不作为首版 2D YOLO 类别：

| 概念 | 归属 |
|---|---|
| `NORMAL` | 空标签和决策结果 |
| `COPLANARITY` | 3D 高度/共面度规则 |
| `SOLDER_VOID`、枕头效应、开路 | X-ray 检测域 |
| `UNKNOWN` | 低置信度或闭集外观异常的 REVIEW 状态 |
| `MISSING_3D`、`MISSING_LIGHT` | 输入完整性状态 |

## 5. 公开资料与数据边界

### 5.1 视觉与分类参考

- Basler BGA 检测资料：<https://www.baslerweb.cn/zh-cn/use-cases/semicon-bga-inspections/>
  - 2D AOI：漏球、多球、尺寸异常、毛刺、偏移、阵列完整性和对齐；
  - 3D：焊球高度和共面性；
  - X-ray：空洞、枕头效应和开路。

厂商网页仅作为分类与成像参考，不由下载脚本抓取，也不进入训练数据。

### 5.2 可核验公开数据

| 数据集 | 许可 | 规模/类别 | 用途 |
|---|---|---|---|
| [BGA RAM Chips Detection](https://universe.roboflow.com/paween/bga-ram-chips-detection-t3cqn) | CC BY 4.0 | 56 张，`OK/NG` | 默认 public smoke，仅验证检测训练链路 |
| [BGA-Balls](https://universe.roboflow.com/kenshin-blirtz/bga-balls-3ihxj) | CC BY 4.0 | 54 张，`Ball` 实例分割 | 焊球定位/分割参考，不并入 7 类正式数据 |
| [Void detection on X-ray](https://universe.roboflow.com/pcbdefectdetection-2qt8g/void-detection-on-x-ray-s8gso) | CC BY 4.0 | 41 张，`chip/void` | X-ray 参考，默认不下载、不并入 2D 数据 |

`download_public_smoke.py` 只通过 Roboflow SDK 和用户提供的 `ROBOFLOW_API_KEY` 获取明确列入来源清单的数据。脚本生成访问日期、URL、许可、版本、文件哈希和用途说明。没有 API key 时明确失败，不绕过登录或抓取预览图。

## 6. 目录结构

```text
tools/vision/fc_bga_yolo/
├── README.md
├── requirements-train.txt
├── train.py
├── predict.py
├── validate_yolo_dataset.py
├── deduplicate_yolo_dataset.py
├── convert_dataset.py
├── export_model.py
├── download_models.py
├── download_public_smoke.py
├── configs/
│   ├── classes.yaml
│   ├── public_smoke.yaml
│   ├── fc_bga_defects.template.yaml
│   ├── train_smoke.yaml
│   └── train_poc.yaml
└── tests/
    ├── test_convert_dataset.py
    ├── test_validate_yolo_dataset.py
    ├── test_deduplicate_yolo_dataset.py
    ├── test_model_metadata.py
    └── test_cli_contracts.py

data/
├── external/fc_bga_public_smoke/
│   ├── README.md
│   └── sources.json
└── vision/fc_bga_defects/
    ├── README.md
    ├── manifests/
    ├── train/images/
    ├── train/labels/
    ├── val/images/
    ├── val/labels/
    ├── test/images/
    └── test/labels/
```

真实图片、标签、下载数据、权重、运行结果和 Engine 默认加入 `.gitignore`。目录中只保留 README、配置模板、来源清单和占位文件。

## 7. 正式数据清单与转换

### 7.1 输入清单

正式转换输入为 UTF-8 JSONL。每行代表一个物理样本：

```json
{"sample_id":"LOT01-TRAY02-A07","group_id":"LOT01","split":"train","images":{"R":"raw/R.png","G":"raw/G.png","B":"raw/B.png","RING":"raw/RING.png"},"label":"annotations/LOT01-TRAY02-A07.txt"}
```

字段要求：

- `sample_id` 全局唯一；
- `group_id` 表示不能跨集合泄漏的最小业务组，首版使用批次或质量批准的等价分组；
- `split` 必须是 `train`、`val` 或 `test`，转换脚本不随机重分正式数据；
- `images` 必须且只能包含 R、G、B、RING；
- `label` 指向标准 YOLO 检测标签，空文件表示正常样本。

### 7.2 R/G/B 堆叠算法

1. 对四张图执行路径、文件大小、SHA-256、JPEG/PNG 解码和尺寸校验。
2. 要求 R/G/B/RING 宽高完全一致；首版不自动缩放单个光源来掩盖配准问题。
3. 分别使用固定亮度转换规则把 R、G、B 图转成 8-bit 灰度图。
4. 按 R、G、B 顺序写入输出图像的三个通道。
5. RING 不进入输出像素，但其哈希写入来源清单。
6. 输出无损 PNG，避免额外 JPEG 压缩改变小缺陷。
7. 标签坐标保持不变。
8. 输出清单记录输入四图哈希、标签哈希、转换版本 `rgb_grayscale_stack_v1` 和输出哈希。

训练转换与在线预处理分别实现同一小型契约，并由共享黄金样本测试比较输出字节，防止训练/服务预处理偏差。

## 8. 数据校验与去重

### 8.1 数据校验

`validate_yolo_dataset.py` 检查：

- train/val/test 的 images/labels 结构；
- 图片与标签一一对应；
- JPEG/PNG 可完整解码；
- 标签每行恰好五列；
- 类别 ID 在 0 到 6；
- 坐标是有限浮点数；
- 中心点在 `[0, 1]`，宽高在 `(0, 1]`；
- 空标签被统计为正常样本但不报错；
- YAML 类别顺序与固定契约一致；
- 每个类别在各划分中的图片数和框数；
- `sample_id` 唯一；
- 相同 `group_id` 不跨 train/val/test；
- 来源清单中的输出哈希与现有文件一致。

发现结构、标签、哈希或泄漏错误时退出码为非零。

### 8.2 内容去重

`deduplicate_yolo_dataset.py` 使用输出图像 SHA-256 分组，默认只生成报告。

- 相同图片且标签内容规范化后相同：可在 `--apply` 下保留 test > val > train 优先级的副本并删除其余副本。
- 相同图片但标签不同：标记 `LABEL_CONFLICT` 并退出非零，不自动合并标签。
- 跨划分重复：始终在报告中单独统计。
- `--apply` 前报告必须包含将保留和将删除的精确路径。
- 执行后再次扫描并生成 postcheck 报告。

这比舌象案例的“自动合并框”更保守，避免把训练标注带入独立验证或测试集合。

## 9. 训练、预测和导出

### 9.1 模型下载

`download_models.py` 默认准备：

- `weights/pretrained/yolov8n.pt`：smoke；
- `weights/pretrained/yolov8s.pt`：正式 PoC 起点。

支持 `--verify-only`、`--force` 和 `--models`。每个文件至少检查存在、合理大小和 SHA-256；不提供来源不明的第三方镜像。

### 9.2 训练

`train.py` 读取训练配置并允许显式 CLI 覆盖。

Smoke 配置：

- 模型 `yolov8n.pt`；
- `imgsz=640`；
- `epochs=3`；
- 只用于验证依赖、数据、训练、验证和权重落盘。

PoC 配置：

- 模型 `yolov8s.pt`；
- `imgsz=1280`；
- `epochs=100`；
- `patience=20`；
- 默认执行验证并在 best.pt 上执行独立 `split=test` 评估；
- batch、device、workers 和学习率由 CLI 或配置明确给出，不根据不存在的硬件事实写死生产值。

正式训练开始前强制执行数据校验。训练输出保留 Ultralytics 原生 `args.yaml`、`results.csv`、曲线、混淆矩阵、best.pt 和 last.pt。

### 9.3 预测

`predict.py` 接受单图、目录或正式 JSONL 清单，输出：

- 带框可视化图；
- 每张图的 JSONL 检测记录；
- 类别、置信度和像素坐标；
- 按类别的图片数、框数和置信度摘要；
- 输入、模型和配置哈希。

### 9.4 模型导出

`export_model.py` 默认从 best.pt 导出 ONNX，并支持显式 `--format engine`。

- ONNX 导出记录 opset、imgsz、dynamic、simplify、权重哈希和输出哈希；
- TensorRT Engine 只在具备 CUDA/TensorRT 的目标主机运行；
- Engine 与生成它的 GPU/TensorRT/CUDA 环境信息一起记录；
- 导出失败时不生成成功元数据。

## 10. 模型包契约

发布候选模型目录至少包含：

```text
best.pt
best.onnx
model_metadata.json
```

`model_metadata.json` 必须包含：

- `model_version`；
- `task=detect`；
- 固定 7 类名称和顺序；
- `input_contract=rgb_grayscale_stack_v1`；
- `imgsz`；
- 训练数据来源清单哈希；
- PT/ONNX SHA-256；
- Ultralytics、Python 和 PyTorch 版本；
- 导出参数；
- 训练与独立测试结果文件路径；
- `intended_use=portfolio_internal_poc`；
- 明确的非生产声明。

适配器只接受元数据与模型实际类别一致、文件哈希一致且输入契约受支持的模型包。

## 11. API 推理接入

### 11.1 接口变化

`apps/api/app/inference/base.py` 增加：

- `InferenceImage`：light_id、Path、SHA-256、宽、高；
- `Detection`：class_id、defect_code、confidence、像素 xywh；
- `InferenceRequest.images`：经证据层验证并按固定光源顺序排列的图片；
- `InferenceOutput.detections`：全部检测结果。

为保持现有决策链路，`InferenceOutput` 继续提供：

- `model_version`；
- `normal_confidence`；
- `defect_score`；
- `defect_code`；
- `latency_ms`。

首版 Ultralytics 检测器设置 `normal_confidence=0.0`。`defect_score` 为保留检测中的最高置信度，`defect_code` 为最高置信度检测的类别；全部检测写入 `defect_bbox` JSON，每项同时包含类别和置信度。

### 11.2 适配器与工厂

新增：

- `app/inference/preprocessing.py`：在线 R/G/B 堆叠；
- `app/inference/ultralytics.py`：懒加载模型、模型包校验、推理和结果转换；
- `app/inference/factory.py`：按运行模式和环境创建适配器；
- `UnavailableInferenceAdapter`：非 demo 模式没有可用模型时的显式 fail-closed 适配器。

环境变量：

```text
AOI_INFERENCE_BACKEND=demo|ultralytics|tensorrt
AOI_MODEL_PATH=/models/fc-bga-v0.1/best.pt
AOI_MODEL_METADATA_PATH=/models/fc-bga-v0.1/model_metadata.json
AOI_MODEL_DEVICE=cpu|0
AOI_MODEL_IMGSZ=1280
AOI_MODEL_CONF=0.25
```

运行规则：

- `APP_MODE=demo` 默认使用 DemoInferenceAdapter；
- `APP_MODE=shadow|controlled` 不得使用 DemoInferenceAdapter；
- 非 demo 模式未配置真实后端时使用不可用适配器，导入事件进入 REVIEW；
- Ultralytics 只作为 API 的 `vision` 可选依赖，基础安装和 GPU-free demo 不加载它；
- `create_app` 支持测试注入适配器；
- PIS-IN 导入路由把 `validated_images` 传给编排器，不再丢失图片输入。

### 11.3 安全决策

- 身份或输入不完整：REVIEW；
- 模型/依赖/元数据/权重不可用：REVIEW，reason_code=`MODEL_UNAVAILABLE`；
- 最高缺陷置信度达到经策略设置的 FAIL 阈值：FAIL；
- 无框或低置信度：REVIEW；
- 首版不从检测器推导自动 PASS；
- RING 缺失仍视为输入不完整，即使模型只消费 R/G/B；
- 推理异常不伪造延迟、置信度或正常结果。

模型不可用时仍保存事件和已验证图片证据，并写入可审计的不可用结果；相同 source key 保持幂等。后续恢复模型后的重推理由独立重处理流程承担，不在本次范围内。

## 12. 依赖与运行环境

- `tools/vision/fc_bga_yolo/requirements-train.txt` 保存训练依赖；
- `apps/api/pyproject.toml` 增加 `vision` 可选依赖；
- 基础 `pip install .` 不安装 Ultralytics；
- `pip install ".[vision]"` 才启用本地 Ultralytics 推理；
- 现有 GPU-free Docker 默认行为保持不变；
- CUDA 版 PyTorch、TensorRT 和 Engine 构建按目标主机环境安装，不在通用 requirements 中强行覆盖。

## 13. 测试设计

### 13.1 训练工具测试

- R/G/B 固定顺序和灰度转换；
- RING 保留在来源清单但不进入模型像素；
- 光源缺失、重复、尺寸不一致、解码失败和哈希不一致；
- 图片/标签配对、空标签、列数、非有限值、零/负宽高和类别越界；
- group_id 跨划分泄漏；
- 去重 audit-only 不改文件；
- 图片和标签完全一致时 apply 删除副本；
- 图片一致但标签不同时拒绝 apply；
- 模型元数据、类别顺序和哈希校验；
- 所有 CLI `--help` 与配置预检不需要联网。

### 13.2 API 测试

- DemoInferenceAdapter 行为保持确定性；
- 非 demo 模式不会选择 DemoInferenceAdapter；
- validated_images 完整传入适配器；
- 在线预处理与训练转换黄金输出一致；
- 多检测框保存类别、置信度和坐标；
- 主 defect_code 选择最高置信度检测；
- 无框和低分结果进入 REVIEW；
- 模型缺失、可选依赖缺失、哈希不一致和类别不一致进入 MODEL_UNAVAILABLE/REVIEW；
- 同一 source key 的并发幂等行为不回退；
- 当前 API 测试全量回归。

涉及 Ultralytics 实际模型加载的集成测试使用显式 `vision` 测试标记；基础测试不联网、不自动下载权重。

## 14. 文档更新

更新或新增：

- `tools/vision/fc_bga_yolo/README.md`：完整操作流程；
- `data/external/fc_bga_public_smoke/README.md`：来源、许可和用途；
- `data/vision/fc_bga_defects/README.md`：正式数据契约和标注规范；
- `.env.example`：推理后端和模型路径；
- `.gitignore`：图片、标签、下载数据、权重、运行结果和 Engine；
- 总说明书与部署运维手册：从“没有真实训练/适配代码”更新为“已有训练与适配代码，但没有现场数据、正式权重和生产实绩”；
- 模型回滚手册：补充模型包元数据和哈希核验。

## 15. 验收标准

1. 训练工具包包含下载、转换、校验、去重、训练、预测和导出入口。
2. 7 类名称、顺序和标注语义在配置、元数据、训练和 API 中一致。
3. 正式转换可从四光源 JSONL 生成标准 YOLO 数据集，并保留完整来源清单。
4. 校验器能拒绝结构、标签、哈希和 group 泄漏错误。
5. 去重默认不修改文件，且不会自动合并冲突标签。
6. smoke 配置与 PoC 配置相互独立，公开数据结果明确标记为非正式。
7. ONNX 导出生成可核验模型元数据；无目标 GPU 时不声称 TensorRT 可用。
8. API demo 默认能力和现有测试不回退。
9. shadow 模式能够消费经验证的 R/G/B/RING 图片并调用真实适配器。
10. 模型缺失、依赖缺失或模型包不匹配时保留证据并进入 REVIEW，不返回假 PASS。
11. 首版任何检测器“无框”结果都不能触发自动 PASS。
12. 文档明确数据、许可、硬件和生产指标事实边界。

## 16. 风险与已确定缓解措施

| 风险 | 缓解措施 |
|---|---|
| 公开数据样本少且域不一致 | 只用于 smoke，正式报告不引用其结果作为现场指标 |
| R/G/B 是否为最佳组合未知 | 固化为 PoC 输入假设，并要求现场消融后才调整 |
| 多光源未配准导致伪影 | 尺寸或同机位前提不满足时拒绝转换，不自动补偿 |
| 缺球属于“缺失目标”，标注容易歧义 | 仅在产品阵列基准和金标准明确时标注 |
| YOLO 无框不等于正常 | normal_confidence 固定为 0，无框进入 REVIEW |
| 训练/服务预处理偏差 | 使用版本化输入契约和黄金样本字节级一致性测试 |
| 类别顺序漂移 | 模型元数据、模型 names 和 API 固定表三方校验 |
| Ultralytics 商业许可 | 限定作品集/内部 PoC，文档明确 AGPL/企业许可边界 |
| TensorRT 环境相关 | 只在目标主机导出并记录 CUDA/TensorRT/GPU 信息 |

## 17. 实施顺序

实施计划按以下依赖顺序展开：

1. 类别和数据契约；
2. 转换、校验与去重；
3. 下载、训练、预测与导出；
4. 模型包元数据；
5. API 请求/输出契约和预处理；
6. Ultralytics 适配器与工厂；
7. PIS-IN 导入链路接入和安全失败路径；
8. 回归测试与文档同步。
