# PCB 实拍图稳定性测试包

本目录由 `tools/data/fetch_pcb_stability_samples.ps1` 生成，素材来自 Wikimedia Commons。

## 适用范围

- `source/`：保留原始比例的公开实拍图，用于图片解码、比例变化、前后面和复杂度差异测试。
- `normalized_1920x1080/`：按接口文档示例尺寸生成的 JPEG。采用等比缩放和黑边填充，不拉伸内容。
- `manifest.csv` / `manifest.json`：来源页、作者、许可、尺寸和 SHA-256 校验值。

由于 Wikimedia Commons 在获取素材时对媒体 CDN 限流，`pcb_01_canonscan` 为 960 px 视觉基线图，其余 9 张是网页已加载的 250–330 px 缩略图。清单中的 `quality_tier` 会将后者标记为 `low_resolution_pipeline_stress`。这些低分辨率图适合验证小图输入、放大、内存、异常尺寸和并发处理，不适合评估小缺陷识别精度。`normalized_1920x1080` 只改变画布和编码，不会创造图像细节。

项目文档表明当前目标设备是 PIS-IN 半导体封装 AOI，默认输入为同一检测事件下的 R/G/B/RING 多光源图像，并结合 3D 量测。这里的 PCB 网络照片没有同机位、多光源、3D 数据或缺陷真值，因此只适合验证：

1. 图片下载、落盘、解码和哈希校验；
2. 不同比例、板型、纹理密度和前后面的展示与预处理；
3. 批量回放、并发、内存占用和长时间运行稳定性；
4. 非同域输入是否被正确隔离或转为 `REVIEW`，而不是自动 `PASS`。

这些图片不能用于声明 PIS-IN 真实缺陷识别准确率、误报率、漏放率或现场节拍达标。精度验收仍需要设备采集的同一物理样本 R/G/B/RING 图像、3D 数据、原 AOI 结果和人工标注。

## 重新生成

在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\data\fetch_pcb_stability_samples.ps1
```

每张图的具体许可和署名要求以 `manifest.csv` 中的 `source_page`、`author`、`license` 和 `license_url` 为准。派生的 1920x1080 版本沿用原图许可。

## 端到端 API 模拟

API 通过 `AOI_IMAGE_ROOT` 只读取受控目录内文件，并在入库前验证 R/G/B/RING 集合、文件路径、SHA-256、JPEG/PNG 解码和尺寸。Compose 默认仍运行连续场景模拟器；一次性图片链路测试使用 `e2e` profile：

```powershell
docker compose -f .\infra\docker-compose.yml --profile e2e run --rm simulator-e2e
```

报告写入 `tmp/pcb-e2e-report.json`。默认对 10 张素材各入库一次，并覆盖幂等重放、光源乱序、附件冲突、缺光源、错哈希、损坏图片、路径越界、人工复核、20 件告警窗口和草稿报告。未设置 `SIM_RUN_ID` 时会自动生成运行命名空间，因此可在持久数据库上重复执行；可用 `SIM_LOOPS` 增加素材循环次数，用 `SIM_CONCURRENCY` 控制素材导入并发数。

本地直接运行 API 时，可在项目根目录设置：

```powershell
$env:AOI_IMAGE_ROOT = (Resolve-Path '.\data\external\pcb_stability_samples\normalized_1920x1080').Path
python -m uvicorn app.main:app --app-dir .\apps\api --port 8000
```

模拟器的 `Scenario` 是合成判定标签，不是对 PCB 图片执行真实模型推理。报告中的 `synthetic_decision=true` 用于明确这一边界。
