from __future__ import annotations

import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Flowable,
    HRFlowable,
    PageBreak,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_TECH = ROOT / "PIS-IN_AOI_AI智能质检技术方案_V3.5_客户版.pdf"
OUT_SHOW = ROOT / "PIS-IN_AOI_AI智能质检项目展示_V3.5.pdf"
FONT = Path(r"C:\Windows\Fonts\Deng.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\Dengb.ttf")
if FONT.exists():
    pdfmetrics.registerFont(TTFont("Deng", str(FONT)))
    pdfmetrics.registerFont(TTFont("Deng-Bold", str(FONT_BOLD if FONT_BOLD.exists() else FONT)))
else:
    pdfmetrics.registerFont(TTFont("Deng", str(Path(r"C:\Windows\Fonts\simhei.ttf"))))
    pdfmetrics.registerFont(TTFont("Deng-Bold", str(Path(r"C:\Windows\Fonts\simhei.ttf"))))

PAGE_W, PAGE_H = A4
INK = colors.HexColor("#172326")
MUTED = colors.HexColor("#5F7478")
TEAL = colors.HexColor("#087F8C")
TEAL_LIGHT = colors.HexColor("#E7F4F4")
RED = colors.HexColor("#D55353")
AMBER = colors.HexColor("#C48128")
LINE = colors.HexColor("#D8E2E3")
PANEL = colors.HexColor("#F4F8F8")
WHITE = colors.white

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="BodyCN", fontName="Deng", fontSize=9.1, leading=14.2, textColor=INK, spaceAfter=5))
styles.add(ParagraphStyle(name="SmallCN", fontName="Deng", fontSize=7.8, leading=11.2, textColor=MUTED, spaceAfter=3))
styles.add(ParagraphStyle(name="H1CN", fontName="Deng-Bold", fontSize=18, leading=24, textColor=INK, spaceBefore=2, spaceAfter=8))
styles.add(ParagraphStyle(name="H2CN", fontName="Deng-Bold", fontSize=12.2, leading=17, textColor=TEAL, spaceBefore=5, spaceAfter=5))
styles.add(ParagraphStyle(name="H3CN", fontName="Deng-Bold", fontSize=10, leading=14, textColor=INK, spaceBefore=4, spaceAfter=3))
styles.add(ParagraphStyle(name="Kicker", fontName="Deng-Bold", fontSize=8.2, leading=11, textColor=TEAL, tracking=1.0, spaceAfter=5))
styles.add(ParagraphStyle(name="CoverTitle", fontName="Deng-Bold", fontSize=28, leading=36, textColor=WHITE, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="CoverSub", fontName="Deng", fontSize=12, leading=18, textColor=colors.HexColor("#CDEBED"), alignment=TA_LEFT))
styles.add(ParagraphStyle(name="Metric", fontName="Deng-Bold", fontSize=20, leading=24, textColor=TEAL, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="MetricLabel", fontName="Deng", fontSize=8, leading=11, textColor=MUTED, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="TableCN", fontName="Deng", fontSize=7.6, leading=10.5, textColor=INK))
styles.add(ParagraphStyle(name="TableHeadCN", fontName="Deng-Bold", fontSize=7.8, leading=10.5, textColor=WHITE, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="Callout", fontName="Deng-Bold", fontSize=9, leading=14, textColor=INK, backColor=TEAL_LIGHT, borderColor=TEAL, borderWidth=0.6, borderPadding=7, spaceBefore=4, spaceAfter=7))


def P(text: str, style: str = "BodyCN") -> Paragraph:
    return Paragraph(text, styles[style])


def bullets(items: list[str]) -> list[Paragraph]:
    return [P(f"<font color='{TEAL}'>•</font> {item}") for item in items]


def table(data: list[list[str]], widths: list[float], header: bool = True, compact: bool = False) -> Table:
    rows = [[P(cell, "TableHeadCN" if header and i == 0 else "TableCN") for cell in row] for i, row in enumerate(data)]
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6 if not compact else 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6 if not compact else 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5 if not compact else 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5 if not compact else 3),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
    ]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), TEAL), ("TEXTCOLOR", (0, 0), (-1, 0), WHITE)]
        for row in range(1, len(rows)):
            commands.append(("BACKGROUND", (0, row), (-1, row), WHITE if row % 2 else PANEL))
    t.setStyle(TableStyle(commands))
    return t


class FlowDiagram(Flowable):
    def __init__(self, nodes: list[tuple[str, str]], width: float = 170 * mm, height: float = 58 * mm) -> None:
        super().__init__()
        self.nodes, self.width, self.height = nodes, width, height

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        return min(self.width, avail_width), self.height

    def draw(self) -> None:
        c = self.canv
        box_w = self.width / len(self.nodes) - 5 * mm
        y = self.height / 2 - 12 * mm
        for index, (title, detail) in enumerate(self.nodes):
            x = index * (self.width / len(self.nodes)) + 2 * mm
            c.setFillColor(TEAL_LIGHT if index in (1, 3) else PANEL)
            c.setStrokeColor(TEAL if index in (1, 3) else LINE)
            c.roundRect(x, y, box_w, 24 * mm, 3 * mm, fill=1, stroke=1)
            c.setFillColor(INK)
            c.setFont("Deng-Bold", 8.5)
            c.drawCentredString(x + box_w / 2, y + 16 * mm, title)
            c.setFont("Deng", 7)
            for line_index, line in enumerate(detail.split("\n")):
                c.drawCentredString(x + box_w / 2, y + (11 - line_index * 4) * mm, line)
            if index < len(self.nodes) - 1:
                c.setStrokeColor(TEAL)
                c.setFillColor(TEAL)
                x1 = x + box_w + 1 * mm
                x2 = x + self.width / len(self.nodes) - 1 * mm
                yy = y + 12 * mm
                c.line(x1, yy, x2, yy)
                c.line(x2, yy, x2 - 2 * mm, yy + 1.3 * mm)
                c.line(x2, yy, x2 - 2 * mm, yy - 1.3 * mm)


class MetricStrip(Flowable):
    def __init__(self, metrics: list[tuple[str, str]], width: float = 170 * mm, height: float = 22 * mm) -> None:
        super().__init__()
        self.metrics, self.width, self.height = metrics, width, height

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        return min(self.width, avail_width), self.height

    def draw(self) -> None:
        c = self.canv
        col_w = self.width / len(self.metrics)
        for i, (value, label) in enumerate(self.metrics):
            x = i * col_w
            c.setFillColor(PANEL if i % 2 == 0 else TEAL_LIGHT)
            c.roundRect(x + 1, 1, col_w - 2, self.height - 2, 3, fill=1, stroke=0)
            c.setFillColor(TEAL)
            c.setFont("Deng-Bold", 15)
            c.drawCentredString(x + col_w / 2, 10 * mm, value)
            c.setFillColor(MUTED)
            c.setFont("Deng", 7)
            c.drawCentredString(x + col_w / 2, 4 * mm, label)


def header_footer(c: Canvas, doc) -> None:
    c.saveState()
    if doc.page > 1:
        c.setStrokeColor(LINE)
        c.setLineWidth(0.4)
        c.line(18 * mm, PAGE_H - 15 * mm, PAGE_W - 18 * mm, PAGE_H - 15 * mm)
        c.setFillColor(MUTED)
        c.setFont("Deng", 7)
        c.drawString(18 * mm, PAGE_H - 11 * mm, "PIS-IN AOI AI 智能质检系统  |  V3.5")
        c.drawRightString(PAGE_W - 18 * mm, 10 * mm, f"{doc.page}")
        c.setStrokeColor(LINE)
        c.line(18 * mm, 14 * mm, PAGE_W - 18 * mm, 14 * mm)
    c.restoreState()


def cover(c: Canvas, doc, title: str, subtitle: str, tag: str) -> None:
    c.saveState()
    c.setFillColor(INK)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.rect(0, PAGE_H - 56 * mm, PAGE_W, 56 * mm, fill=1, stroke=0)
    c.setFillColor(RED)
    c.rect(18 * mm, 46 * mm, 3 * mm, 74 * mm, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Deng-Bold", 8)
    c.drawString(28 * mm, PAGE_H - 28 * mm, tag)
    c.setFont("Deng-Bold", 27)
    c.drawString(28 * mm, PAGE_H - 75 * mm, title)
    c.setFillColor(colors.HexColor("#CDEBED"))
    c.setFont("Deng", 12)
    c.drawString(28 * mm, PAGE_H - 88 * mm, subtitle)
    c.setFillColor(WHITE)
    c.setFont("Deng", 9)
    c.drawString(28 * mm, 39 * mm, "项目周期  2024.09 - 2025.01")
    c.drawString(28 * mm, 31 * mm, "版本  V3.5  |  状态  可运行 PoC / 生产门禁明确")
    c.drawRightString(PAGE_W - 18 * mm, 18 * mm, "PIS-IN AOI AI")
    c.restoreState()


def build_doc(story: list[Flowable], path: Path, title: str, subtitle: str, tag: str) -> None:
    from reportlab.platypus import SimpleDocTemplate

    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=22 * mm, bottomMargin=19 * mm, title=title, author="PIS-IN AOI")
    doc.build(story, onFirstPage=lambda c, d: cover(c, d, title, subtitle, tag), onLaterPages=header_footer)


def page(title: str, kicker: str, blocks: list[Flowable]) -> list[Flowable]:
    return [P(kicker.upper(), "Kicker"), P(title, "H1CN"), HRFlowable(width="100%", thickness=0.7, color=LINE, spaceAfter=7)] + blocks + [PageBreak()]


def tech_story() -> list[Flowable]:
    s: list[Flowable] = [PageBreak()]
    s += page("一、交付结论与边界", "01 / DELIVERY POSITION", [
        P("V3.5 已形成可运行的单机 PoC：操作员或本机 AOI 软件触发采集，FastAPI + React + PostgreSQL 完成数据、推理、复核和报表闭环；Agent/RAG 作为独立辅助服务接入。自动化 Handler 暂不接入，相关接口默认关闭。", "Callout"),
        MetricStrip([("63", "API 测试"), ("6", "Agent/RAG 测试"), ("3", "前端单测"), ("1", "E2E 主流程")]),
        P("已实现", "H2CN"), *bullets(["24 条确定性演示事件、PASS/FAIL/REVIEW 三态决策、Tray Map、复核写回、工站预警、异常报告和模型治理页面。", "Source Key 采用 DeviceID + DeviceSessionID + InspectionSequence + TrayID + SlotIndex + Surface，Light ID 仅作为附件属性。", "真实 PIS-IN 导入接口、TensorRT 可替换适配器、Docker Compose 与 GPU overlay 已提供；Handler 通过 HANDLER_INTEGRATION_ENABLED=false 默认关闭。"]),
        P("必须如实披露", "H2CN"), *bullets(["当前仓库没有真实 AOI 图片、标注集和 YOLO 权重；YOLOv8/TensorRT 的真实精度与端到端吞吐需现场数据复测。", "Compose 单机服务可启动并通过健康检查；这证明软件部署链路可复现，不代表相机、TensorRT Engine 和连续生产验收已完成。", "质量指标采用分阶段目标：原 AOI NG 候选池误报基线 12%，PoC <=6%，单机受控上线 <=3%，成熟阶段 <=1.5%；全检口径单独以 <=0.5% 管理。"]),
    ])
    s += page("二、项目背景与业务目标", "02 / BUSINESS", [
        P("PIS-IN AOI 在本机产生 2D 多光源图像、3D 量测和原 AOI 结果。当前由操作员或本机软件触发，不依赖 Handler。项目增加可追溯的 AI 旁路复判层，解决多文件乱序、重复、缺失、设备重启计数归零和异常处置无法闭环的问题。"),
        FlowDiagram([("本机触发", "操作员 / AOI 软件"), ("本机采集", "R/G/B/RING\n3D / AOI log"), ("关联推理", "Source Key\nYOLOv8 / TRT"), ("决策", "PASS / FAIL / REVIEW"), ("运营闭环", "复核 / 预警\n报告 / 可选 MES")]),
        P("主要业务流程", "H2CN"), *bullets(["采集文件稳定性检查 -> 解析身份 -> 生成确定性 Source Key -> 以 event_uuid 聚合多光源附件。", "附件完整且校验通过后进入推理；未知、身份缺失、错配、低置信度和缺失关键输入统一进入 REVIEW 或 QUARANTINED。", "工站缺陷率达到滑动窗口门槛后生成告警；人工确认告警后生成 DRAFT 异常报告，质量/工艺人员完成最终处置。"]),
    ])
    s += page("三、总体技术架构与数据流", "03 / ARCHITECTURE", [
        FlowDiagram([("输入层", "PIS-IN\n模拟器"), ("数据层", "适配 / 关联\nPostgreSQL"), ("推理层", "Demo / YOLOv8\nTensorRT"), ("治理层", "3 Agents\npgvector RAG"), ("应用层", "React\n8 个页面")], height=63 * mm),
        P("架构原则", "H2CN"), *bullets(["当前采用单机容器化模块服务架构：原 AOI 继续承担基础检测，AI 在同一工控机完成旁路复判，Handler 自动握手与实时 BIN 回传默认关闭。", "业务 API、持久化、推理适配器和 Agent/RAG 均以可替换接口隔离；Agent 超时只降级为确定性草稿，不阻断检测看板。", "数据从事件、附件、模型结果、复核、告警到异常报告全程留有 event_uuid、模型版本、策略版本和证据引用；MES 仅可选异步同步。"]),
        P("数据流向", "H2CN"), P("PIS-IN 文件 / AOI 模拟器 -> PIS-IN Source Adapter -> Source Key / 附件校验 -> InspectionEvent -> DemoInferenceAdapter 或 TensorRtInferenceAdapter -> 三态决策 -> PostgreSQL -> React 看板 / 工站告警 / Agent 离线治理。"),
    ])
    s += page("四、数据关联一致性与状态机", "04 / DATA INTEGRITY", [
        P("Source Key 规范化", "H2CN"), P("规范化 JSON 按 key 排序并使用紧凑分隔符后 SHA-256，避免下划线拼接冲突。设备重启通过 DeviceSessionID 隔离重复计数；R/G/B/RING 不拆分事件，均关联到同一 event_uuid。"),
        Preformatted('{"device_id":"PIS-01","device_session_id":"BOOT-77","inspection_sequence":"1042","tray_id":"TRAY-09","slot_index":"A07","surface":"TOP"}', styles["SmallCN"]),
        table([["状态", "进入条件", "可转换 / 处理"], ["RECEIVED / COLLECTING", "首个文件到达，身份已解析", "等待附件；超时进入 EXPIRED"], ["READY / VALIDATED", "必需光源、3D/AOI 和文件校验通过", "进入推理；校验失败 REVIEW_REQUIRED"], ["INFERRED", "推理与规则决策完成", "ARCHIVED 或 REVIEW_REQUIRED"], ["QUARANTINED", "身份缺失或无法归属", "人工确认、重处理或丢弃"], ["ARCHIVED / INVALID / EXPIRED", "结果归档或终态", "不可自动回到实时链路"]], [38 * mm, 72 * mm, 60 * mm]),
        P("幂等约束", "H2CN"), P("inspection_events.source_key_hash 唯一；附件以 event_uuid + data_type + light_id + file_hash 去重；推理结果以 event_uuid + model_version + policy_version + input_fingerprint 共存，支持新旧模型影子对比。"),
    ])
    s += page("五、核心数据模型与 API", "05 / DATA MODEL", [
        table([["实体", "职责", "关键字段 / 约束"], ["inspection_events", "一次物理检测事件", "event_uuid、source_key_hash、设备/Tray/Slot、三态决策"], ["data_attachment_links", "多光源、3D、AOI 附件", "light_id、file_hash、quarantine_id、唯一去重"], ["inference_results", "模型版本化结果", "model_version、policy_version、bbox、3D 摘要、延迟"], ["quarantine_events", "身份缺失隔离", "quarantine_id、解析错误、部分字段、处理结果"], ["review_records", "人在回路金标准", "人工决策、缺陷码、备注、复核人"], ["station_alerts / anomaly_reports", "工站推送与异常闭环", "窗口缺陷率、阈值、确认人、报告证据"]], [42 * mm, 45 * mm, 83 * mm]),
        P("已验证接口", "H2CN"), *bullets(["GET /api/v1/dashboard/summary、/inspections、/trays/{tray_id}、/reviews、/alerts、/reports、/model-releases。", "POST /api/v1/inspections/import/pis-in：正常模式规范化附件并幂等写入；身份缺失返回 202 + quarantine_id。", "POST /api/v1/reviews、/alerts/{id}/acknowledge、/reports：操作持久化，页面刷新后仍可追溯。", "Agent 内部端点：/agent-api/v1/health、knowledge/search、assess-data-quality、draft-report、recommend-model-release。"]),
    ])
    s += page("六、YOLOv8 与 TensorRT 实施方案", "06 / VISION INFERENCE", [
        P("模型选型", "H2CN"), P("PoC 采用 YOLOv8s 作为速度与精度的平衡基线；按缺陷尺寸、漏检风险和边缘 GPU 资源再决定升级 YOLOv8m/l/x。模型输出不直接决定 PASS，必须经过输入完整性、3D 硬规则和置信度策略层。"),
        table([["阶段", "模型 / 运行时", "验收参数"], ["GPU-free 演示", "DemoInferenceAdapter", "确定性输出、21-28ms 模拟延迟，仅验证业务闭环"], ["PoC 联调", "YOLOv8s ONNX / TensorRT FP16", "mAP50、召回率、关键缺陷漏放率、P95、吞吐"], ["生产候选", "YOLOv8s/m + TensorRT Engine", "盲测、影子差异率、回滚演练、静默错配率=0"], ["升级条件", "YOLOv8m/l/x", "仅当小缺陷召回、遮挡场景或多尺度误检经数据证明需要"]], [34 * mm, 57 * mm, 79 * mm]),
        P("推理链路", "H2CN"), P("预处理 -> 多光源通道组织 -> YOLO 检测框/置信度 -> 3D 量测摘要 -> 规则护栏 -> PASS / FAIL / REVIEW。TensorRtInferenceAdapter 在 engine 缺失或 runtime 不可用时 fail-closed，返回不可用状态，不伪造 PASS。"),
        P("必须采集的真实参数", "H2CN"), *bullets(["按工站/产品/缺陷类别统计 precision、recall、F1、关键缺陷漏放率和人工复核率。", "记录模型版本、输入 fingerprint、P50/P95 延迟、持续吞吐、队列积压、GPU 显存和温度。", "以全检件为分母的误报率与 AOI NG 候选池误报率分开统计，不能用一个 6% 覆盖所有风险。"]),
    ])
    s += page("七、PASS / FAIL / REVIEW 决策与人在回路", "07 / DECISION", [
        FlowDiagram([("身份", "Source Key\n完整?"), ("输入", "光源 / 3D / AOI\n完整?"), ("硬规则", "3D 超限\n缺陷分数"), ("策略", "置信度\n阈值"), ("结果", "PASS / FAIL\nREVIEW")], height=58 * mm),
        table([["规则优先级", "条件", "结果"], ["1", "身份缺失、错配、关键输入缺失", "REVIEW，或 QUARANTINED"], ["2", "3D 硬限超限或缺陷分数达到门槛", "FAIL + defect_code"], ["3", "完整输入且正常置信度达到策略阈值", "PASS"], ["4", "其余低置信度、未知缺陷、Agent 不可用", "REVIEW"]], [28 * mm, 90 * mm, 52 * mm]),
        P("人工复核 vs 人在回路", "H2CN"), P("人工复核是具体操作：对 REVIEW 事件查看图像、框、3D、原 AOI 结果并提交 PASS/FAIL、缺陷码和备注；人在回路是治理边界：人拥有最终确认权，金标准回流、模型发布、异常报告结论和高风险放行均不能由 Agent 自动替代。"),
    ])
    s += page("八、3 个 Agent 与 1 套 RAG", "08 / AGENT RAG", [
        table([["Agent", "输入 / 输出", "边界"], ["数据质量 Agent", "附件、Source Key、时间偏差、重复与缺失 -> 风险等级、事实、处置建议", "不能修改事件身份，不能创建自动 PASS"], ["复核与异常报告 Agent", "告警窗口、代表事件、缺陷码、历史案例 -> DRAFT 报告与开放问题", "事实与假设分开；必须人工确认"], ["模型治理 Agent", "召回、漏放、误报、P95、吞吐、盲测、回滚、静默错配 -> 发布建议", "只能 BLOCK / SHADOW_MORE / READY_FOR_APPROVAL，不能执行发布"]], [40 * mm, 78 * mm, 52 * mm]),
        P("RAG 知识库", "H2CN"), P("PostgreSQL + pgvector 保存缺陷字典、SOP、设备手册、历史异常报告和模型发布说明。检索结果携带 document_id、chunk_id、category 和 citation；确定性 Demo embedding 保障无 LLM 时可演示，生产可替换为企业 embedding 服务。"),
        P("故障隔离", "H2CN"), P("主 API 通过超时客户端调用 Agent。Agent 超时、连接失败或引用校验失败时，报告仍落库为 DRAFT 并标记 agent_status=UNAVAILABLE；实时检测、看板和复核 API 不被拖垮。"),
    ])
    s += page("九、缺陷报表、工站推送与异常报告", "09 / QUALITY OPERATIONS", [
        P("可行性结论", "H2CN"), P("高缺陷率工站推送与异常报告是可执行的质量运营闭环，但必须采用最小样本数 + 滑动窗口 + 人工确认三道门，避免单件异常造成误报警。当前服务已实现窗口缺陷率计算、OPEN/ACKNOWLEDGED/CLOSED 状态和 DRAFT 报告持久化。"),
        table([["报表维度", "示例指标", "用途"], ["缺陷分类", "BALL_BRIDGE、SOLDER_VOID、COPLANARITY、MISSING_BALL、UNKNOWN", "定位材料、设备、工艺和模型问题"], ["工站推送", "窗口样本数、FAIL 率、阈值、连续异常窗口", "推送班组长、质量、工艺"], ["异常报告", "事实、代表事件、证据引用、开放问题、处理状态", "形成 8D / CAPA 输入"], ["模型反馈", "人工金标准、误报、漏放、影子差异", "数据闭环与版本治理"]], [34 * mm, 65 * mm, 71 * mm]),
        P("建议生产门槛", "H2CN"), *bullets(["窗口样本数 >= 20（按产线节拍调整），缺陷率 >= 8% 才进入 OPEN；连续异常窗口可提高等级。", "告警确认后才允许生成报告草稿；报告批准、关闭和恢复均需角色与时间戳。", "报告内事实必须来自事件与检索证据，推测必须写为开放问题，避免自动归因。"]),
    ])
    s += page("十、部署、算力与环境切换", "10 / DEPLOYMENT", [
        FlowDiagram([("Frontend", "Nginx\n8080"), ("API", "FastAPI\n8000"), ("Database", "PostgreSQL 16\npgvector"), ("Agent/RAG", "8013\n内网"), ("Optional GPU", "L4 / RTX 4000\nTensorRT")], height=58 * mm),
        table([["模式", "配置", "用途"], ["GPU-free demo", "PostgreSQL + API + simulator + frontend + deterministic Agent/RAG", "本地演示、培训、接口联调"], ["local-llm", "Agent/RAG overlay 请求 1 张 L4 类 GPU，OpenAI-compatible endpoint", "离线报告和知识问答质量验证"], ["tensorrt", "边缘 GPU 运行 TensorRT Engine，保留同一 API 与决策护栏", "影子运行和受控上线"], ["推荐 PoC", "训练 2 x L40S 48GB；边缘 2 台 RTX 4000 Ada；Agent/RAG 1 x L4；40-80TB NAS", "标准估算，须现场核算"]], [35 * mm, 90 * mm, 45 * mm]),
        P("启动与回退", "H2CN"), P("Compose 基础模式在一台工控机运行且不请求 GPU；GPU overlay 仅在 local-llm/tensorrt profile 请求 GPU。TensorRT engine 缺失时自动回到 REVIEW 或停用旁路，不允许静默放行。Handler 路由默认不注册；数据库迁移使用 Alembic，服务通过健康检查和 restart policy 管理。"),
    ])
    s += page("十一、安全、可靠性与审计", "11 / RELIABILITY", [
        table([["风险", "控制措施", "验证证据"], ["乱序 / 重复文件", "Source Key + hash 幂等，附件汇聚后再推理", "导入重复测试、状态机测试"], ["设备重启计数归零", "DeviceSessionID / boot id 隔离", "跨 session 同序号测试"], ["身份缺失 / 错配", "quarantine_events 持久化，人工归属", "202 状态与 quarantine_id"], ["Agent 超时", "连接/总超时、确定性草稿降级、实时链路隔离", "agent_failure_isolation 测试"], ["模型误放", "REVIEW 默认、发布门禁、回滚演练、静默错配=0", "治理 Agent BLOCK 测试"], ["数据泄露", "Agent 内网暴露，不经 Nginx 发布；日志脱敏与角色鉴权待现场接入", "部署配置审计"]], [36 * mm, 83 * mm, 51 * mm]),
        P("审计最小字段", "H2CN"), P("event_uuid、source_key_hash、输入 fingerprint、model_version、policy_version、decision、reason_code、reviewer、acknowledged_by、report_id、Agent 引用和时间戳。"),
    ])
    s += page("十二、测试与实测结果", "12 / VERIFICATION", [
        table([["层级", "结果", "覆盖"], ["FastAPI / 领域", "63 passed", "Source Key、三态决策、导入幂等、隔离、告警、Handler 默认关闭"], ["Agent/RAG", "6 passed", "检索引用、数据质量、报告草稿、发布门禁、API"], ["模拟器", "2 passed", "确定性场景与 payload"], ["React", "3 passed + typecheck + build", "导航、Tray Map、复核交互、ECharts 按需构建"], ["Playwright", "1 passed", "复核 -> Tray 追溯 -> 告警 -> DRAFT 报告"], ["Compose", "运行健康 PASS", "单机服务、端口、健康检查、无 GPU 基础模式" ]], [35 * mm, 42 * mm, 93 * mm]),
        P("当前可量化结果", "H2CN"), MetricStrip([("24", "确定性演示事件"), ("16/4/4", "PASS / FAIL / REVIEW"), ("8", "前端运营页面"), ("0", "自动发布动作")]),
        P("尚未宣称的结果", "H2CN"), P("真实 YOLO 精度、现场端到端 P95、持续吞吐、生产误报率、漏放率和静默错配率必须用连续生产样本和独立盲测完成；当前测试证明的是软件链路和安全护栏，不是现场验收结论。"),
    ])
    s += page("十三、生产上线门禁与路线图", "13 / RELEASE GATES", [
        table([["阶段", "进入条件", "退出条件"], ["设计评审", "V3.5 架构、Source Key、DDL 和状态机确认", "接口字段清单签字"], ["PoC 数据审计", "连续生产样例、身份字段、光源集合可获得", "错配 / 重复 / 延迟 / 断网注入通过"], ["YOLO/TensorRT 联调", "真实标注集、engine、GPU 环境", "盲测召回、关键缺陷漏放、P95 和吞吐达标"], ["影子运行", "新旧模型共存、报告与告警可回溯", "静默错配率=0，差异可解释，回滚演练通过"], ["生产自动 PASS", "质量、设备、工艺共同批准", "持续监控、应急停用、审计与培训齐备"]], [35 * mm, 68 * mm, 67 * mm]),
        P("后续优化", "H2CN"), *bullets(["接入真实 PIS-IN manifest / 文件监听与鉴权，补齐多工站、多产品和夜班异常统计。", "用人工复核金标准做 hard-negative mining 和阈值校准；按缺陷类别独立设定风险门槛。", "完善 Prometheus 指标、日志检索、告警通知和模型回滚操作审计。", "对 Agent/RAG 增加知识版本、引用命中率、回答拒答率和人工采纳率指标。"]),
    ])
    s += page("十四、面试与项目答辩口径", "14 / INTERVIEW EVIDENCE", [
        P("一句话介绍", "H2CN"), P("我负责把 PIS-IN AOI 的多源数据关联、AI 旁路复判和质量运营闭环做成可在一台工控机运行的 V3.5 PoC：本机触发采集，用 Source Key 解决事件一致性，用 YOLOv8/TensorRT 承载推理，用三态决策与人在回路控制风险，再用 3 个 Agent + pgvector RAG 生成可引用的治理建议。", "Callout"),
        table([["追问", "回答重点"], ["为什么不是 Agent 直接判定？", "实时链路必须确定性、低延迟、可回退；Agent 只做解释、检索和报告草稿，最终放行由规则和人确认。"], ["人工复核和人在回路区别？", "前者是 REVIEW 队列中的具体动作；后者是整个系统的责任边界和审批权。"], ["6% 是否代表生产效果？", "不是。6% 是 PoC 目标，生产需用现场分母、盲测和漏放/静默错配共同验收。"], ["为什么先用 YOLOv8s？", "以边缘资源和实时性为约束的基线；是否升级 m/l/x 由小缺陷召回和 P95 数据决定。"], ["如何证明可落地？", "真实 API、持久化、故障隔离、导入幂等、E2E 主流程和 Compose 部署合同均可复现；现场门禁仍保留。"]], [43 * mm, 127 * mm]),
    ])
    s += page("十五、附录：项目固定口径", "15 / APPENDIX", [
        table([["项目", "固定口径"], ["周期", "2024.09 - 2025.01"], ["团队", "8 人按阶段参与：产品/AI 负责人 1、后端 2、前端 1、算法 2、测试 1、实施运维 1"], ["运行", "AOI 工控机单机独立运行；Handler 默认关闭；MES 仅可选异步同步"], ["Agent", "数据质量 Agent、复核与异常报告 Agent、模型治理 Agent，共 3 个；不进入实时自动 PASS"], ["RAG", "PostgreSQL + pgvector；缺陷字典、SOP、设备手册、历史异常报告、模型发布说明"], ["质量目标", "AOI NG 候选池误报：基线 12% -> PoC <=6% -> 单机受控 <=3% -> 成熟 <=1.5%；全检口径 <=0.5% 单列"], ["版本状态", "V3.5：可运行单机 PoC / 数据适配与部署基线；自动 PASS 待现场门禁"]], [42 * mm, 128 * mm]),
        Spacer(1, 12 * mm), P("文档结束", "Kicker"),
    ])
    return s


def show_story() -> list[Flowable]:
    s: list[Flowable] = [PageBreak()]
    s += page("1. 项目一句话", "PROJECT STORY", [
        P("把 AOI 的多源检测数据变成可追溯、可复核、可治理的 AI 质量闭环。", "Callout"),
        MetricStrip([("8", "运营页面"), ("3", "治理 Agent"), ("24", "演示事件"), ("V3.5", "可运行基线")]),
        P("项目背景", "H2CN"), P("PIS-IN AOI 同时产出多光源 2D、3D 量测和原 AOI 结果。当前采用操作员或本机 AOI 软件触发，在一台工控机完成采集、复判、复核和报表，不接入自动化 Handler。项目解决数据关联不稳定、缺陷统计难追溯、工站异常无法闭环和模型发布缺乏证据的问题。"),
        P("交付判断", "H2CN"), P("当前交付的是可运行 PoC 和生产实施基线，而不是未经现场验收的自动放行系统。该边界让面试时可以清楚回答“做成了什么”和“还需要哪些证据”。"),
    ])
    s += page("2. 核心业务流程", "BUSINESS FLOW", [
        FlowDiagram([("本机触发", "操作员 / AOI"), ("采集关联", "四光源\nSource Key"), ("推理", "YOLOv8 / 3D"), ("决策", "三态护栏"), ("闭环", "复核 / 报告")], height=59 * mm),
        P("主要业务模块", "H2CN"), *bullets(["生产总览：PASS/FAIL/REVIEW、趋势和工站门禁。", "实时检测与 Tray Map：定位槽位、图像证据、缺陷框、3D 摘要和模型版本。", "人工复核：只处理 REVIEW，结论写入金标准。", "缺陷报表、预警与异常报告：按产品、批次、工站和缺陷分类分析。", "模型治理、项目说明：影子版本、门禁证据、部署口径与个人交付成果。"]),
    ])
    s += page("3. 技术亮点", "TECHNICAL HIGHLIGHTS", [
        table([["亮点", "解决的问题", "可验证证据"], ["Source Key + quarantine", "乱序、重复、重启归零、身份缺失", "导入幂等、202 隔离、状态机测试"], ["三态决策", "高置信度误放和异常吞噬", "输入优先 REVIEW，3D/缺陷规则 FAIL"], ["YOLOv8s -> TensorRT", "边缘推理速度与模型可替换", "Demo/TensorRT 适配器、fail-closed"], ["3 Agent + RAG", "从检测结果到质量运营", "引用 evidence、DRAFT、发布 BLOCK"], ["故障隔离", "Agent/模型故障拖垮实时链路", "超时降级测试、实时 API 保持 200"]], [38 * mm, 73 * mm, 59 * mm]),
        P("最有价值的设计选择", "H2CN"), P("把确定性实时链路与非确定性治理链路分开：视觉模型负责证据，规则负责安全边界，Agent 负责汇总和解释，人负责最终确认。"),
    ])
    s += page("4. 运行展示与交互", "RUNNABLE DEMO", [
        P("演示主流程", "H2CN"), P("打开总览 -> 进入人工复核并提交缺陷 -> Tray Map 点击槽位查看证据 -> 确认工站预警 -> 生成 DRAFT 异常报告。上述流程通过 Playwright 在隔离端口真实执行通过。"),
        table([["页面", "展示内容", "用户动作"], ["生产总览", "事件计数、趋势、工站门禁", "查看当前风险"], ["实时检测", "2D 框、置信度、3D、原 AOI 结果", "点击事件追溯"], ["Tray Map", "槽位热力和批次定位", "点击槽位打开证据"], ["人工复核", "REVIEW 队列和证据", "确认正常 / 缺陷"], ["预警与报告", "工站阈值、确认和 DRAFT", "确认预警、生成报告"], ["模型治理", "生产/影子版本和指标", "查看门禁，不直接发布"]], [35 * mm, 88 * mm, 47 * mm]),
    ])
    s += page("5. 我负责的工作与个人优势", "OWNERSHIP", [
        table([["我负责", "交付结果", "转化优势"], ["需求拆解与门禁", "把误报、漏放、P95、错配率转成阶段验收指标", "能把模糊 AI 需求变成可交付产品"], ["数据关联设计", "Source Key、状态机、quarantine、版本共存", "工业数据建模与一致性治理"], ["推理与决策", "YOLOv8/TensorRT 接口、3D 融合、三态护栏", "算法工程化和生产风险意识"], ["Agent/RAG 方案", "3 Agent、证据引用、故障降级", "AI 应用架构与人在回路设计"], ["全链路交付", "API、React、测试、Compose、部署文档", "跨前后端和交付团队推进"]], [40 * mm, 80 * mm, 50 * mm]),
        P("面试表达", "H2CN"), P("我不把“模型准确率”单独当成项目成功，而是用数据关联正确、关键缺陷不漏放、系统可回退、复核可追溯和质量闭环能落地来定义成功。"),
    ])
    s += page("6. 团队、周期与算力", "DELIVERY CONTEXT", [
        table([["阶段", "参与角色", "重点"], ["需求与数据审计", "产品/AI、算法、后端、实施", "字段、缺陷字典、门禁和样本"], ["PoC 开发", "产品/AI、算法 2、后端 2、前端", "推理适配、业务 API、运营页面"], ["联调与影子", "全团队，测试/实施加大投入", "故障注入、盲测、部署与培训"], ["项目周期", "2024.09 - 2025.01", "8 人按阶段参与"]], [43 * mm, 58 * mm, 69 * mm]),
        P("算力口径", "H2CN"), P("标准 PoC 估算：训练 2 x L40S 48GB；边缘 2 台 RTX 4000 Ada 20GB；Agent/RAG 1 x L4 24GB；40-80TB NAS。无 GPU 时的基础演示只使用确定性推理，避免把硬件作为 Demo 前置条件。"),
    ])
    s += page("7. 结果、风险与下一步", "OUTCOME", [
        MetricStrip([("63", "后端测试"), ("6", "Agent 测试"), ("1", "E2E 通过"), ("0", "自动 PASS 越权")]),
        P("价值", "H2CN"), *bullets(["减少人工从文件夹找图、对 Tray 和工站的时间，把异常处置从“看结果”变成“看证据”。", "将缺陷分类、工站推送、异常报告和模型治理连接起来，为质量团队形成持续改进入口。", "采用单机运行并保留人工确认边界，减少跨设备联调范围；Handler 接口作为未来扩展而非当前依赖。"]),
        P("生产前仍需完成", "H2CN"), *bullets(["真实 PIS-IN 文件字段和光源集合确认；真实图片、标注集和 TensorRT engine 接入。", "连续样本盲测、端到端 P95/吞吐、关键缺陷漏放率、静默错配率=0。", "Prometheus/日志/通知/权限接入，并在目标 AOI 工控机完成真实部署和断电恢复验收。"]),
    ])
    s += page("8. 面试收口", "CLOSING", [
        P("推荐回答", "H2CN"), P("这个项目的难点不只是训练 YOLO，而是把工业现场的多文件关联、视觉证据、3D 规则、人工复核、工站告警和模型治理串成可审计闭环。我负责把这些边界固化成 API、数据模型、决策策略和可运行演示，并明确哪些指标已经测试、哪些必须等现场验收。"),
        P("一句话价值", "H2CN"), P("让 AI 检测结果可以被追溯、被复核、被解释、被运营，而不是只输出一个 PASS 或 FAIL。", "Callout"),
    ])
    return s


if __name__ == "__main__":
    OUT_TECH.parent.mkdir(parents=True, exist_ok=True)
    build_doc(tech_story(), OUT_TECH, "PIS-IN AOI AI 智能质检技术方案", "V3.5 单机实施与生产门禁版", "STANDALONE BASELINE / 2026.08")
    build_doc(show_story(), OUT_SHOW, "PIS-IN AOI AI 智能质检项目展示", "V3.5 单机可运行 PoC 与面试证据版", "STANDALONE PROJECT STORY / 2026.08")
    print(OUT_TECH)
    print(OUT_SHOW)
