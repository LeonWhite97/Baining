# -*- coding: utf-8 -*-
"""重建《建设方案 V3.5》第 8 页（架构，含 B 修正）与第 9 页（性能，含 C、D 修正）。
保留原文档的设计令牌与页眉/大页码样式，输出两页独立 PDF，供 PyMuPDF 替换回原文件。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / "tmp"
TMP.mkdir(parents=True, exist_ok=True)

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ---- 字体（与原文档一致）----
FONT = Path(r"C:\Windows\Fonts\Deng.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\Dengb.ttf")
if FONT.exists():
    pdfmetrics.registerFont(TTFont("Deng", str(FONT)))
    pdfmetrics.registerFont(TTFont("Deng-Bold", str(FONT_BOLD if FONT_BOLD.exists() else FONT)))
else:
    pdfmetrics.registerFont(TTFont("Deng", str(Path(r"C:\Windows\Fonts\simhei.ttf"))))
    pdfmetrics.registerFont(TTFont("Deng-Bold", str(Path(r"C:\Windows\Fonts\simhei.ttf"))))

# ---- 设计令牌（与原文档共用）----
PAGE_W, PAGE_H = A4
INK = colors.HexColor("#172326")
MUTED = colors.HexColor("#5F7478")
TEAL = colors.HexColor("#087F8C")
TEAL_LIGHT = colors.HexColor("#E7F4F4")
LINE = colors.HexColor("#D8E2E3")
PANEL = colors.HexColor("#F4F8F8")
WHITE = colors.white
BIGNUM = colors.HexColor("#DCE7E8")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="BodyCN", fontName="Deng", fontSize=9.1, leading=14.2, textColor=INK, spaceAfter=5))
styles.add(ParagraphStyle(name="SmallCN", fontName="Deng", fontSize=7.8, leading=11.2, textColor=MUTED, spaceAfter=3))
styles.add(ParagraphStyle(name="H1CN", fontName="Deng-Bold", fontSize=18, leading=24, textColor=INK, spaceBefore=2, spaceAfter=8))
styles.add(ParagraphStyle(name="H2CN", fontName="Deng-Bold", fontSize=12.2, leading=17, textColor=TEAL, spaceBefore=5, spaceAfter=5))
styles.add(ParagraphStyle(name="Kicker", fontName="Deng-Bold", fontSize=8.2, leading=11, textColor=TEAL, spaceAfter=5))
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


def cp_header(c, doc, page_no: int) -> None:
    """原建设方案页眉：标题行 + 大页码 + 上下分隔线 + 页脚页码。"""
    c.saveState()
    # 上分隔线
    c.setStrokeColor(LINE)
    c.setLineWidth(0.4)
    c.line(18 * mm, PAGE_H - 15 * mm, PAGE_W - 18 * mm, PAGE_H - 15 * mm)
    # 标题行
    c.setFillColor(MUTED)
    c.setFont("Deng", 7)
    c.drawString(18 * mm, PAGE_H - 11 * mm, "PIS-IN AOI AI 智能质检项目建设方案  |  V3.5")
    # 大页码（右侧，浅色）
    c.setFillColor(BIGNUM)
    c.setFont("Deng-Bold", 46)
    c.drawRightString(PAGE_W - 18 * mm, PAGE_H - 30 * mm, f"{page_no:02d}")
    # 下分隔线
    c.line(18 * mm, 14 * mm, PAGE_W - 18 * mm, 14 * mm)
    c.restoreState()


def cp_page(kicker: str, title: str, blocks: list) -> list:
    return [P(kicker.upper(), "Kicker"), P(title, "H1CN"), HRFlowable(width="100%", thickness=0.7, color=LINE, spaceAfter=7)] + blocks


def build_single(path: Path, page_no: int, kicker: str, title: str, blocks: list) -> None:
    from reportlab.platypus import SimpleDocTemplate

    def _header(c, doc):
        cp_header(c, doc, page_no)

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=22 * mm,
        bottomMargin=19 * mm,
        title="PIS-IN AOI AI 智能质检项目建设方案",
        author="PIS-IN AOI",
    )
    doc.build(cp_page(kicker, title, blocks), onFirstPage=_header, onLaterPages=_header)


# ===================== 第 8 页：技术架构（含 B 修正） =====================
def page8_blocks() -> list:
    arch_table = table(
        [
            ["层级", "核心组件", "职责与设计选择"],
            ["展示层", "React 18 / TypeScript / Vite / ECharts / Nginx", "8 个质量运营页面；Nginx 代理 API 与 SPA fallback"],
            ["核心应用", "FastAPI / Pydantic / SQLAlchemy 2 / Alembic", "模块化单体承载确定性实时链路，降低单机排障复杂度"],
            ["推理层", "YOLOv8s / ONNX / TensorRT / 3D 规则", "可替换 adapter；异常 fail-closed 到 REVIEW"],
            # B 修正：PostgreSQL 16 -> PostgreSQL 14+（目标 16）
            ["数据层", "PostgreSQL 14+（目标 16）/ pgvector；测试用 SQLite", "事务幂等、版本共存、审计与向量检索"],
            ["治理层", "数据质量/报告/模型治理 Agent", "独立进程，超时不影响检测主链路"],
            ["部署层", "Docker Compose / GPU overlay / healthcheck", "单工控机容器化、服务可独立重启"],
        ],
        [26 * mm, 54 * mm, 94 * mm],
    )
    return [
        arch_table,
        Spacer(1, 4 * mm),
        P("架构判断", "H2CN"),
        P("当前是单机容器化模块服务架构，不是全微服务。确定性检测主链路保持紧凑；非实时 Agent/RAG 独立隔离，以获得低延迟、易回滚和可维护性的平衡。"),
    ]


# ===================== 第 9 页：性能基准（含 C、D 修正） =====================
def page9_blocks() -> list:
    perf_table = table(
        [
            ["指标", "当前证据", "阶段目标/验收方式"],
            ["软件正确性", "API 63、Agent/RAG 6、Simulator 2 条历史通过记录", "发布前重新执行全套单测、前端 build/typecheck、E2E"],
            # C 修正：在演示数据行显式引用 ≥1000 事件 + 故障注入 验收门槛
            ["演示数据", "24 条确定性事件；PASS/FAIL/REVIEW = 16/4/4", "仅证明流程可复现，不代表生产分布；验收门槛为 ≥1000 事件 + 故障注入（详见数据一致性专章）"],
            ["推理延迟 P95", "仓库无真实权重与目标工控机实测", "在目标 GPU 上按模型/产品/光源组合分桶压测"],
            ["持续吞吐", "尚无连续生产样本", "记录稳态 FPS、队列深度、GPU/CPU/内存与磁盘水位"],
            ["质量效果", "误报率为阶段目标，非实绩", "独立盲测同时报告关键缺陷漏放、REVIEW 与置信区间"],
            ["静默错配", "设计目标 0", "乱序/重复/重启/延迟/断网故障注入验证"],
        ],
        [30 * mm, 72 * mm, 72 * mm],
    )
    # D 修正：算力基线拆成 训练侧 / 边缘推理侧 / 治理侧 / 存储 四栏，并注明“单机”= 单体设备；
    # 进一步按“需求推导 + 待实测”重写：补选型依据、存储推导、与阶段目标的闭环缺口、功耗/冗余说明
    compute_table = table(
        [
            ["侧别", "配置", "选型依据（由需求推导）", "状态"],
            ["训练侧", "2 × L40S 48GB", "中心侧机房资源；按训练吞吐与 TensorRT 引擎编译并行度估算，非本机部署", "待实测"],
            ["边缘推理侧（单体设备 · PoC 节点）", "2 × RTX 4000 Ada 20GB", "本机旁路复判；按“相机路数 × 模型 FPS”估算，单卡已够，2 卡为并发/冗余预留；仓库暂无真实权重实测", "待实测"],
            ["治理侧", "1 × L4 24GB", "离线检索/报告/门禁；若 LLM/RAG 走外部 API，本地可不配", "待实测"],
            ["存储", "容量 40–80TB NAS", "容量 = 事件率 × 单事件图像体积 × 保留周期；NAS 为顺序写，需关注带宽/IOPS 而非仅容量", "待实测"],
        ],
        [30 * mm, 38 * mm, 78 * mm, 28 * mm],
    )
    return [
        perf_table,
        Spacer(1, 4 * mm),
        P("建议算力基线（由需求推导，待现场实测定稿）", "H2CN"),
        compute_table,
        Spacer(1, 3 * mm),
        P("<b>选型说明（需求推导口径）：</b>“单机”指<b>单体设备</b>——即部署在检测工位的一台独立工控机/边缘设备，仅承载本表“边缘推理侧（2 × RTX 4000 Ada）”的本机旁路复判；训练侧（2 × L40S）与治理侧（1 × L4）为<b>独立于该单体设备的中心侧资源</b>，不与其混装，请勿理解为“全部模块运行于同一台机器”。", "SmallCN"),
        P("<b>存储容量推导（示例，非承诺）：</b>容量 ≈ 事件率 × 单事件图像体积 × 保留周期。例：4 路相机 × 约 6MB/图 × 约 3 图/事件 × 600 事件/班次 × 3 班次/日 × 90 天留存 ≈ 数十 TB，故给出 40–80TB 区间；实际以现场事件率与留存策略为准。", "SmallCN"),
        P("<b>待实测与缺口：</b>算力为凭经验初列、尚未经目标工控机实测，且与第 9 页阶段目标（P95 时延、稳态 FPS、队列深度、GPU 水位）尚未闭环映射；PoC 为非 HA 部署，生产需补充供电/散热冗余与机架规划。GPU-free 模式可完成确定性演示与接口联调。", "SmallCN"),
    ]


if __name__ == "__main__":
    p8 = TMP / "cp_page8.pdf"
    p9 = TMP / "cp_page9.pdf"
    build_single(p8, 8, "06 / ARCHITECTURE", "技术架构", page8_blocks())
    build_single(p9, 9, "07 / PERFORMANCE", "核心组件与性能基准", page9_blocks())
    print("WROTE", p8, p9)
