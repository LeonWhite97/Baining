from __future__ import annotations

"""Regenerate the public-external review artifacts from the canonical manifest.

Outputs (both intentionally git-ignored as *derived* artifacts — rebuild with this script):
  - data/external/fc_bga_public_external/review/contact_sheet.html
  - data/external/fc_bga_public_external/review/candidates.enriched.json

Inputs (tracked):
  - data/external/fc_bga_public_external/review/candidates.jsonl  (source of truth)
  - data/external/fc_bga_public_external/review/images/*.jpg      (tracked thumbnails)
  - data/external/fc_bga_public_external/sources.json             (source metadata)

Run from the project root (D:/YOLO/Baining):
  python tools/vision/fc_bga_yolo/build_review_artifacts.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from contracts import DEFECT_NAMES
except ImportError:  # pragma: no cover - run as script from project root
    DEFECT_NAMES = (
        "BALL_BRIDGE",
        "MISSING_BALL",
        "EXTRA_BALL",
        "BALL_SIZE_ABNORMAL",
        "BALL_OFFSET",
        "BALL_SHAPE_ABNORMAL",
        "FOREIGN_MATERIAL",
    )

from PIL import Image

_REVIEW_DIR = Path("data/external/fc_bga_public_external/review")
_SOURCES = Path("data/external/fc_bga_public_external/sources.json")


def load_source_names(sources_path: Path) -> dict[str, str]:
    if not sources_path.exists():
        return {}
    data = json.loads(sources_path.read_text(encoding="utf-8"))
    records = data.get("sources", data) if isinstance(data, dict) else data
    out: dict[str, str] = {}
    for rec in records:
        sid = rec.get("source_id")
        if sid:
            out[sid] = rec.get("name", sid)
    return out


def image_dimensions(image_path: Path) -> tuple[int, int]:
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        return (0, 0)


def load_candidates(manifest_path: Path) -> list[dict]:
    out: list[dict] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_enriched(candidates: list[dict], review_dir: Path) -> list[dict]:
    enriched: list[dict] = []
    for c in candidates:
        rec = dict(c)
        img_file = Path(c.get("image_path", "")).name
        img_path = review_dir / "images" / img_file
        w, h = image_dimensions(img_path) if img_path.exists() else (0, 0)
        rec["_w"] = w
        rec["_h"] = h
        rec["_img_file"] = img_file
        enriched.append(rec)
    return enriched


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_html(enriched: list[dict], source_names: dict[str, str], review_dir: Path) -> str:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for rec in enriched:
        by_source[rec.get("source_id", "unknown")].append(rec)

    legend_items = "".join(f"<li><code>{_esc(name)}</code></li>" for name in DEFECT_NAMES)
    total = len(enriched)
    review_required = sum(1 for r in enriched if r.get("review_status") == "review_required")

    sections = []
    for source_id, recs in sorted(by_source.items(), key=lambda kv: kv[0]):
        label = source_names.get(source_id, source_id)
        tiles = []
        for r in sorted(recs, key=lambda x: x.get("_img_file", "")):
            name = r.get("_img_file", "").removesuffix(".jpg")
            dim = f'{r.get("_w", 0)}x{r.get("_h", 0)}'
            tiles.append(
                '<div class="tile" id="{nid}">'
                '<img loading="lazy" src="images/{n}.jpg" alt="{n}">'
                '<div class="meta"><span class="sid">{n}</span><span class="dim">{dim}</span></div>'
                '<div class="src">{src}</div></div>'.format(
                    nid=_esc(name), n=_esc(name), dim=_esc(dim), src=_esc(label)
                )
            )
        sections.append(
            f'<section><h2>{_esc(label)} — {len(recs)} 张</h2>'
            f'<div class="grid">{"".join(tiles)}</div></section>'
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FC-BGA 公开外部候选审查联系表</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ background:#0f1115; color:#e6e6e6; font-family: system-ui, sans-serif; margin:0; padding:20px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .sub {{ color:#9aa0a6; font-size:13px; margin-bottom:16px; }}
  h2 {{ font-size:15px; border-bottom:1px solid #2a2e35; padding-bottom:6px; margin:24px 0 12px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:10px; }}
  .tile {{ background:#1a1d23; border:1px solid #2a2e35; border-radius:8px; overflow:hidden; }}
  .tile img {{ width:100%; height:150px; object-fit:contain; background:#000; display:block; }}
  .meta {{ display:flex; justify-content:space-between; font-size:11px; padding:4px 6px 0; }}
  .sid {{ color:#7aa2f7; font-family:monospace; }}
  .dim {{ color:#9aa0a6; }}
  .src {{ font-size:10px; color:#6b7280; padding:2px 6px 6px; }}
  .legend {{ background:#1a1d23; border:1px solid #2a2e35; border-radius:8px; padding:12px 16px; margin-bottom:16px; font-size:13px; }}
  .legend code {{ color:#7aa2f7; }}
  .legend ul {{ margin:6px 0 0; padding-left:18px; }}
</style>
</head>
<body>
  <h1>FC-BGA 公开外部候选审查联系表</h1>
  <div class="sub">共 {total} 张 · {review_required} 张 <b>review_required</b> · 目标：接受 ≥20 张且覆盖 ≥2 个七类，解锁 B0 门控</div>
  <div class="legend">
    <b>七类缺陷（标注时从可见边界判定，不可猜测）：</b>
    <ul>{legend_items}</ul>
    <p style="margin:8px 0 0;color:#9aa0a6">审查方式：在原生分辨率下确认可见的球栅缺陷类别与边界 → 在 <code>candidates.jsonl</code> 将该样本 <code>review_status</code> 改为 <code>accepted</code> 并填 <code>accepted_classes</code>。模糊/无法判定的保持 <code>review_required</code> 或标 <code>quarantined</code>。详见 <code>ANNOTATION_SPEC.md</code>。</p>
  </div>
  {''.join(sections)}
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild public-external review artifacts.")
    parser.add_argument("--review-dir", type=Path, default=_REVIEW_DIR)
    parser.add_argument("--sources", type=Path, default=_SOURCES)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    review_dir = args.review_dir
    manifest = args.manifest or (review_dir / "candidates.jsonl")

    candidates = load_candidates(manifest)
    source_names = load_source_names(args.sources)
    enriched = build_enriched(candidates, review_dir)

    html = render_html(enriched, source_names, review_dir)
    (review_dir / "contact_sheet.html").write_text(html, encoding="utf-8")

    with (review_dir / "candidates.enriched.json").open("w", encoding="utf-8") as fh:
        json.dump(enriched, fh, ensure_ascii=False, indent=2)

    print(f"wrote contact_sheet.html ({len(html)} bytes), candidates.enriched.json ({len(enriched)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
