import type {InspectionEvent} from "../api/types"
import {StatusBadge} from "./StatusBadge"


export function EventEvidence({event}: {event: InspectionEvent}) {
  return <section className="evidence-layout">
    <div className="inspection-image"><img src={event.image_url} alt={`${event.event_uuid} AOI 检测图`}/>{event.bbox.map((box, index) => <span key={index} className="bbox" style={{left: `${box.x}%`, top: `${box.y}%`, width: `${box.w}%`, height: `${box.h}%`}}/>)}</div>
    <dl className="evidence-data">
      <div><dt>最终判定</dt><dd><StatusBadge value={event.decision}/></dd></div>
      <div><dt>置信度</dt><dd>{(event.confidence * 100).toFixed(1)}%</dd></div>
      <div><dt>缺陷分类</dt><dd>{event.defect_code ?? "无"}</dd></div>
      <div><dt>理由码</dt><dd>{event.reason_code}</dd></div>
      <div><dt>模型版本</dt><dd>{event.model_version ?? "演示推理"}</dd></div>
      <div><dt>Tray / Slot</dt><dd>{event.tray_id} / {event.slot_index}</dd></div>
    </dl>
  </section>
}
