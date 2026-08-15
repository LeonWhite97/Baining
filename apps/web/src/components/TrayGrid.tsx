import type {InspectionEvent} from "../api/types"
import {StatusBadge} from "./StatusBadge"


export function TrayGrid({slots, onSelect}: {slots: InspectionEvent[]; onSelect: (event: InspectionEvent) => void}) {
  return <div className="tray-grid" aria-label="Tray 槽位图">{slots.map((slot) => (
    <button key={slot.event_uuid} className={`tray-cell tray-${slot.decision.toLowerCase()}`} onClick={() => onSelect(slot)} aria-label={`槽位 ${slot.slot_index} · ${slot.decision}`}>
      <span className="slot-number">{slot.slot_index}</span><StatusBadge value={slot.decision}/><small>{slot.defect_code ?? "正常"}</small>
    </button>
  ))}</div>
}
