import {useEffect, useState} from "react"
import {RefreshCw} from "lucide-react"

import {api} from "../api/client"
import type {InspectionEvent} from "../api/types"
import {AsyncState} from "../components/AsyncState"
import {EventEvidence} from "../components/EventEvidence"
import {PageHeader} from "../components/PageHeader"
import {StatusBadge} from "../components/StatusBadge"
import {useAsync} from "../hooks/useAsync"


export function InspectionPage() {
  const state = useAsync(api.inspections, [])
  const [selected, setSelected] = useState<InspectionEvent | null>(null)
  useEffect(() => { if (!selected && state.data?.items[0]) setSelected(state.data.items[0]) }, [selected, state.data])
  return <div><PageHeader title="实时检测" subtitle="模型框、3D 量测与规则理由在同一证据视图中呈现" actions={<button className="icon-button" title="刷新" onClick={() => void state.reload()}><RefreshCw size={18}/></button>}/>
    <AsyncState loading={state.loading} error={state.error} onRetry={state.reload}/>{selected && <EventEvidence event={selected}/>}<div className="panel data-table-panel"><table><thead><tr><th>时间</th><th>工站</th><th>Tray / Slot</th><th>判定</th><th>缺陷</th><th>置信度</th></tr></thead><tbody>{state.data?.items.slice(0, 12).map((item) => <tr key={item.event_uuid} onClick={() => setSelected(item)}><td>{new Date(item.created_at).toLocaleTimeString()}</td><td>{item.station}</td><td>{item.tray_id} / {item.slot_index}</td><td><StatusBadge value={item.decision}/></td><td>{item.defect_code ?? "-"}</td><td>{(item.confidence * 100).toFixed(1)}%</td></tr>)}</tbody></table></div>
  </div>
}
