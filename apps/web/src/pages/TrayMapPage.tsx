import {useState} from "react"

import {api} from "../api/client"
import type {InspectionEvent} from "../api/types"
import {AsyncState} from "../components/AsyncState"
import {EventEvidence} from "../components/EventEvidence"
import {PageHeader} from "../components/PageHeader"
import {TrayGrid} from "../components/TrayGrid"
import {useAsync} from "../hooks/useAsync"


export function TrayMapPage() {
  const [trayId, setTrayId] = useState("TRAY-001")
  const state = useAsync(() => api.tray(trayId), [trayId])
  const [selected, setSelected] = useState<InspectionEvent | null>(null)
  return <div><PageHeader title="Tray Map" subtitle="按槽位追溯一次物理检测及其全部光源、模型与复核证据" actions={<select value={trayId} onChange={(event) => {setTrayId(event.target.value); setSelected(null)}}><option>TRAY-001</option><option>TRAY-002</option></select>}/><AsyncState loading={state.loading} error={state.error} onRetry={state.reload}/>{state.data && <><div className="panel"><TrayGrid slots={state.data.slots} onSelect={setSelected}/></div>{selected && <EventEvidence event={selected}/>}</>}</div>
}
