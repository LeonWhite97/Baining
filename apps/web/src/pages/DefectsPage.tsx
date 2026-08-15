import {useMemo} from "react"
import type {EChartsOption} from "echarts"

import {api} from "../api/client"
import {AsyncState} from "../components/AsyncState"
import {EChart} from "../components/EChart"
import {PageHeader} from "../components/PageHeader"
import {StatusBadge} from "../components/StatusBadge"
import {useAsync} from "../hooks/useAsync"


export function DefectsPage() {
  const state = useAsync(api.inspections, [])
  const defects = useMemo(() => {const counts = new Map<string, number>(); state.data?.items.forEach((item) => {if (item.defect_code && item.defect_code !== "UNKNOWN") counts.set(item.defect_code, (counts.get(item.defect_code) ?? 0) + 1)}); return [...counts.entries()]}, [state.data])
  const option = useMemo<EChartsOption>(() => ({grid: {left: 92, right: 24, top: 12, bottom: 28}, xAxis: {type: "value"}, yAxis: {type: "category", data: defects.map(([name]) => name)}, series: [{type: "bar", data: defects.map(([, count]) => count), itemStyle: {color: "#d45454"}, barWidth: 16}]}), [defects])
  return <div><PageHeader title="缺陷报表" subtitle="按缺陷、工站、批次和模型版本分析质量分布"/><AsyncState loading={state.loading} error={state.error} onRetry={state.reload}/>{state.data && <section className="workspace-grid"><div className="panel chart-panel"><div className="panel-title"><h2>缺陷分类</h2><span>当前批次</span></div><EChart option={option} ariaLabel="缺陷分类数量"/></div><div className="panel"><div className="panel-title"><h2>异常明细</h2><span>{state.data.items.filter((item) => item.decision !== "PASS").length} 条</span></div><div className="compact-list">{state.data.items.filter((item) => item.decision !== "PASS").map((item) => <div key={item.event_uuid}><span>{item.station} · {item.tray_id}/{item.slot_index}</span><span>{item.defect_code}</span><StatusBadge value={item.decision}/></div>)}</div></div></section>}</div>
}
