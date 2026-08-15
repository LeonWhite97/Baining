import {useMemo} from "react"
import {AlertTriangle, CheckCircle2, ClipboardCheck, ScanLine} from "lucide-react"
import type {EChartsOption} from "echarts"

import {api} from "../api/client"
import {AsyncState} from "../components/AsyncState"
import {EChart} from "../components/EChart"
import {PageHeader} from "../components/PageHeader"
import {useAsync} from "../hooks/useAsync"


export function DashboardPage() {
  const state = useAsync(api.dashboard, [])
  const option = useMemo<EChartsOption>(() => ({grid: {left: 32, right: 12, top: 24, bottom: 28}, xAxis: {type: "category", data: state.data?.defect_trend.map((item) => item.time) ?? [], axisLabel: {color: "#64747a"}}, yAxis: {type: "value", min: 0, max: 1, axisLabel: {show: false}}, series: [{type: "line", step: "middle", symbol: "circle", data: state.data?.defect_trend.map((item) => item.decision === "PASS" ? 0.1 : item.decision === "REVIEW" ? 0.55 : 0.9) ?? [], lineStyle: {color: "#328f82", width: 2}, itemStyle: {color: "#e85b5b"}, areaStyle: {color: "rgba(73,183,165,.08)"}}]}), [state.data])
  return <div><PageHeader title="生产质量总览" subtitle="PIS-IN 旁路复判 · 2D/3D 证据融合 · 三态决策" actions={<span className="live-chip"><span/>实时事件流</span>}/>
    <AsyncState loading={state.loading} error={state.error} onRetry={state.reload}/>{state.data && <>
      <section className="metric-strip">
        <div><ScanLine/><span>检测总数</span><strong>{state.data.counts.total}</strong></div>
        <div><CheckCircle2/><span>PASS</span><strong>{state.data.counts.pass}</strong></div>
        <div><AlertTriangle/><span>FAIL</span><strong>{state.data.counts.fail}</strong></div>
        <div><ClipboardCheck/><span>REVIEW</span><strong>{state.data.counts.review}</strong></div>
      </section>
      <section className="workspace-grid"><div className="panel chart-panel"><div className="panel-title"><h2>最近检测序列</h2><span>判定风险阶梯</span></div><EChart option={option} ariaLabel="最近检测判定趋势"/></div>
      <div className="panel alert-summary"><div className="panel-title"><h2>工站门禁</h2><span>滚动窗口</span></div><strong>{state.data.open_alerts}</strong><p>个工站预警待处置</p><div className="threshold-row"><span>ST-02 缺陷率</span><b>18.0%</b></div><div className="threshold-row"><span>触发阈值</span><b>8.0%</b></div></div></section>
    </>}</div>
}
