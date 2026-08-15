import {useState} from "react"

import {api} from "../api/client"
import type {ReportRecord} from "../api/types"
import {AsyncState} from "../components/AsyncState"
import {PageHeader} from "../components/PageHeader"
import {StatusBadge} from "../components/StatusBadge"
import {useAsync} from "../hooks/useAsync"


export function AlertsPage() {
  const state = useAsync(api.alerts, [])
  const reports = useAsync(api.reports, [])
  const [activeReport, setActiveReport] = useState<ReportRecord | null>(null)
  const acknowledge = async (id: string) => {await api.acknowledgeAlert(id); await state.reload()}
  const createReport = async (id: string) => {const report = await api.createReport(id); setActiveReport(report); await reports.reload()}
  return <div><PageHeader title="预警与异常报告" subtitle="高缺陷率触发工站推送，处置证据和报告全程可追溯"/><AsyncState loading={state.loading} error={state.error} onRetry={state.reload}/><section className="workspace-grid"> <div className="panel"><div className="panel-title"><h2>工站预警</h2><span>滚动窗口门禁</span></div><div className="alert-list">{state.data?.items.map((alert) => <div className="alert-row" key={alert.alert_id}><div><strong>{alert.station}</strong><span>{alert.alert_id}</span></div><div><b>{(alert.defect_rate * 100).toFixed(1)}%</b><small>阈值 {(alert.threshold * 100).toFixed(1)}%</small></div><StatusBadge value={alert.status}/><div className="row-actions">{alert.status === "OPEN" && <button className="button secondary" onClick={() => void acknowledge(alert.alert_id)}>确认预警</button>}{alert.status === "ACKNOWLEDGED" && <button className="button primary" onClick={() => void createReport(alert.alert_id)}>生成异常报告</button>}</div></div>)}</div></div>
    <div className="panel"><div className="panel-title"><h2>报告记录</h2><span>{reports.data?.total ?? 0} 份</span></div>{activeReport && <div className="report-preview"><StatusBadge value={activeReport.status}/><h3>{activeReport.summary}</h3><ul>{activeReport.observed_facts.map((fact) => <li key={fact}>{fact}</li>)}</ul></div>}<div className="compact-list">{reports.data?.items.map((report) => <button key={report.report_id} onClick={() => setActiveReport(report)}><span>{report.report_id.slice(0, 18)}</span><StatusBadge value={report.status}/></button>)}</div></div></section></div>
}
