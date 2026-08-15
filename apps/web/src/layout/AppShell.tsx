import type {ReactNode} from "react"
import {AlertTriangle, Boxes, ChartNoAxesCombined, CircleGauge, ClipboardCheck, FileChartColumn, ScanLine, Settings2} from "lucide-react"
import {NavLink} from "react-router-dom"

import "./AppShell.css"


const destinations = [
  {to: "/", label: "总览", icon: CircleGauge},
  {to: "/inspections", label: "实时检测", icon: ScanLine},
  {to: "/trays", label: "Tray Map", icon: Boxes},
  {to: "/reviews", label: "人工复核", icon: ClipboardCheck},
  {to: "/defects", label: "缺陷报表", icon: ChartNoAxesCombined},
  {to: "/alerts", label: "预警与报告", icon: AlertTriangle},
  {to: "/models", label: "模型治理", icon: Settings2},
  {to: "/project", label: "项目说明", icon: FileChartColumn},
]


export function AppShell({children}: {children: ReactNode}) {
  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div className="brand-block">
          <span className="brand-mark">PI</span>
          <div>
            <strong>PIS-IN AOI</strong>
            <span>AI 质检控制台</span>
          </div>
        </div>
        <nav aria-label="主导航">
          {destinations.map(({to, label, icon: Icon}) => (
            <NavLink key={to} to={to} end={to === "/"}>
              <Icon aria-hidden="true" size={18}/>
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="system-state">
          <span className="pulse-dot"/>
          <div><strong>DEMO 在线</strong><span>V3.5 · 无 GPU 模式</span></div>
        </div>
      </aside>
      <main className="app-main">{children}</main>
    </div>
  )
}
