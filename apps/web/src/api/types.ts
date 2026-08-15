export type Decision = "PASS" | "FAIL" | "REVIEW"

export interface InspectionEvent {
  event_uuid: string
  device_id: string
  product_id: string
  batch_id: string
  tray_id: string
  slot_index: string
  station: string
  decision: Decision
  confidence: number
  defect_code: string | null
  reason_code: string
  image_url: string
  bbox: Array<{x: number; y: number; w: number; h: number}>
  measures_3d: Record<string, number>
  model_version: string | null
  created_at: string
}

export interface ItemList<T> { items: T[]; total: number }
export interface DashboardSummary {
  counts: {total: number; pass: number; fail: number; review: number}
  open_alerts: number
  defect_trend: Array<{time: string; decision: Decision}>
}
export interface TrayResponse {tray_id: string; slots: InspectionEvent[]}
export interface ReviewCommand {event_uuid: string; decision: "PASS" | "FAIL"; defect_code: string | null; comment: string; reviewer: string}
export interface ReviewResult {review_id: number; event_uuid: string; golden_status: string}
export interface AlertRecord {alert_id: string; station: string; defect_rate: number; threshold: number; sample_count: number; status: "OPEN" | "ACKNOWLEDGED" | "CLOSED"; acknowledged_by: string | null}
export interface ReportRecord {report_id: string; alert_id: string; status: string; summary: string; observed_facts: string[]; open_questions: string[]; event_uuids: string[]}
export interface ModelRelease {model_version: string; status: string; metrics: Record<string, string | number>}
export interface ProjectProfile {name: string; version: string; period: string; team_count: number; team: string[]; agents: string[]; quality_targets: Record<string, string>; compute: string[]}
