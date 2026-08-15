import type {AlertRecord, DashboardSummary, InspectionEvent, ItemList, ModelRelease, ProjectProfile, ReportRecord, ReviewCommand, ReviewResult, TrayResponse} from "./types"


async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const apiOrigin = import.meta.env.VITE_API_ORIGIN ?? ""
  const response = await fetch(`${apiOrigin}/api/v1${path}`, {
    ...init,
    headers: {"Content-Type": "application/json", ...init?.headers},
  })
  if (!response.ok) throw new Error(`API ${response.status}: ${await response.text()}`)
  return response.json() as Promise<T>
}


export const api = {
  dashboard: () => request<DashboardSummary>("/dashboard/summary"),
  inspections: (decision?: string) => request<ItemList<InspectionEvent>>(`/inspections${decision ? `?decision=${decision}` : ""}`),
  inspection: (id: string) => request<InspectionEvent>(`/inspections/${id}`),
  tray: (id: string) => request<TrayResponse>(`/trays/${id}`),
  reviews: () => request<ItemList<InspectionEvent>>("/reviews"),
  createReview: (payload: ReviewCommand) => request<ReviewResult>("/reviews", {method: "POST", body: JSON.stringify(payload)}),
  alerts: () => request<ItemList<AlertRecord>>("/alerts"),
  acknowledgeAlert: (id: string) => request<AlertRecord>(`/alerts/${id}/acknowledge`, {method: "POST", body: JSON.stringify({operator: "line_leader"})}),
  closeAlert: (id: string) => request<AlertRecord>(`/alerts/${id}/close`, {method: "POST", body: JSON.stringify({operator: "line_leader"})}),
  reports: () => request<ItemList<ReportRecord>>("/reports"),
  createReport: (alertId: string) => request<ReportRecord>("/reports", {method: "POST", body: JSON.stringify({alert_id: alertId})}),
  modelReleases: () => request<{items: ModelRelease[]}>("/model-releases"),
  projectProfile: () => request<ProjectProfile>("/project-profile"),
}
