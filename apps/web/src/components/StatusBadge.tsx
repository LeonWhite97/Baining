import type {Decision} from "../api/types"


export function StatusBadge({value}: {value: Decision | string}) {
  return <span className={`status-badge status-${value.toLowerCase()}`}>{value}</span>
}
