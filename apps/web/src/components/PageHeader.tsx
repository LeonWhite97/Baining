import type {ReactNode} from "react"


export function PageHeader({title, subtitle, actions}: {title: string; subtitle: string; actions?: ReactNode}) {
  return <header className="page-header"><div><h1>{title}</h1><p>{subtitle}</p></div>{actions && <div className="page-actions">{actions}</div>}</header>
}
