import {api} from "../api/client"
import {AsyncState} from "../components/AsyncState"
import {PageHeader} from "../components/PageHeader"
import {StatusBadge} from "../components/StatusBadge"
import {useAsync} from "../hooks/useAsync"


export function ModelsPage() {
  const state = useAsync(api.modelReleases, [])
  return <div><PageHeader title="模型治理" subtitle="生产版本、影子对比、门禁证据与回退状态"/><AsyncState loading={state.loading} error={state.error} onRetry={state.reload}/><div className="model-table">{state.data?.items.map((model) => <section className="model-row" key={model.model_version}><div><span className="mono">{model.model_version}</span><StatusBadge value={model.status}/></div><dl>{Object.entries(model.metrics).map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{String(value)}</dd></div>)}</dl><button className="icon-button" title="查看版本详情">···</button></section>)}</div></div>
}
