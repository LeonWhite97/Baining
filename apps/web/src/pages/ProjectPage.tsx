import {api} from "../api/client"
import {AsyncState} from "../components/AsyncState"
import {PageHeader} from "../components/PageHeader"
import {useAsync} from "../hooks/useAsync"


export function ProjectPage() {
  const state = useAsync(api.projectProfile, [])
  return <div><PageHeader title="项目说明" subtitle="架构边界、团队投入、AI 能力与阶段目标"/><AsyncState loading={state.loading} error={state.error} onRetry={state.reload}/>{state.data && <div className="project-bands"><section><h2>{state.data.name}</h2><p>{state.data.version} · {state.data.period} · {state.data.team_count} 人按阶段参与</p><div className="flow-line"><span>PIS-IN / Simulator</span><b>→</b><span>Source Key</span><b>→</b><span>YOLOv8 / TensorRT</span><b>→</b><span>2D + 3D 决策</span><b>→</b><span>PASS / FAIL / REVIEW</span></div></section><section><h2>3 个离线治理 Agent</h2><div className="three-columns">{state.data.agents.map((agent) => <div key={agent}><strong>{agent}</strong><p>基于 PostgreSQL + pgvector 的可引用知识证据，不进入实时自动 PASS 链路。</p></div>)}</div></section><section><h2>质量目标</h2><div className="target-ladder"><div><span>原 AOI 基线</span><b>{state.data.quality_targets.baseline}</b></div><div><span>PoC 门禁</span><b>{state.data.quality_targets.poc}</b></div><div><span>受控上线</span><b>{state.data.quality_targets.controlled_rollout}</b></div><div><span>成熟阶段</span><b>{state.data.quality_targets.mature}</b></div></div></section><section><h2>推荐算力</h2><ul>{state.data.compute.map((item) => <li key={item}>{item}</li>)}</ul></section></div>}</div>
}
