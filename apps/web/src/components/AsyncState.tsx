export function AsyncState({loading, error, empty, onRetry}: {loading: boolean; error: string | null; empty?: boolean; onRetry: () => void}) {
  if (loading) return <div className="async-state">数据加载中</div>
  if (error) return <div className="async-state error"><span>{error}</span><button onClick={onRetry}>重试</button></div>
  if (empty) return <div className="async-state">暂无数据</div>
  return null
}
