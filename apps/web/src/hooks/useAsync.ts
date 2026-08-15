import {useCallback, useEffect, useState} from "react"


export function useAsync<T>(loader: () => Promise<T>, dependencies: unknown[] = []) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try { setData(await loader()) } catch (reason) { setError(reason instanceof Error ? reason.message : "加载失败") } finally { setLoading(false) }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, dependencies)
  useEffect(() => { void load() }, [load])
  return {data, error, loading, reload: load, setData}
}
