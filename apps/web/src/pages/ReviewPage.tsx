import {useEffect, useState} from "react"

import {api} from "../api/client"
import type {InspectionEvent, ItemList, ReviewCommand, ReviewResult} from "../api/types"
import {EventEvidence} from "../components/EventEvidence"
import {PageHeader} from "../components/PageHeader"


export interface ReviewService {list: () => Promise<ItemList<InspectionEvent>>; create: (payload: ReviewCommand) => Promise<ReviewResult>}
const defaultService: ReviewService = {list: api.reviews, create: api.createReview}


export function ReviewPage({service = defaultService}: {service?: ReviewService}) {
  const [items, setItems] = useState<InspectionEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [message, setMessage] = useState("")
  const [error, setError] = useState("")
  useEffect(() => { service.list().then((result) => setItems(result.items)).catch((reason) => setError(String(reason))).finally(() => setLoading(false)) }, [service])
  const submit = async (decision: "PASS" | "FAIL") => {
    const current = items[0]
    if (!current) return
    setError("")
    try {
      await service.create({event_uuid: current.event_uuid, decision, defect_code: decision === "FAIL" ? current.defect_code ?? "UNKNOWN" : null, comment: decision === "FAIL" ? "复核确认缺陷" : "复核确认正常", reviewer: "qa_demo"})
      setItems((existing) => existing.slice(1))
      setMessage("复核结果已保存并进入金标准集")
    } catch (reason) { setError(reason instanceof Error ? reason.message : "复核保存失败") }
  }
  return <div><PageHeader title="人工复核" subtitle="仅处理 REVIEW 事件，人工结论写入金标准回流记录"/>
    {message && <div className="notice success">{message}</div>}{error && <div className="notice error">{error}</div>}
    {loading ? <div className="async-state">复核队列加载中</div> : items[0] ? <>
      <div className="review-toolbar"><span>待复核 {items.length} 件</span><div><button className="button secondary" onClick={() => void submit("PASS")}>确认正常</button><button className="button danger" onClick={() => void submit("FAIL")}>确认缺陷</button></div></div>
      <EventEvidence event={items[0]}/>
    </> : <div className="empty-state">当前没有待复核事件</div>}
  </div>
}
