import {render, screen} from "@testing-library/react"
import userEvent from "@testing-library/user-event"

import {ReviewPage, type ReviewService} from "./ReviewPage"


const event = {
  event_uuid: "event-review-1",
  device_id: "PIS-01",
  product_id: "BGA-256",
  batch_id: "LOT-01",
  tray_id: "TRAY-001",
  slot_index: "09",
  station: "ST-01",
  decision: "REVIEW" as const,
  confidence: 0.71,
  defect_code: "UNKNOWN",
  reason_code: "LOW_CONFIDENCE",
  image_url: "/api/v1/demo/images/event-review-1.svg",
  bbox: [{x: 58, y: 42, w: 24, h: 18}],
  measures_3d: {ball_height_max: 0.43},
  model_version: "yolov8s-aoi-3.5.2",
  created_at: "2024-12-18T08:00:00+00:00",
}


it("removes a reviewed event only after the persisted command succeeds", async () => {
  const service: ReviewService = {
    list: async () => ({items: [event], total: 1}),
    create: async () => ({review_id: 1, event_uuid: event.event_uuid, golden_status: "CONFIRMED"}),
  }
  render(<ReviewPage service={service}/>)

  await userEvent.click(await screen.findByRole("button", {name: "确认缺陷"}))

  expect(await screen.findByText("复核结果已保存并进入金标准集")).toBeInTheDocument()
  expect(screen.getByText("当前没有待复核事件")).toBeInTheDocument()
})
