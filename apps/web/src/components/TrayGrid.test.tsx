import {render, screen} from "@testing-library/react"
import userEvent from "@testing-library/user-event"

import {TrayGrid} from "./TrayGrid"
import type {InspectionEvent} from "../api/types"


const slots: InspectionEvent[] = Array.from({length: 12}, (_, index) => ({
  event_uuid: `event-${index + 1}`,
  device_id: "PIS-01",
  product_id: "BGA-256",
  batch_id: "LOT-01",
  tray_id: "TRAY-001",
  slot_index: String(index + 1).padStart(2, "0"),
  station: "ST-01",
  decision: index === 4 ? "FAIL" : "PASS",
  confidence: 0.98,
  defect_code: index === 4 ? "BALL_BRIDGE" : null,
  reason_code: index === 4 ? "DEFECT_SCORE" : "POLICY_AUTO_PASS",
  image_url: `/api/v1/demo/images/event-${index + 1}.svg`,
  bbox: [],
  measures_3d: {},
  model_version: "yolov8s-aoi-3.5.2",
  created_at: "2024-12-18T08:00:00+00:00",
}))


it("opens the exact event represented by a tray slot", async () => {
  const selected: string[] = []
  render(<TrayGrid slots={slots} onSelect={(event) => selected.push(event.event_uuid)}/>)

  await userEvent.click(screen.getByRole("button", {name: /槽位 05.*FAIL/}))

  expect(selected).toEqual(["event-5"])
  expect(screen.getAllByRole("button")).toHaveLength(12)
})
