import {render, screen} from "@testing-library/react"
import {MemoryRouter} from "react-router-dom"

import {AppShell} from "./AppShell"


it("renders all eight operational destinations without budget navigation", () => {
  render(
    <MemoryRouter>
      <AppShell><div>当前页面</div></AppShell>
    </MemoryRouter>,
  )

  for (const label of ["总览", "实时检测", "Tray Map", "人工复核", "缺陷报表", "预警与报告", "模型治理", "项目说明"]) {
    expect(screen.getByRole("link", {name: label})).toBeInTheDocument()
  }
  expect(screen.queryByText(/预算/)).not.toBeInTheDocument()
})
