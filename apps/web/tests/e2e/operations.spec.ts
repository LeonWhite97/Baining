import {expect, test} from "@playwright/test"


test("inspection can be reviewed, traced and reported", async ({page}) => {
  const reset = await page.request.post("/api/v1/demo/reset", {data: {seed: 202408}})
  expect(reset.ok()).toBeTruthy()
  await page.goto("/")
  await expect(page.getByRole("heading", {name: "生产质量总览"})).toBeVisible()

  await page.getByRole("link", {name: "人工复核"}).click()
  await expect(page.getByText(/待复核 4 件/)).toBeVisible()
  await page.getByRole("button", {name: "确认缺陷"}).click()
  await expect(page.getByText(/待复核 3 件/)).toBeVisible()

  await page.getByRole("link", {name: "Tray Map"}).click()
  await page.getByRole("button", {name: /槽位 01/}).click()
  await expect(page.getByText("TRAY-001 / 01")).toBeVisible()

  await page.getByRole("link", {name: "预警与报告"}).click()
  await page.getByRole("button", {name: "确认预警"}).click()
  await page.getByRole("button", {name: "生成异常报告"}).click()
  await expect(page.getByText("DRAFT").first()).toBeVisible()
})
