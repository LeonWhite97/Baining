import {defineConfig} from "@playwright/test"


const e2eDatabasePath = `./playwright-aoi-${process.pid}.db`

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 30_000,
  use: {baseURL: "http://127.0.0.1:4173", trace: "retain-on-failure"},
  webServer: [
    {
      command: "python -m uvicorn app.main:app --app-dir ../api --host 127.0.0.1 --port 8014",
      url: "http://127.0.0.1:8014/api/v1/health",
      env: {...process.env, APP_MODE: "demo", DEMO_AUTO_SEED: "202408", DATABASE_URL: `sqlite+pysqlite:///${e2eDatabasePath}`},
      reuseExistingServer: true,
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 4173",
      url: "http://127.0.0.1:4173",
      env: {...process.env, VITE_API_PROXY_TARGET: "http://127.0.0.1:8014"},
      reuseExistingServer: true,
    },
  ],
})
