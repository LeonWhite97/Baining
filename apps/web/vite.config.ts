import react from "@vitejs/plugin-react"
import {loadEnv} from "vite"
import {defineConfig} from "vitest/config"

export default defineConfig(({mode}) => {
  const env = loadEnv(mode, ".", "")
  return {
  plugins: [react()],
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    exclude: ["tests/e2e/**", "node_modules/**", "dist/**"],
  },
  server: {
    proxy: {"/api": env.VITE_API_PROXY_TARGET || "http://localhost:8000"},
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          charts: ["echarts"],
          vendor: ["react", "react-dom", "react-router-dom", "lucide-react"],
        },
      },
    },
  },
  }
})
