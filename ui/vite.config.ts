import path from "path"
import { readFileSync } from "fs"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"
import { defineConfig } from "vite"
import { earloopEngineBridgePlugin } from "./dev/engine-bridge"

const packageJson = JSON.parse(readFileSync(path.resolve(__dirname, "./package.json"), "utf-8")) as { version: string }

export default defineConfig({
  base: "./",
  define: {
    __APP_VERSION__: JSON.stringify(packageJson.version),
  },
  plugins: [react(), tailwindcss(), earloopEngineBridgePlugin()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
})
