import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, proxy API + SSE calls to the FastAPI backend so the browser talks to
// one origin (avoids CORS entirely and keeps EventSource happy).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Backend serves everything under /api, so forward the prefix as-is.
      "/api": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
