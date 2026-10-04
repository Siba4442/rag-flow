import { defineConfig } from "vite";

// Dev: the UI calls /api/*, proxied to the FastAPI backend with the prefix stripped.
// Prod: built with base /ui/ so FastAPI can serve dist/ (SERVE_FRONTEND=true).
export default defineConfig(({ command }) => ({
  base: command === "build" ? "/ui/" : "/",
  server: {
    port: 5173,
    // WSL: files under /mnt/c don't emit inotify events, so poll for changes there.
    watch: process.cwd().startsWith("/mnt/") ? { usePolling: true, interval: 300 } : null,
    proxy: {
      "/api": {
        target: process.env.RAGFLOW_API ?? "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
}));
