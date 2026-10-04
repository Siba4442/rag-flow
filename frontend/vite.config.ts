import { resolve } from "node:path";
import { defineConfig } from "vite";

// Dev: the UI calls /api/*, proxied to the FastAPI backend with the prefix stripped.
// Prod: built with base /ui/ so FastAPI can serve dist/ (SERVE_FRONTEND=true).
import type { Plugin } from "vite";

// Dev: /design → /design/ so the style tile opens instead of the SPA fallback.
const designRoute: Plugin = {
  name: "design-route",
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      if (req.url === "/design") {
        res.statusCode = 302;
        res.setHeader("Location", "/design/");
        res.end();
        return;
      }
      next();
    });
  },
};

export default defineConfig(({ command }) => ({
  plugins: [designRoute],
  base: command === "build" ? "/ui/" : "/",
  build: {
    rollupOptions: {
      // App + the design-system style tile (/design).
      input: { main: resolve(__dirname, "index.html"), design: resolve(__dirname, "design/index.html") },
    },
  },
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
