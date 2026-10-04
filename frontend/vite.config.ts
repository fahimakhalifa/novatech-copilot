import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the browser talks only to Vite (port 5173), which forwards /api to the
// FastAPI server. One origin: no CORS preflight, and the same URLs work behind Nginx in Docker.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://localhost:8000" } },
});
