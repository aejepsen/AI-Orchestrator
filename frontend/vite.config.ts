import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev: o frontend roda em :5173 e proxia /chat para o gateway em :8100.
// Prod: o gateway serve o dist diretamente (mesma origem, sem proxy).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/chat": { target: "http://localhost:8100", changeOrigin: true },
      "/health": { target: "http://localhost:8100", changeOrigin: true },
    },
  },
});
