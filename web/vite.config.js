import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// build straight into the python package so it ships as bundled static assets
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../visprompter/ui/static", emptyOutDir: true },
});
