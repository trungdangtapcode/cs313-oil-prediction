import react from "@vitejs/plugin-react";
import { defineConfig, type UserConfig } from "vite";

type VitestUserConfig = UserConfig & {
  test: {
    include: string[];
    environment: string;
  };
};

const config: VitestUserConfig = {
  plugins: [react()],
  test: {
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
    environment: "jsdom",
  },
  server: {
    port: 5173,
  },
  preview: {
    port: 4173,
  },
  build: {
    sourcemap: true,
  },
};

export default defineConfig(config);
