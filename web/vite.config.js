import { defineConfig } from 'vite';

// 开发时把 /api 代理到 FastAPI（main.py，默认 127.0.0.1:8000）
export default defineConfig({
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
  build: { chunkSizeWarningLimit: 1500 },
});
