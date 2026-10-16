import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// In development the SPA proxies /api to the FastAPI server so there is no
// CORS friction. The target is resolved in this order:
//   1. VITE_BACKEND_URL from frontend/.env   (recommended, per machine)
//   2. the VITE_BACKEND_URL shell variable
//   3. http://127.0.0.1:8000                 (default)
// Copy frontend/.env.example to frontend/.env and edit it if your backend
// does not run on port 8000 (for example when another service owns 8000).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'VITE_')
  const backendUrl =
    env.VITE_BACKEND_URL || process.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000'

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        '/api': {
          target: backendUrl,
          changeOrigin: true,
        },
      },
    },
    preview: { port: 4173 },
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            react: ['react', 'react-dom'],
          },
        },
      },
    },
    test: {
      environment: 'jsdom',
      globals: true,
      setupFiles: './src/tests/setup.js',
      css: false,
      include: ['src/**/*.test.{js,jsx}'],
    },
  }
})
