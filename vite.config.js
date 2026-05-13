import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  assetsInclude: ['**/*.onnx', '**/*.wasm'],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('onnxruntime-web')) return 'ort';
          if (id.includes('node_modules')) return 'vendor';
        }
      }
    }
  }
})
