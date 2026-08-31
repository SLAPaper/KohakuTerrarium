import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'

const here = path.dirname(fileURLToPath(import.meta.url))
const webview = path.resolve(here, 'src/webview')

export default defineConfig({
  root: webview,
  build: {
    outDir: path.join(here, 'dist'),
    emptyOutDir: false,
    lib: {
      entry: path.join(webview, 'index.js'),
      formats: ['iife'],
      name: 'KohakuTerrariumVsCode',
      fileName: () => 'webview.js',
      cssFileName: 'webview',
    },
    sourcemap: true,
  },
})
