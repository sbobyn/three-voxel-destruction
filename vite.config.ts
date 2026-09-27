import { defineConfig } from 'vite';

export default defineConfig({
  resolve: {
    // One Three.js, even if three-avbd is linked from a checkout of its own (pnpm link) to work on both
    dedupe: ['three'],
  },
  server: {
    // WebGPU needs a secure page, so a phone on the network reaches the dev server through an
    // HTTPS tunnel (cloudflared tunnel --url http://localhost:5320): let its hostnames in
    allowedHosts: ['.trycloudflare.com'],
  },
  build: {
    target: 'es2022',
    outDir: 'dist',
    rollupOptions: {
      input: { main: 'index.html', skyLab: 'sky-lab.html', particlesLab: 'particles-lab.html' },
    },
  },
});
