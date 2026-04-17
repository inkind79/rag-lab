import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

// Backend URL for the dev proxy. Override with RAGLAB_BACKEND_URL so the
// same config works locally (default :8000) and in CI / E2E (e.g. :8002).
const BACKEND = process.env.RAGLAB_BACKEND_URL || 'http://localhost:8000';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		port: 5173,
		proxy: {
			'/api': { target: BACKEND, changeOrigin: true },
			'/auth': { target: BACKEND, changeOrigin: true },
			'/document': { target: BACKEND, changeOrigin: true },
			'/static': { target: BACKEND, changeOrigin: true },
			'/health': { target: BACKEND, changeOrigin: true },
		},
	},
});
