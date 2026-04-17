import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for RAG Lab end-to-end tests.
 *
 * These tests drive a real Chromium browser against the SvelteKit dev server
 * (with the Vite proxy pointing at the FastAPI backend at :8000).
 *
 * They're **not** part of `npm run check` and they don't run in CI today —
 * E2E needs the full backend + Ollama running, which isn't a fit for a
 * lightweight CI stage. Run locally before shipping a UI change:
 *
 *     cd frontend
 *     npm run test:e2e              # full suite, headless
 *     npm run test:e2e -- --ui      # Playwright UI runner
 *     npm run test:e2e -- --headed  # watch the browser
 *
 * A smoke-only subset (tagged @smoke) is fast enough to gate merges on once
 * we wire an integration-only CI runner.
 */
export default defineConfig({
	testDir: './tests/e2e',
	fullyParallel: false,      // shared server state — keep ordering deterministic
	forbidOnly: !!process.env.CI,
	retries: process.env.CI ? 2 : 0,
	workers: 1,                // one worker so tests don't race on the session DB
	reporter: process.env.CI ? 'github' : 'list',

	use: {
		baseURL: process.env.RAGLAB_BASE_URL || 'http://localhost:5173',
		trace: 'on-first-retry',
		screenshot: 'only-on-failure',
		video: 'retain-on-failure',
	},

	projects: [
		{
			name: 'chromium',
			use: { ...devices['Desktop Chrome'] },
		},
	],

	// Don't auto-start the dev server — it needs the backend to be up first,
	// and the backend needs ColPali/Ollama. Tests assume the stack is already
	// running. Document this in README: "npm run dev + uvicorn before npm run test:e2e".
});
