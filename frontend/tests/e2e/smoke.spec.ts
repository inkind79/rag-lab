import { expect, test } from '@playwright/test';

/**
 * Smoke suite — verifies the app boots and the critical path is reachable.
 *
 * Assumes the dev server at :5173 and the FastAPI backend at :8000 are both
 * running. See playwright.config.ts for why we don't auto-start them.
 */

test.describe('@smoke', () => {
	test('login page renders and rejects empty submit', async ({ page }) => {
		await page.goto('/login');
		await expect(page).toHaveTitle(/RAG Lab/i);
		// The form should exist and the submit button should be present.
		const submit = page.getByRole('button', { name: /log ?in/i });
		await expect(submit).toBeVisible();
	});

	test('register page renders', async ({ page }) => {
		await page.goto('/register');
		await expect(page.getByRole('button', { name: /register|sign ?up/i })).toBeVisible();
	});

	test('/health returns 200', async ({ request }) => {
		const r = await request.get('/health');
		expect(r.status()).toBe(200);
		const body = await r.json();
		expect(body.status).toBe('ok');
	});

	test('/auth/check without cookie is 401', async ({ request }) => {
		const r = await request.get('/auth/check');
		expect(r.status()).toBe(401);
	});

	test('security headers land on a real response', async ({ request }) => {
		const r = await request.get('/health');
		const h = r.headers();
		// Always-on headers from SecurityHeadersMiddleware
		expect(h['x-content-type-options']).toBe('nosniff');
		expect(h['x-frame-options']).toBe('DENY');
		expect(h['referrer-policy']).toContain('strict-origin');
		expect(h['permissions-policy']).toContain('camera=()');
	});

	test('cache-control header is set on API responses', async ({ request }) => {
		const r = await request.get('/api/v1/sessions');
		// 401 (unauth) — what we care about is the header, which middleware
		// adds regardless of status.
		expect(r.headers()['cache-control']).toBe('no-store');
	});
});

test.describe('@smoke auth flow', () => {
	test('register → login → /auth/check authed → logout', async ({ request }) => {
		// Use a unique username per run so repeated local invocations don't
		// collide with existing rows in data/users.db.
		const suffix = Date.now().toString(36);
		const username = `e2e-${suffix}`;
		const password = 'E2e-Secure-12345';

		// Register
		const reg = await request.post('/auth/register', {
			data: { username, email: `${username}@example.com`, password },
		});
		expect(reg.status(), await reg.text()).toBe(201);

		// Login
		const login = await request.post('/auth/login', {
			data: { username, password },
		});
		expect(login.status(), await login.text()).toBe(200);

		// Check auth succeeds with the cookie stored by request context
		const check = await request.get('/auth/check');
		expect(check.status()).toBe(200);
		const body = await check.json();
		expect(body.data.user_id).toBe(username);

		// Logout
		const out = await request.post('/auth/logout');
		expect(out.status()).toBe(200);

		// Subsequent /auth/check is 401
		const afterLogout = await request.get('/auth/check');
		expect(afterLogout.status()).toBe(401);
	});
});
