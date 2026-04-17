/**
 * Client-side hooks.
 *
 * Sentry browser SDK is initialized when PUBLIC_SENTRY_DSN is set at build
 * time (Vite inlines PUBLIC_* into the bundle). Falls through quietly when
 * the dep isn't installed or the DSN isn't set, so the default OSS install
 * ships nothing to any external service.
 */

import type { HandleClientError } from '@sveltejs/kit';

const dsn = import.meta.env.PUBLIC_SENTRY_DSN as string | undefined;
let sentryReady = false;

if (dsn) {
	// Dynamic import so the bundle stays lean for users who don't ship Sentry.
	import('@sentry/sveltekit')
		.then((Sentry) => {
			Sentry.init({
				dsn,
				environment: (import.meta.env.PUBLIC_SENTRY_ENVIRONMENT as string) || import.meta.env.MODE,
				release: (import.meta.env.PUBLIC_SENTRY_RELEASE as string) || undefined,
				tracesSampleRate: Number(import.meta.env.PUBLIC_SENTRY_TRACES_SAMPLE_RATE ?? 0),
			});
			sentryReady = true;
		})
		.catch((err) => {
			// Don't crash the app over an observability failure.
			console.warn('[sentry] init failed; continuing without it:', err);
		});
}

export const handleError: HandleClientError = ({ error, event }) => {
	// Log to console regardless — Sentry capture is best-effort.
	console.error('[client error]', error, event);
	if (sentryReady) {
		// Lazy import again to avoid pulling Sentry into the path when DSN isn't set.
		import('@sentry/sveltekit')
			.then((Sentry) => Sentry.captureException(error))
			.catch(() => {});
	}
};
