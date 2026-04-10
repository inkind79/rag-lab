<script lang="ts">
	import { page } from '$app/state';
</script>

<div class="error-page">
	<div class="error-card">
		<div class="error-code">{page.status}</div>
		<h1>{page.status === 404 ? 'Page Not Found' : 'Something Went Wrong'}</h1>
		<p class="error-message">{page.error?.message || 'An unexpected error occurred.'}</p>
		<a href="/" class="back-link">Back to RAG Lab</a>
	</div>
</div>

<style>
	.error-page {
		display: flex; align-items: center; justify-content: center;
		min-height: 100vh; background: var(--bg-page);
	}
	.error-card {
		text-align: center; background: var(--bg-surface); border-radius: 12px;
		box-shadow: 0 4px 20px var(--shadow);
		padding: 2.5rem; max-width: 420px;
	}
	.error-code {
		font-size: 3rem; font-weight: 800; color: var(--border);
		line-height: 1; margin-bottom: 0.5rem;
	}
	h1 { margin: 0 0 0.75rem; font-size: 1.2rem; color: var(--text-primary); }
	.error-message {
		color: var(--text-secondary); font-size: 0.9rem; margin: 0 0 1.5rem;
		line-height: 1.5;
	}
	.back-link {
		display: inline-block; padding: 0.5rem 1.25rem;
		background: var(--accent); color: white; border-radius: 8px;
		text-decoration: none; font-size: 0.88rem; font-weight: 600;
	}
	.back-link:hover { filter: brightness(0.9); }
</style>
