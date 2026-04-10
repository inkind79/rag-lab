<script lang="ts">
	import { goto } from '$app/navigation';
	import { login } from '$lib/api/client';

	let username = $state('');
	let password = $state('');
	let error = $state('');
	let loading = $state(false);
	let showPassword = $state(false);

	async function handleLogin() {
		if (!username || !password) return;
		loading = true;
		error = '';

		try {
			const success = await login(username, password);
			if (success) {
				// Full page load ensures the auth cookie is picked up by the layout
				window.location.href = '/';
				return;
			} else {
				error = 'Invalid username or password';
			}
		} catch (e: unknown) {
			error = e instanceof Error ? e.message : 'Login failed';
		} finally {
			loading = false;
		}
	}
</script>

<div class="login-page">
	<div class="login-scene">
		<div class="ambient-glow"></div>

		<div class="login-card">
			<div class="login-header">
				<div class="logo-mark">
					<svg width="32" height="32" viewBox="0 0 32 32" fill="none">
						<rect x="2" y="2" width="12" height="12" rx="2" fill="currentColor" opacity="0.9"/>
						<rect x="18" y="2" width="12" height="12" rx="2" fill="currentColor" opacity="0.5"/>
						<rect x="2" y="18" width="12" height="12" rx="2" fill="currentColor" opacity="0.5"/>
						<rect x="18" y="18" width="12" height="12" rx="2" fill="currentColor" opacity="0.2"/>
					</svg>
				</div>
				<h1 class="logo-text">RAG LAB</h1>
				<p class="logo-sub">Visual Document Intelligence</p>
			</div>

			<form class="login-form" onsubmit={(e) => { e.preventDefault(); handleLogin(); }}>
				{#if error}
					<div class="error-msg">
						<span class="error-icon">!</span>
						{error}
					</div>
				{/if}

				<label class="field">
					<span>Username</span>
					<input type="text" bind:value={username} placeholder="Enter username" autocomplete="username" autofocus />
				</label>

				<label class="field">
					<span>Password</span>
					<div class="password-wrap">
						<input type={showPassword ? 'text' : 'password'} bind:value={password} placeholder="Enter password" autocomplete="current-password" />
						<button type="button" class="pw-toggle" onclick={() => (showPassword = !showPassword)} tabindex="-1">
							{showPassword ? '/' : 'o'}
						</button>
					</div>
				</label>

				<button type="submit" class="login-btn" disabled={loading || !username || !password}>
					{#if loading}
						<span class="btn-spinner"></span>
						Signing in...
					{:else}
						Sign In
					{/if}
				</button>
			</form>

			<p class="login-footer">
				Don't have an account? <a href="/register" class="register-link">Create one</a>
			</p>
		</div>
	</div>
</div>

<style>
	.login-page {
		display: flex; align-items: center; justify-content: center;
		min-height: 100vh;
		background: var(--bg-page);
		position: relative; overflow: hidden;
	}

	.login-scene { position: relative; z-index: 1; }

	.ambient-glow {
		position: fixed; top: 30%; left: 50%; transform: translate(-50%, -50%);
		width: 600px; height: 400px;
		background: radial-gradient(ellipse, rgba(37, 99, 235, 0.06) 0%, transparent 70%);
		pointer-events: none;
	}

	.login-card {
		background: var(--bg-surface);
		border-radius: 16px;
		border: 1px solid var(--border);
		box-shadow: 0 8px 40px var(--shadow-lg), 0 1px 3px var(--shadow);
		padding: 2.5rem 2.5rem 1.75rem;
		width: 380px;
		animation: cardEnter 0.5s cubic-bezier(0.16, 1, 0.3, 1);
	}
	@keyframes cardEnter {
		from { opacity: 0; transform: translateY(12px); }
		to { opacity: 1; transform: translateY(0); }
	}

	.login-header { text-align: center; margin-bottom: 2rem; }
	.logo-mark { color: var(--text-heading); margin-bottom: 0.75rem; }
	.logo-text {
		font-family: var(--font-sans);
		font-size: 1.5rem; font-weight: 800; letter-spacing: 0.18em;
		color: var(--text-heading); margin: 0;
	}
	.logo-sub {
		font-size: 0.72rem; color: var(--text-muted); margin: 0.3rem 0 0;
		letter-spacing: 0.06em; font-weight: 400;
	}

	.login-form { display: flex; flex-direction: column; gap: 1.1rem; }

	.error-msg {
		display: flex; align-items: center; gap: 0.5rem;
		background: #fef2f2; color: #dc2626; padding: 0.55rem 0.8rem;
		border-radius: 8px; font-size: 0.82rem; border: 1px solid #fecaca;
		animation: shake 0.4s ease;
	}
	@keyframes shake {
		0%, 100% { transform: translateX(0); }
		25% { transform: translateX(-4px); }
		75% { transform: translateX(4px); }
	}
	.error-icon {
		width: 18px; height: 18px; border-radius: 50%; background: #dc2626; color: white;
		font-size: 0.65rem; font-weight: 800; display: flex; align-items: center; justify-content: center;
		flex-shrink: 0;
	}

	.field { display: flex; flex-direction: column; gap: 5px; }
	.field span {
		font-size: 0.72rem; font-weight: 600; color: var(--text-secondary);
		text-transform: uppercase; letter-spacing: 0.06em;
	}
	.field input {
		padding: 0.6rem 0.8rem; border: 1px solid var(--border); border-radius: 8px;
		font-size: 0.88rem; color: var(--text-primary); background: var(--bg-input);
		outline: none; width: 100%; font-family: var(--font-sans);
		transition: border-color 0.15s, box-shadow 0.15s;
	}
	.field input:focus {
		border-color: var(--accent);
		box-shadow: 0 0 0 3px rgba(194, 117, 10, 0.1);
	}
	.field input::placeholder { color: var(--text-muted); }

	.password-wrap { position: relative; display: flex; }
	.password-wrap input { padding-right: 2.5rem; }
	.pw-toggle {
		position: absolute; right: 10px; top: 50%; transform: translateY(-50%);
		background: none; border: none; cursor: pointer;
		font-size: 0.85rem; font-weight: 700; font-family: var(--font-mono);
		padding: 0; line-height: 1; color: var(--text-muted);
		width: 20px; height: 20px; display: flex; align-items: center; justify-content: center;
	}
	.pw-toggle:hover { color: var(--text-secondary); }

	.login-btn {
		background: var(--text-heading); color: var(--bg-surface); border: none;
		padding: 0.65rem; border-radius: 8px;
		font-size: 0.88rem; font-weight: 700; cursor: pointer;
		font-family: var(--font-sans); letter-spacing: 0.03em;
		margin-top: 0.3rem;
		transition: all 0.15s;
		display: flex; align-items: center; justify-content: center; gap: 0.4rem;
	}
	.login-btn:hover:not(:disabled) { opacity: 0.9; transform: translateY(-1px); }
	.login-btn:active:not(:disabled) { transform: translateY(0); }
	.login-btn:disabled { opacity: 0.45; cursor: not-allowed; }

	.btn-spinner {
		width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.3);
		border-top-color: white; border-radius: 50%;
		animation: spin 0.6s linear infinite;
	}
	@keyframes spin { to { transform: rotate(360deg); } }

	.login-footer {
		text-align: center; margin: 1.5rem 0 0; padding-top: 1rem;
		border-top: 1px solid var(--border-light);
		font-size: 0.68rem; color: var(--text-muted); letter-spacing: 0.03em;
	}

	.register-link {
		color: var(--accent); text-decoration: none; font-weight: 600;
	}
	.register-link:hover { text-decoration: underline; }

	@media (max-width: 480px) {
		.login-card { width: 90vw; padding: 2rem 1.5rem 1.5rem; }
	}
</style>
