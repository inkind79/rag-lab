<script lang="ts">
	import { toasts } from '$lib/stores/toast';
	import type { Toast } from '$lib/stores/toast';

	function typeClass(type: Toast['type']): string {
		switch (type) {
			case 'success': return 'toast-success';
			case 'error': return 'toast-error';
			case 'warning': return 'toast-warning';
			default: return 'toast-info';
		}
	}

	function typeIcon(type: Toast['type']): string {
		switch (type) {
			case 'success': return '\u2713';
			case 'error': return '\u2717';
			case 'warning': return '!';
			default: return 'i';
		}
	}
</script>

{#if $toasts.length > 0}
	<div class="toast-container">
		{#each $toasts as toast (toast.id)}
			<div class="toast {typeClass(toast.type)}">
				<span class="toast-icon">{typeIcon(toast.type)}</span>
				<span class="toast-msg">{toast.message}</span>
				<button class="toast-close" onclick={() => toasts.dismiss(toast.id)}>&times;</button>
			</div>
		{/each}
	</div>
{/if}

<style>
	.toast-container {
		position: fixed; top: 0.75rem; right: 0.75rem;
		z-index: 9999; display: flex; flex-direction: column;
		gap: 0.4rem; max-width: 360px;
	}

	.toast {
		display: flex; align-items: center; gap: 0.5rem;
		padding: 0.5rem 0.75rem; border-radius: 8px;
		font-size: 0.82rem; color: #fff;
		box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
		animation: slide-in 0.25s ease-out;
	}

	.toast-success { background: #059669; }
	.toast-error   { background: #dc2626; }
	.toast-warning { background: #d97706; }
	.toast-info    { background: #2563eb; }

	.toast-icon {
		width: 20px; height: 20px; border-radius: 50%;
		background: rgba(255,255,255,0.2);
		display: flex; align-items: center; justify-content: center;
		font-size: 0.7rem; font-weight: 700; flex-shrink: 0;
	}

	.toast-msg { flex: 1; line-height: 1.3; }

	.toast-close {
		background: none; border: none; color: rgba(255,255,255,0.7);
		font-size: 1.1rem; cursor: pointer; line-height: 1; padding: 0 0.1rem;
	}
	.toast-close:hover { color: white; }

	@keyframes slide-in {
		from { opacity: 0; transform: translateX(100%); }
		to { opacity: 1; transform: translateX(0); }
	}
</style>
