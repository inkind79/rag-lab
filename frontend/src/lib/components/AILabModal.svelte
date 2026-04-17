<script lang="ts">
	import { activeSessionId } from '$lib/stores/session';
	import { getSessionData, saveSettings, getAvailableModels } from '$lib/api/client';
	import { toasts } from '$lib/stores/toast';

	let { open = false, onclose }: { open: boolean; onclose: () => void } = $props();

	let generation_model = $state('ollama-gemma3n-vision-fp16');
	let saving = $state(false);
	let loadingModels = $state(false);

	// Per-provider params
	let temperature = $state(0.7);
	let max_tokens = $state(2048);
	let top_p = $state(0.9);
	let top_k = $state(40);
	let num_ctx = $state(8192);
	let num_predict = $state(2048);
	let repeat_penalty = $state(1.1);

	// Dynamic model list — fetched from backend
	interface ModelOption { value: string; label: string; provider?: string }
	let huggingfaceModels = $state<ModelOption[]>([]);
	let ollamaModels = $state<ModelOption[]>([]);

	function getProvider(model: string): 'ollama' | 'huggingface' {
		if (model.startsWith('huggingface')) return 'huggingface';
		return 'ollama';
	}

	$effect(() => {
		if (open && $activeSessionId) {
			loadSettings();
			loadModels();
		}
	});

	async function loadModels() {
		loadingModels = true;
		try {
			const data = await getAvailableModels();
			if (data.huggingface?.length) huggingfaceModels = data.huggingface;
			if (data.ollama?.length) ollamaModels = data.ollama;
		} catch { /* keep fallback */ }
		loadingModels = false;
	}

	async function loadSettings() {
		try {
			const resp = await getSessionData() as Record<string, any>;
			const data = resp?.session_data || resp;
			if (data) {
				generation_model = data.generation_model || 'ollama-gemma3n-vision-fp16';
				const provider = getProvider(generation_model);
				const mp = data.model_params || {};
				const providerParams = mp[provider] || {};
				temperature = providerParams.temperature ?? 0.7;
				max_tokens = providerParams.max_tokens ?? 2048;
				top_p = providerParams.top_p ?? 0.9;
				top_k = providerParams.top_k ?? 40;
				num_ctx = providerParams.num_ctx ?? 8192;
				num_predict = providerParams.num_predict ?? 2048;
				repeat_penalty = providerParams.repeat_penalty ?? 1.1;
			}
		} catch { /* ignore */ }
	}

	async function save() {
		saving = true;
		const provider = getProvider(generation_model);
		const params: Record<string, any> = { temperature, max_tokens, top_p };

		if (provider === 'ollama') {
			params.top_k = top_k;
			params.num_ctx = num_ctx;
			params.num_predict = num_predict;
			params.repeat_penalty = repeat_penalty;
		}

		try {
			await saveSettings({
				generation_model,
				model_params: { [provider]: params },
			});
			toasts.success('Model settings applied');
			onclose();
		} catch (e) {
			console.error('Save model settings failed:', e);
			toasts.error('Failed to apply model settings');
		} finally {
			saving = false;
		}
	}

	function handleBackdrop(e: MouseEvent) {
		if ((e.target as HTMLElement).classList.contains('modal-backdrop')) onclose();
	}

	let provider = $derived(getProvider(generation_model));
	let isTextOnly = $derived(generation_model.includes('phi4') || generation_model.includes('qwq') || generation_model.includes('qwen3') || generation_model.includes('magistral') || generation_model.includes('deepcoder') || generation_model.includes('olmo'));
</script>

{#if open}
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
	<div class="modal-backdrop" onclick={handleBackdrop} onkeydown={(e) => { if (e.key === 'Escape') onclose(); }} role="dialog" tabindex="-1">
		<div class="modal">
			<div class="modal-header">
				<h2>AI LAB</h2>
				<button class="close-btn" onclick={onclose}>&times;</button>
			</div>

			<div class="modal-body">
				<div class="two-col">
					<!-- Left: model selection -->
					<div class="col-left">
						<span class="field-label">Generation Model</span>
						<select bind:value={generation_model} class="model-select">
							{#if huggingfaceModels.length > 0}
								<optgroup label="HuggingFace (Local)">
									{#each huggingfaceModels as m}
										<option value={m.value}>{m.label}</option>
									{/each}
								</optgroup>
							{/if}
							{#if ollamaModels.length > 0}
								<optgroup label="Ollama (Installed)">
									{#each ollamaModels as m}
										<option value={m.value}>{m.label}</option>
									{/each}
								</optgroup>
							{:else if loadingModels}
								<optgroup label="Ollama">
									<option disabled>Loading models...</option>
								</optgroup>
							{:else}
								<optgroup label="Ollama">
									<option disabled>No Ollama models detected</option>
								</optgroup>
							{/if}
						</select>
						{#if isTextOnly}
							<div class="text-only-note">OCR will be auto-enabled for this text-only model.</div>
						{/if}
					</div>

					<!-- Right: provider config -->
					<div class="col-right">
						<div class="config-title">Model Parameters</div>

						<!-- Temperature (all providers) -->
						<div class="param">
							<span class="param-label">Temperature</span>
							<div class="slider-row">
								<input type="range" bind:value={temperature} min={0} max={1} step={0.05} />
								<span class="slider-val">{temperature.toFixed(2)}</span>
							</div>
							<span class="param-hint">Higher = more creative, lower = more deterministic</span>
						</div>

						<!-- Max Tokens (all providers) -->
						<div class="param">
							<span class="param-label">Max Tokens</span>
							{#if provider === 'ollama'}
								<input type="number" bind:value={num_predict} min={256} max={32768} step={256} />
								<span class="param-hint">Max output tokens</span>
							{:else}
								<input type="number" bind:value={max_tokens} min={100} max={65536} step={100} />
							{/if}
						</div>

						<!-- Top P (all providers) -->
						<div class="param">
							<span class="param-label">Top P</span>
							<div class="slider-row">
								<input type="range" bind:value={top_p} min={0} max={1} step={0.05} />
								<span class="slider-val">{top_p.toFixed(2)}</span>
							</div>
						</div>

						<!-- Top K (Ollama) -->
						{#if provider === 'ollama'}
							<div class="param">
								<span class="param-label">Top K</span>
								<input type="number" bind:value={top_k} min={1} max={100} />
							</div>
						{/if}

						<!-- Ollama-specific -->
						{#if provider === 'ollama'}
							<div class="param">
								<span class="param-label">Context Window</span>
								<input type="number" bind:value={num_ctx} min={1024} max={32768} step={1024} />
								<span class="param-hint">Max context tokens (default 8192)</span>
							</div>
							<div class="param">
								<span class="param-label">Repeat Penalty</span>
								<div class="slider-row">
									<input type="range" bind:value={repeat_penalty} min={0.1} max={2.0} step={0.1} />
									<span class="slider-val">{repeat_penalty.toFixed(1)}</span>
								</div>
							</div>
						{/if}
					</div>
				</div>
			</div>

			<div class="modal-footer">
				<button class="btn secondary" onclick={onclose}>Cancel</button>
				<button class="btn primary" onclick={save} disabled={saving}>
					{saving ? 'Applying...' : 'Apply Model Changes'}
				</button>
			</div>
		</div>
	</div>
{/if}

<style>
	.modal-backdrop {
		position: fixed; inset: 0; background: rgba(0,0,0,0.4);
		display: flex; align-items: center; justify-content: center; z-index: 1000;
	}
	.modal {
		background: var(--bg-card, #fff); border-radius: 12px; width: 640px;
		max-height: 85vh; overflow-y: auto;
		box-shadow: 0 20px 60px rgba(0,0,0,0.15);
	}
	.modal-header {
		display: flex; justify-content: space-between; align-items: center;
		padding: 1rem 1.25rem; border-bottom: 1px solid var(--border, #e5e7eb);
	}
	.modal-header h2 { margin: 0; font-size: 1.1rem; color: var(--text-heading, #1f2937); letter-spacing: 0.1em; }
	.close-btn {
		background: none; border: none; font-size: 1.5rem;
		color: var(--text-muted, #9ca3af); cursor: pointer; line-height: 1;
	}
	.modal-body { padding: 1.25rem; }

	.two-col { display: flex; gap: 1.25rem; }
	.col-left { flex: 1; min-width: 0; }
	.col-right { flex: 1; min-width: 0; }

	.field-label { font-size: 0.78rem; font-weight: 600; color: var(--text-secondary, #374151); display: block; margin-bottom: 0.3rem; }

	.model-select {
		width: 100%; padding: 0.4rem 0.5rem; border: 1px solid var(--border, #d1d5db);
		border-radius: 6px; font-size: 0.82rem;
		color: var(--text-primary, #1f2937); background: var(--bg-input, #fff);
	}
	.model-select:focus { border-color: var(--accent, #6366f1); outline: none; box-shadow: 0 0 0 2px rgba(99,102,241,0.1); }

	.text-only-note {
		margin-top: 0.5rem; font-size: 0.72rem; color: #d97706;
		background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px;
		padding: 0.35rem 0.5rem;
	}

	.config-title {
		font-size: 0.78rem; font-weight: 700; color: var(--accent, #6366f1);
		text-transform: uppercase; letter-spacing: 0.04em;
		margin-bottom: 0.6rem;
	}

	.param { margin-bottom: 0.6rem; }
	.param label { font-size: 0.75rem; font-weight: 600; color: var(--text-secondary, #374151); display: block; margin-bottom: 2px; }
	.param input[type='number'] {
		width: 100%; padding: 0.35rem 0.45rem; border: 1px solid var(--border, #d1d5db);
		border-radius: 5px; font-size: 0.82rem;
		color: var(--text-primary, #1f2937); background: var(--bg-input, #fff);
	}
	.param input:focus { border-color: var(--accent, #6366f1); outline: none; }
	.param-hint { font-size: 0.65rem; color: var(--text-muted, #9ca3af); display: block; margin-top: 1px; }

	.slider-row { display: flex; align-items: center; gap: 0.4rem; }
	.slider-row input[type='range'] { flex: 1; accent-color: var(--accent, #6366f1); }
	.slider-val {
		font-size: 0.75rem; font-weight: 600; color: var(--text-secondary, #374151);
		font-variant-numeric: tabular-nums; min-width: 2.2em; text-align: center;
	}

	.modal-footer {
		display: flex; justify-content: flex-end; gap: 0.5rem;
		padding: 0.75rem 1.25rem; border-top: 1px solid var(--border, #e5e7eb);
	}
	.btn {
		padding: 0.4rem 1rem; border-radius: 6px; font-size: 0.85rem;
		font-weight: 600; cursor: pointer; border: none;
	}
	.btn.primary { background: var(--accent, #6366f1); color: white; }
	.btn.primary:hover:not(:disabled) { background: var(--accent-hover, #4f46e5); }
	.btn.primary:disabled { opacity: 0.5; }
	.btn.secondary { background: var(--bg-hover, #f3f4f6); color: var(--text-secondary, #374151); }
	.btn.secondary:hover { background: var(--bg-active, #e5e7eb); }
</style>
