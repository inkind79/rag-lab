<script lang="ts">
	import { activeSessionId } from '$lib/stores/session';
	import { getSessionData, saveSettings, getAvailableModels } from '$lib/api/client';
	import { toasts } from '$lib/stores/toast';

	let { open = false, onclose }: { open: boolean; onclose: () => void } = $props();
	let activeTab = $state<'model' | 'retrieval' | 'advanced'>('model');

	// ── Model tab ──
	let generation_model = $state('ollama-gemma4:latest');
	let temperature = $state(0.7);
	let max_tokens = $state(2048);
	let top_p = $state(0.9);
	let top_k = $state(40);
	let num_ctx = $state(8192);
	let num_predict = $state(2048);
	let repeat_penalty = $state(1.1);

	// ── Retrieval tab ──
	let retrieval_method = $state('colpali');
	let indexer_model = $state('athrael-soju/colqwen3.5-4.5B-v3');
	let retrieval_count = $state(1);
	let chunk_size = $state(512);
	let chunk_overlap = $state(64);
	let hybrid_visual_weight = $state(0.6);
	let similarity_threshold = $state(0.2);
	let use_score_slope = $state(true);
	let rel_drop_threshold = $state(0.65);
	let abs_score_threshold = $state(0.25);
	let distance_metric = $state('cosine');

	// ── Advanced tab ──
	let resized_height = $state(448);
	let resized_width = $state(448);
	let use_ocr = $state(false);
	let ocr_engine = $state('easyocr');
	let save_globally = $state(false);

	// ── Ollama Cloud ──
	let ollamaApiKey = $state('');
	let hasOllamaKey = $state(false);
	let savingKey = $state(false);

	// UI state
	let saving = $state(false);
	let documentsExist = $state(false);
	let loadingModels = $state(false);

	// Dynamic model lists
	interface ModelOption { value: string; label: string }
	let huggingfaceModels = $state<ModelOption[]>([]);
	let ollamaModels = $state<ModelOption[]>([]);

	function getProvider(model: string): 'ollama' | 'huggingface' {
		if (model.startsWith('huggingface')) return 'huggingface';
		return 'ollama';
	}

	let provider = $derived(getProvider(generation_model));
	let isTextOnly = $derived(
		generation_model.includes('phi4') || generation_model.includes('qwq') ||
		generation_model.includes('qwen3') || generation_model.includes('magistral') ||
		generation_model.includes('deepcoder') || generation_model.includes('olmo')
	);

	const allMethods = [
		{ value: 'colpali', label: 'ColPali (Visual)' },
		{ value: 'hybrid', label: 'Hybrid (Visual + Keyword)' },
		{ value: 'bm25', label: 'BM25 (Keyword)' },
		{ value: 'dense', label: 'Dense (Semantic)' },
		{ value: 'hybrid_rrf', label: 'Hybrid RRF' },
	];
	const embeddingModels = [
		{ value: 'athrael-soju/colqwen3.5-4.5B-v3', label: 'ColQwen3.5 4.5B — Best Quality', type: 'visual' as const },
		{ value: 'vidore/colSmol-500M', label: 'ColSmol 500M — Lightweight', type: 'visual' as const },
		{ value: 'tsystems/colqwen2.5-3b-multilingual-v1.0-merged', label: 'ColQwen2.5 3B — Legacy', type: 'visual' as const },
		{ value: 'nomic-ai/colnomic-embed-multimodal-3b', label: 'ColNomic 3B', type: 'visual' as const },
	];
	// Visual models support ColPali and Hybrid; text models support keyword/semantic methods
	let methods = $derived.by(() => {
		const selected = embeddingModels.find(m => m.value === indexer_model);
		if (!selected || selected.type === 'visual') {
			return allMethods.filter(m => m.value === 'colpali' || m.value === 'hybrid');
		}
		return allMethods.filter(m => m.value !== 'colpali' && m.value !== 'hybrid');
	});
	// Auto-correct retrieval_method when model changes
	$effect(() => {
		const valid = methods.some(m => m.value === retrieval_method);
		if (!valid && methods.length > 0) retrieval_method = methods[0].value;
	});
	const distanceMetrics = [
		{ value: 'cosine', label: 'Cosine' },
		{ value: 'l2', label: 'Euclidean (L2)' },
		{ value: 'ip', label: 'Inner Product' },
	];
	const ocrEngines = [
		{ value: 'easyocr', label: 'EasyOCR' },
		{ value: 'smoldocling', label: 'SmolDocling (Markdown)' },
		{ value: 'smoldocling-otsl', label: 'SmolDocling (OTSL)' },
	];

	$effect(() => {
		if (open && $activeSessionId) {
			loadSettings();
			loadModels();
			loadOllamaKeyStatus();
		}
	});

	async function loadOllamaKeyStatus() {
		try {
			const resp = await fetch('/api/v1/settings/ollama-key', { credentials: 'include' });
			if (resp.ok) { const d = await resp.json(); hasOllamaKey = d?.data?.has_key || false; }
		} catch {}
	}

	async function saveOllamaKey() {
		savingKey = true;
		try {
			const resp = await fetch('/api/v1/settings/ollama-key', {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				credentials: 'include',
				body: JSON.stringify({ api_key: ollamaApiKey }),
			});
			if (resp.ok) {
				hasOllamaKey = !!ollamaApiKey.trim();
				ollamaApiKey = '';
				toasts.success(hasOllamaKey ? 'Ollama Cloud key saved' : 'Ollama Cloud key cleared');
				await loadModels();
			}
		} catch {
			toasts.error('Failed to save key');
		} finally {
			savingKey = false;
		}
	}

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
				// Model
				generation_model = data.generation_model || 'ollama-gemma4:latest';
				const p = getProvider(generation_model);
				const mp = data.model_params || {};
				const pp = mp[p] || {};
				temperature = pp.temperature ?? 0.7;
				max_tokens = pp.max_tokens ?? 2048;
				top_p = pp.top_p ?? 0.9;
				top_k = pp.top_k ?? 40;
				num_ctx = pp.num_ctx ?? 8192;
				num_predict = pp.num_predict ?? 2048;
				repeat_penalty = pp.repeat_penalty ?? 1.1;
				// Retrieval
				retrieval_method = data.retrieval_method || 'colpali';
				indexer_model = data.indexer_model || 'athrael-soju/colqwen3.5-4.5B-v3';
				retrieval_count = data.retrieval_count || 1;
				chunk_size = data.chunk_size || 512;
				chunk_overlap = data.chunk_overlap || 64;
				similarity_threshold = data.similarity_threshold ?? 0.2;
				use_score_slope = data.use_score_slope ?? true;
				rel_drop_threshold = data.rel_drop_threshold ?? 0.65;
				abs_score_threshold = data.abs_score_threshold ?? 0.25;
				distance_metric = data.distance_metric || 'cosine';
				hybrid_visual_weight = data.hybrid_visual_weight ?? 0.6;
				// Advanced
				resized_height = data.resized_height || 448;
				resized_width = data.resized_width || 448;
				use_ocr = data.use_ocr ?? false;
				ocr_engine = data.ocr_engine || 'easyocr';
				documentsExist = !!(data.indexed_files?.length);
			}
		} catch { /* ignore */ }
	}

	async function save() {
		saving = true;
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
				retrieval_method, indexer_model, retrieval_count,
				chunk_size, chunk_overlap, similarity_threshold, hybrid_visual_weight,
				use_score_slope, rel_drop_threshold, abs_score_threshold,
				distance_metric, resized_height, resized_width,
				use_ocr, ocr_engine, save_globally,
			});
			toasts.success('Settings saved');
			onclose();
		} catch (e) {
			console.error('Save settings failed:', e);
			toasts.error('Failed to save settings');
		} finally {
			saving = false;
		}
	}
</script>

{#if open}
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
	<div class="backdrop" onclick={(e) => { if ((e.target as HTMLElement).classList.contains('backdrop')) onclose(); }} onkeydown={(e) => { if (e.key === 'Escape') onclose(); }} role="dialog" tabindex="-1">
		<div class="modal">
			<div class="modal-header">
				<h2>Settings</h2>
				<button class="close-btn" onclick={onclose}>&times;</button>
			</div>

			<!-- Tabs -->
			<div class="tabs">
				<button class="tab" class:active={activeTab === 'model'} onclick={() => (activeTab = 'model')}>Model</button>
				<button class="tab" class:active={activeTab === 'retrieval'} onclick={() => (activeTab = 'retrieval')}>Retrieval</button>
				<button class="tab" class:active={activeTab === 'advanced'} onclick={() => (activeTab = 'advanced')}>Advanced</button>
			</div>

			<div class="modal-body">
				<!-- ═══ MODEL TAB ═══ -->
				{#if activeTab === 'model'}
					<div class="group">
						<label class="label">Generation Model</label>
						<select bind:value={generation_model} class="full">
							{#if huggingfaceModels.length > 0}
								<optgroup label="HuggingFace (Local)">
									{#each huggingfaceModels as m}<option value={m.value}>{m.label}</option>{/each}
								</optgroup>
							{/if}
							{#if ollamaModels.length > 0}
								<optgroup label="Ollama (Installed)">
									{#each ollamaModels as m}<option value={m.value}>{m.label}</option>{/each}
								</optgroup>
							{:else if loadingModels}
								<optgroup label="Ollama"><option disabled>Loading...</option></optgroup>
							{/if}
						</select>
						{#if isTextOnly}
							<span class="hint warn">Text-only model — OCR will be auto-enabled.</span>
						{/if}
					</div>

					<div class="group">
						<label class="label">Model Parameters</label>

						<div class="param">
							<span class="param-name">Temperature</span>
							<div class="slider-row">
								<input type="range" bind:value={temperature} min={0} max={1} step={0.05} />
								<span class="param-val">{temperature.toFixed(2)}</span>
							</div>
						</div>

						{#if provider === 'ollama'}
							<div class="param">
								<span class="param-name">Max Output Tokens</span>
								<input type="number" bind:value={num_predict} min={256} max={32768} step={256} />
							</div>
						{:else}
							<div class="param">
								<span class="param-name">Max Tokens</span>
								<input type="number" bind:value={max_tokens} min={100} max={65536} step={100} />
							</div>
						{/if}

						<div class="param">
							<span class="param-name">Top P</span>
							<div class="slider-row">
								<input type="range" bind:value={top_p} min={0} max={1} step={0.05} />
								<span class="param-val">{top_p.toFixed(2)}</span>
							</div>
						</div>

						{#if provider === 'ollama'}
							<div class="param">
								<span class="param-name">Top K</span>
								<input type="number" bind:value={top_k} min={1} max={100} />
							</div>
						{/if}

						{#if provider === 'ollama'}
							<div class="param">
								<span class="param-name">Context Window</span>
								<input type="number" bind:value={num_ctx} min={1024} max={32768} step={1024} />
							</div>
							<div class="param">
								<span class="param-name">Repeat Penalty</span>
								<div class="slider-row">
									<input type="range" bind:value={repeat_penalty} min={0.1} max={2.0} step={0.1} />
									<span class="param-val">{repeat_penalty.toFixed(1)}</span>
								</div>
							</div>
						{/if}
					</div>

				<!-- ═══ RETRIEVAL TAB ═══ -->
				{:else if activeTab === 'retrieval'}
					<div class="group">
						<label class="label">Embedding Model</label>
						<select bind:value={indexer_model} class="full">
							{#each embeddingModels as m}<option value={m.value}>{m.label}</option>{/each}
						</select>
						<span class="hint">Model used for document indexing and visual retrieval.</span>
					</div>

					<div class="group">
						<label class="label">Retrieval Method</label>
						<select bind:value={retrieval_method} class="full">
							{#each methods as m}<option value={m.value}>{m.label}</option>{/each}
						</select>
						{#if retrieval_method === 'hybrid'}
							<div class="param" style="margin-top: 0.5rem;">
								<span class="param-name">Visual / Keyword Balance</span>
								<div style="display: flex; align-items: center; gap: 0.5rem;">
									<span style="font-size: 0.7rem; color: var(--text-muted); white-space: nowrap;">Keyword</span>
									<input type="range" bind:value={hybrid_visual_weight} min={0} max={1} step={0.05} style="flex: 1;" />
									<span style="font-size: 0.7rem; color: var(--text-muted); white-space: nowrap;">Visual</span>
								</div>
								<span class="hint" style="text-align: center; display: block;">Visual {(hybrid_visual_weight * 100).toFixed(0)}% — Keyword {((1 - hybrid_visual_weight) * 100).toFixed(0)}%</span>
							</div>
						{/if}
						{#if retrieval_method !== 'colpali' && retrieval_method !== 'hybrid'}
							<div class="row" style="margin-top: 0.5rem;">
								<div class="field">
									<span class="param-name">Chunk Size</span>
									<input type="number" bind:value={chunk_size} min={64} max={4096} step={64} />
								</div>
								<div class="field">
									<span class="param-name">Chunk Overlap</span>
									<input type="number" bind:value={chunk_overlap} min={0} max={512} step={16} />
								</div>
							</div>
						{/if}
					</div>

					<div class="group">
						<label class="label">Search Parameters</label>
						<div class="row">
							<div class="field">
								<span class="param-name">Max Results</span>
								<input type="number" bind:value={retrieval_count} min={1} max={50} />
							</div>
							<div class="field">
								<span class="param-name">Distance Metric</span>
								<select bind:value={distance_metric} disabled={documentsExist}>
									{#each distanceMetrics as m}<option value={m.value}>{m.label}</option>{/each}
								</select>
								{#if documentsExist}<span class="hint warn">Locked after upload.</span>{/if}
							</div>
						</div>
						<div class="param" style="margin-top: 0.5rem;">
							<span class="param-name">Similarity Threshold</span>
							<div class="slider-row">
								<input type="range" bind:value={similarity_threshold} min={0} max={1} step={0.05} />
								<span class="param-val">{similarity_threshold.toFixed(2)}</span>
							</div>
						</div>
					</div>

					<div class="group">
						<label class="label">Adaptive Score-Slope</label>
						<label class="toggle">
							<input type="checkbox" bind:checked={use_score_slope} />
							<span>Enable adaptive analysis</span>
						</label>
						{#if use_score_slope}
							<div class="row" style="margin-top: 0.5rem;">
								<div class="field">
									<span class="param-name">Relative Drop</span>
									<div class="slider-row">
										<input type="range" bind:value={rel_drop_threshold} min={0.4} max={0.9} step={0.05} />
										<span class="param-val">{rel_drop_threshold.toFixed(2)}</span>
									</div>
								</div>
								<div class="field">
									<span class="param-name">Absolute Score</span>
									<div class="slider-row">
										<input type="range" bind:value={abs_score_threshold} min={0.1} max={0.5} step={0.05} />
										<span class="param-val">{abs_score_threshold.toFixed(2)}</span>
									</div>
								</div>
							</div>
						{/if}
						<span class="hint">Dynamically determines how many pages to retrieve based on relevance drop-off.</span>
					</div>

				<!-- ═══ ADVANCED TAB ═══ -->
				{:else}
					<div class="group">
						<label class="label">Ollama Cloud</label>
						<span class="hint">Connect to Ollama's cloud models with your API key from <a href="https://ollama.com/settings/keys" target="_blank" rel="noopener">ollama.com/settings/keys</a></span>
						<div class="key-row">
							<input type="password" bind:value={ollamaApiKey}
								placeholder={hasOllamaKey ? 'Key configured — enter new to replace' : 'Paste API key'}
								class="full" />
							<button class="save-key" onclick={saveOllamaKey} disabled={savingKey}>
								{savingKey ? '...' : ollamaApiKey ? 'Save' : hasOllamaKey ? 'Clear' : 'Save'}
							</button>
						</div>
					</div>

					<div class="group">
						<label class="label">OCR Text Extraction</label>
						<label class="toggle">
							<input type="checkbox" bind:checked={use_ocr} />
							<span>Enable OCR</span>
						</label>
						{#if use_ocr}
							<div style="margin-top: 0.5rem;">
								<span class="param-name">OCR Engine</span>
								<select bind:value={ocr_engine} class="full">
									{#each ocrEngines as e}<option value={e.value}>{e.label}</option>{/each}
								</select>
							</div>
						{/if}
						<span class="hint">OCR is auto-enabled for text-only models.</span>
					</div>

					<div class="group">
						<label class="toggle">
							<input type="checkbox" bind:checked={save_globally} />
							<span>Save as my defaults for new sessions</span>
						</label>
					</div>
				{/if}
			</div>

			<div class="modal-footer">
				<button class="btn secondary" onclick={onclose}>Cancel</button>
				<button class="btn primary" onclick={save} disabled={saving}>
					{saving ? 'Saving...' : 'Save'}
				</button>
			</div>
		</div>
	</div>
{/if}

<style>
	.backdrop {
		position: fixed; inset: 0; background: rgba(0,0,0,0.4);
		display: flex; align-items: center; justify-content: center; z-index: 1000;
		backdrop-filter: blur(2px); animation: fadeIn 0.15s ease;
	}
	@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

	.modal {
		background: var(--bg-surface); border-radius: 14px; border: 1px solid var(--border);
		width: 520px; max-height: 85vh; display: flex; flex-direction: column;
		box-shadow: 0 20px 60px var(--shadow-lg);
		animation: scaleIn 0.15s ease;
	}
	@keyframes scaleIn { from { transform: scale(0.96); opacity: 0; } to { transform: scale(1); opacity: 1; } }

	.modal-header {
		display: flex; justify-content: space-between; align-items: center;
		padding: 0.85rem 1.1rem; border-bottom: 1px solid var(--border);
	}
	.modal-header h2 { margin: 0; font-size: 1rem; font-weight: 600; color: var(--text-heading); }
	.close-btn { background: none; border: none; font-size: 1.4rem; color: var(--text-muted); cursor: pointer; line-height: 1; }
	.close-btn:hover { color: var(--text-heading); }

	/* Tabs */
	.tabs {
		display: flex; border-bottom: 1px solid var(--border);
		padding: 0 1.1rem; gap: 0;
	}
	.tab {
		background: none; border: none; border-bottom: 2px solid transparent;
		padding: 0.6rem 0.9rem; font-size: 0.82rem; font-weight: 600;
		color: var(--text-muted); cursor: pointer; font-family: var(--font-sans);
		transition: all 0.12s;
	}
	.tab:hover { color: var(--text-secondary); }
	.tab.active { color: var(--text-heading); border-bottom-color: var(--text-heading); }

	.modal-body { padding: 0.85rem 1.1rem; overflow-y: auto; flex: 1; }

	/* Groups */
	.group { margin-bottom: 1rem; }
	.group:last-child { margin-bottom: 0; }
	.key-row { display: flex; gap: 0.35rem; margin-top: 0.35rem; }
	.key-row input { flex: 1; }
	.save-key {
		padding: 0.3rem 0.7rem; border-radius: 6px; border: 1px solid var(--border);
		background: var(--bg-active); color: var(--text-heading); font-size: 0.75rem;
		font-weight: 600; cursor: pointer; white-space: nowrap;
	}
	.save-key:hover { background: var(--bg-hover); }
	.save-key:disabled { opacity: 0.5; }
	.label {
		display: block; font-size: 0.72rem; font-weight: 700; color: var(--text-muted);
		text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.45rem;
	}

	.row { display: flex; gap: 0.75rem; }
	.field { flex: 1; display: flex; flex-direction: column; gap: 3px; }

	.param { margin-bottom: 0.5rem; }
	.param:last-child { margin-bottom: 0; }
	.param-name { font-size: 0.78rem; font-weight: 500; color: var(--text-heading); display: block; margin-bottom: 2px; }
	.param-val {
		font-size: 0.75rem; font-weight: 600; color: var(--text-heading);
		font-family: var(--font-mono); min-width: 2.5em; text-align: right;
	}

	select, input[type='number'] {
		width: 100%; padding: 0.4rem 0.55rem;
		border: 1px solid var(--border); border-radius: 8px;
		font-size: 0.82rem; color: var(--text-primary); background: var(--bg-input);
		font-family: var(--font-sans); outline: none;
		transition: border-color 0.12s;
	}
	select.full { width: 100%; }
	select:focus, input:focus { border-color: var(--text-muted); }
	select:disabled { opacity: 0.5; }

	input[type='range'] { flex: 1; accent-color: var(--text-heading); }
	.slider-row { display: flex; align-items: center; gap: 0.5rem; }

	.hint { display: block; font-size: 0.68rem; color: var(--text-muted); margin-top: 0.2rem; }
	.hint.warn { color: #d97706; }

	.toggle { display: flex; align-items: center; gap: 0.5rem; cursor: pointer; }
	.toggle input { accent-color: var(--text-heading); width: 15px; height: 15px; }
	.toggle span { font-size: 0.82rem; color: var(--text-heading); }

	.modal-footer {
		display: flex; justify-content: flex-end; gap: 0.5rem;
		padding: 0.65rem 1.1rem; border-top: 1px solid var(--border);
	}
	.btn {
		padding: 0.4rem 1rem; border-radius: 8px; font-size: 0.82rem;
		font-weight: 600; cursor: pointer; border: none; font-family: var(--font-sans);
		transition: all 0.1s;
	}
	.btn.primary { background: var(--text-heading); color: var(--bg-surface); }
	.btn.primary:hover:not(:disabled) { opacity: 0.85; }
	.btn.primary:disabled { opacity: 0.4; }
	.btn.secondary { background: var(--bg-hover); color: var(--text-secondary); }
	.btn.secondary:hover { background: var(--bg-active); }

	@media (max-width: 768px) {
		.modal { width: 95vw; }
		.row { flex-direction: column; gap: 0.5rem; }
	}
</style>
