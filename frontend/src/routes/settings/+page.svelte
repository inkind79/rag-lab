<script lang="ts">
	import { onMount } from 'svelte';
	import { activeSessionId } from '$lib/stores/session';
	import { getSessionData, saveSettings as apiSaveSettings } from '$lib/api/client';

	// Settings state
	let retrieval_method = $state('colpali');
	let generation_model = $state('ollama-gemma3n-vision-fp16');
	let text_embedding_model = $state('BAAI/bge-m3');
	let retrieval_count = $state(1);
	let chunk_size = $state(512);
	let chunk_overlap = $state(64);
	let similarity_threshold = $state(0.2);
	let use_score_slope = $state(true);
	let use_ocr = $state(false);
	let saving = $state(false);
	let saveMessage = $state('');
	let loaded = $state(false);

	const retrievalMethods = [
		{ value: 'colpali', label: 'ColPali (Visual)', description: 'Multi-vector visual retrieval using page images' },
		{ value: 'bm25', label: 'BM25 (Keyword)', description: 'Classic keyword-based retrieval, no GPU needed' },
		{ value: 'dense', label: 'Dense (Bi-Encoder)', description: 'Semantic similarity using dense embeddings' },
		{ value: 'hybrid_rrf', label: 'Hybrid RRF', description: 'Combines ColPali + BM25 with rank fusion' },
	];

	const generationModels = [
		{ value: 'ollama-gemma3n-vision-fp16', label: 'Gemma 3n Vision (FP16)' },
		{ value: 'ollama-gemma-vision', label: 'Gemma 3 Vision (12B Q4)' },
		{ value: 'ollama-qwen3', label: 'Qwen3 30B (Text Only)' },
		{ value: 'ollama-phi4', label: 'Phi-4 Plus (Text Only)' },
		{ value: 'ollama-mistral-small32', label: 'Mistral Small 3.2 (24B)' },
		{ value: 'ollama-magistral', label: 'Magistral 24B (Text Only)' },
		{ value: 'chatgpt', label: 'GPT-4o (Cloud)' },
		{ value: 'gemini', label: 'Gemini (Cloud)' },
	];

	onMount(async () => {
		if ($activeSessionId) {
			await loadSettings($activeSessionId);
		}
	});

	// Reload settings when session changes
	$effect(() => {
		if ($activeSessionId && loaded) {
			loadSettings($activeSessionId);
		}
	});

	async function loadSettings(_sessionId?: string) {
		try {
			const resp = await getSessionData() as Record<string, any>;
			const data = resp?.session_data || resp;
			if (data) {
				retrieval_method = data.retrieval_method || 'colpali';
				generation_model = data.generation_model || 'ollama-gemma3n-vision-fp16';
				text_embedding_model = data.text_embedding_model || 'BAAI/bge-m3';
				retrieval_count = data.retrieval_count || 1;
				chunk_size = data.chunk_size || 512;
				chunk_overlap = data.chunk_overlap || 64;
				similarity_threshold = data.similarity_threshold || 0.2;
				use_score_slope = data.use_score_slope ?? true;
				use_ocr = data.use_ocr ?? false;
				loaded = true;
			}
		} catch (e) {
			console.error('Failed to load settings:', e);
		}
	}

	async function saveSettings() {
		saving = true;
		saveMessage = '';
		try {
			await apiSaveSettings({
				retrieval_method,
				generation_model,
				text_embedding_model,
				retrieval_count,
				chunk_size,
				chunk_overlap,
				similarity_threshold,
				use_score_slope,
				use_ocr,
			});
			saveMessage = 'Settings saved';
			setTimeout(() => (saveMessage = ''), 2000);
		} catch (e) {
			saveMessage = `Error: ${e}`;
		} finally {
			saving = false;
		}
	}

	function needsTextConfig(method: string): boolean {
		return method === 'bm25' || method === 'dense' || method === 'hybrid_rrf';
	}

	function needsDenseModel(method: string): boolean {
		return method === 'dense';
	}
</script>

<div class="settings-page">
	<header class="page-header">
		<h2>Settings</h2>
		<div class="header-actions">
			{#if saveMessage}
				<span class="save-message" class:error={saveMessage.startsWith('Error')}>{saveMessage}</span>
			{/if}
			<button class="save-btn" onclick={saveSettings} disabled={saving}>
				{saving ? 'Saving...' : 'Save Settings'}
			</button>
		</div>
	</header>

	<div class="settings-grid">
		<!-- Retrieval Method -->
		<section class="settings-section">
			<h3>Retrieval Method</h3>
			<p class="section-desc">Choose how documents are indexed and searched</p>

			<div class="method-cards">
				{#each retrievalMethods as method}
					<label class="method-card" class:selected={retrieval_method === method.value}>
						<input type="radio" name="retrieval_method" value={method.value} bind:group={retrieval_method} />
						<div class="method-content">
							<span class="method-label">{method.label}</span>
							<span class="method-desc">{method.description}</span>
						</div>
					</label>
				{/each}
			</div>
		</section>

		<!-- Text Retrieval Config (shown for BM25/dense/hybrid) -->
		{#if needsTextConfig(retrieval_method)}
			<section class="settings-section">
				<h3>Text Chunking</h3>
				<p class="section-desc">Configure how documents are split for text-based retrieval</p>

				<div class="form-row">
					<label class="form-field">
						<span>Chunk Size (chars)</span>
						<input type="number" bind:value={chunk_size} min={64} max={4096} step={64} />
					</label>
					<label class="form-field">
						<span>Chunk Overlap (chars)</span>
						<input type="number" bind:value={chunk_overlap} min={0} max={512} step={16} />
					</label>
				</div>

				{#if needsDenseModel(retrieval_method)}
					<label class="form-field">
						<span>Dense Embedding Model</span>
						<input type="text" bind:value={text_embedding_model} placeholder="BAAI/bge-m3" />
					</label>
				{/if}
			</section>
		{/if}

		<!-- Generation Model -->
		<section class="settings-section">
			<h3>Generation Model</h3>
			<p class="section-desc">LLM used for answering questions</p>

			<select bind:value={generation_model} class="model-select">
				{#each generationModels as model}
					<option value={model.value}>{model.label}</option>
				{/each}
			</select>
		</section>

		<!-- Retrieval Parameters -->
		<section class="settings-section">
			<h3>Retrieval Parameters</h3>

			<div class="form-row">
				<label class="form-field">
					<span>Max Results</span>
					<input type="number" bind:value={retrieval_count} min={1} max={50} />
				</label>
				<label class="form-field">
					<span>Similarity Threshold</span>
					<input type="range" bind:value={similarity_threshold} min={0} max={1} step={0.05} />
					<span class="range-value">{similarity_threshold.toFixed(2)}</span>
				</label>
			</div>

			<div class="form-row">
				<label class="toggle-field">
					<input type="checkbox" bind:checked={use_score_slope} />
					<span>Adaptive Score-Slope Analysis</span>
				</label>
				<label class="toggle-field">
					<input type="checkbox" bind:checked={use_ocr} />
					<span>Enable OCR</span>
				</label>
			</div>
		</section>
	</div>
</div>

<style>
	.settings-page {
		padding: 1.5rem 2rem;
		max-width: 900px;
		overflow-y: auto;
		height: 100%;
	}

	.page-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 2rem;
		padding-bottom: 1rem;
		border-bottom: 1px solid #e5e7eb;
	}

	.page-header h2 {
		margin: 0;
		font-size: 1.4rem;
		color: #1f2937;
	}

	.header-actions {
		display: flex;
		align-items: center;
		gap: 1rem;
	}

	.save-message {
		font-size: 0.85rem;
		color: #22c55e;
	}
	.save-message.error {
		color: #ef4444;
	}

	.save-btn {
		background: #6366f1;
		color: white;
		border: none;
		padding: 0.5rem 1.5rem;
		border-radius: 6px;
		cursor: pointer;
		font-weight: 600;
		font-size: 0.85rem;
	}
	.save-btn:hover:not(:disabled) {
		background: #4f46e5;
	}
	.save-btn:disabled {
		opacity: 0.5;
	}

	.settings-grid {
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
	}

	.settings-section {
		background: #fff;
		border: 1px solid #e5e7eb;
		border-radius: 10px;
		padding: 1.25rem;
	}

	.settings-section h3 {
		margin: 0 0 0.25rem;
		font-size: 1rem;
		color: #1f2937;
	}

	.section-desc {
		margin: 0 0 1rem;
		font-size: 0.8rem;
		color: #9ca3af;
	}

	.method-cards {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.75rem;
	}

	.method-card {
		display: flex;
		align-items: flex-start;
		gap: 0.75rem;
		padding: 0.75rem;
		border: 1px solid #e5e7eb;
		border-radius: 8px;
		cursor: pointer;
		transition: all 0.15s;
	}
	.method-card:hover {
		border-color: #4f46e5;
	}
	.method-card.selected {
		border-color: #6366f1;
		background: rgba(99, 102, 241, 0.08);
	}
	.method-card input[type='radio'] {
		margin-top: 3px;
		accent-color: #6366f1;
	}

	.method-content {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.method-label {
		font-weight: 600;
		font-size: 0.9rem;
		color: #1f2937;
	}
	.method-desc {
		font-size: 0.75rem;
		color: #9ca3af;
	}

	.form-row {
		display: flex;
		gap: 1rem;
		margin-bottom: 0.75rem;
	}

	.form-field {
		display: flex;
		flex-direction: column;
		gap: 4px;
		flex: 1;
	}
	.form-field span {
		font-size: 0.8rem;
		color: #9ca3af;
	}
	.form-field input[type='number'],
	.form-field input[type='text'] {
		background: #fff;
		border: 1px solid #e5e7eb;
		color: #1f2937;
		padding: 0.5rem;
		border-radius: 6px;
		font-size: 0.85rem;
	}
	.form-field input:focus {
		border-color: #6366f1;
		outline: none;
	}

	.form-field input[type='range'] {
		accent-color: #6366f1;
		width: 100%;
	}
	.range-value {
		text-align: right;
		font-variant-numeric: tabular-nums;
	}

	.model-select {
		width: 100%;
		background: #fff;
		border: 1px solid #e5e7eb;
		color: #1f2937;
		padding: 0.5rem;
		border-radius: 6px;
		font-size: 0.85rem;
	}
	.model-select:focus {
		border-color: #6366f1;
		outline: none;
	}

	.toggle-field {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		cursor: pointer;
		flex: 1;
	}
	.toggle-field input[type='checkbox'] {
		accent-color: #6366f1;
		width: 16px;
		height: 16px;
	}
	.toggle-field span {
		font-size: 0.85rem;
		color: #9ca3af;
	}
</style>
