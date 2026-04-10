<script lang="ts">
	import { onDestroy } from 'svelte';
	import { startOptimization, getOptimizationStatus, getOptimizationResults } from '$lib/api/client';
	import { toasts } from '$lib/stores/toast';

	let destroyed = false;
	onDestroy(() => { destroyed = true; });

	let {
		messageId = '',
		relevantImages = [],
	}: {
		messageId: string;
		relevantImages: string[];
	} = $props();

	let expanded = $state(false);
	let expectedResponse = $state('');
	let iterationCount = $state(3);
	let optimizing = $state(false);
	let progress = $state<{ current: number; total: number } | null>(null);
	let results = $state<any | null>(null);

	async function handleOptimize() {
		if (!expectedResponse.trim() && relevantImages.length === 0) {
			toasts.warning('Provide an expected response or mark relevant images');
			return;
		}

		optimizing = true;
		progress = { current: 0, total: iterationCount };
		results = null;

		try {
			const resp = await startOptimization({
				message_id: messageId,
				iteration_count: iterationCount,
				expected_response: expectedResponse,
				relevant_images: relevantImages,
			});

			if (resp?.optimization_run_id) {
				await pollProgress(resp.optimization_run_id);
			} else {
				toasts.error(resp?.error || 'Optimization failed to start');
				optimizing = false;
				progress = null;
			}
		} catch (e) {
			console.error('Optimization error:', e);
			toasts.error('Optimization request failed');
			optimizing = false;
			progress = null;
		}
	}

	async function pollProgress(runId: string) {
		const maxPolls = 120; // 10 minutes at 5s intervals
		for (let i = 0; i < maxPolls; i++) {
			if (destroyed) return;
			try {
				const status = await getOptimizationStatus(runId);
				const s = status?.status || status?.data?.status;

				if (s === 'completed') {
					// Fetch results
					const res = await getOptimizationResults(runId);
					results = res?.results || res?.data?.results || res;
					progress = null;
					optimizing = false;
					toasts.success('Optimization complete');
					return;
				}

				if (s === 'failed' || s === 'error') {
					toasts.error(status?.error || 'Optimization failed');
					optimizing = false;
					progress = null;
					return;
				}

				// Update progress
				const iteration = status?.current_iteration || status?.data?.current_iteration || 0;
				progress = { current: iteration, total: iterationCount };
			} catch {
				// Network hiccup — continue polling
			}

			await new Promise(r => setTimeout(r, 5000));
		}

		toasts.warning('Optimization timed out');
		optimizing = false;
		progress = null;
	}

	function getScoreColor(score: number): string {
		if (score >= 8) return '#22c55e';
		if (score >= 6) return '#eab308';
		if (score >= 4) return '#f97316';
		return 'var(--danger)';
	}

	function cancel() {
		expanded = false;
		expectedResponse = '';
		results = null;
	}
</script>

<div class="feedback-section">
	<button class="feedback-toggle" onclick={() => (expanded = !expanded)}>
		{expanded ? 'Close' : 'Feedback Loop'}
	</button>

	{#if expanded}
		<div class="feedback-body">
			{#if optimizing && progress}
				<!-- Progress display -->
				<div class="progress-area">
					<div class="progress-header">
						<span>Optimizing prompt...</span>
						<span class="progress-count">{progress.current} / {progress.total}</span>
					</div>
					<div class="progress-bar-bg">
						<div
							class="progress-bar-fill"
							style="width: {progress.total > 0 ? (progress.current / progress.total * 100) : 0}%"
						></div>
					</div>
				</div>
			{:else if results}
				<!-- Results display -->
				<div class="results-area">
					<h4 class="results-title">Optimization Results</h4>
					{#if results.iterations && Array.isArray(results.iterations)}
						{#each results.iterations as iter, i}
							<div class="iteration-card">
								<div class="iter-header">Iteration {i + 1}</div>
								{#if iter.scores}
									<div class="scores-grid">
										{#each Object.entries(iter.scores) as [key, val]}
											<div class="score-item">
												<span class="score-label">{key.replace(/_/g, ' ')}</span>
												<span class="score-val" style="color: {getScoreColor(Number(val))}">{Number(val).toFixed(1)}</span>
											</div>
										{/each}
									</div>
								{/if}
								{#if iter.overall_score != null}
									<div class="overall-score">
										Overall: <strong style="color: {getScoreColor(iter.overall_score)}">{iter.overall_score.toFixed(1)}/10</strong>
									</div>
								{/if}
							</div>
						{/each}
					{:else if results.overall_score != null}
						<div class="overall-score">
							Overall Score: <strong style="color: {getScoreColor(results.overall_score)}">{results.overall_score.toFixed(1)}/10</strong>
						</div>
					{/if}
					{#if results.optimized_prompt}
						<div class="optimized-prompt">
							<label>Optimized Prompt:</label>
							<pre>{results.optimized_prompt}</pre>
						</div>
					{/if}
					<button class="fb-btn secondary" onclick={cancel}>Close</button>
				</div>
			{:else}
				<!-- Input form -->
				<div class="feedback-form">
					<div class="form-field">
						<label>Expected Response:</label>
						<textarea
							bind:value={expectedResponse}
							rows={4}
							placeholder="Describe the ideal response to this query. This will be used as ground truth for evaluation."
						></textarea>
					</div>
					<div class="form-field">
						<label>Optimization Iterations:</label>
						<input type="number" bind:value={iterationCount} min={1} max={10} />
					</div>
					{#if relevantImages.length > 0}
						<div class="relevance-note">
							{relevantImages.length} image{relevantImages.length > 1 ? 's' : ''} marked as relevant
						</div>
					{/if}
					<div class="form-actions">
						<button class="fb-btn secondary" onclick={cancel}>Cancel</button>
						<button class="fb-btn primary" onclick={handleOptimize}>Optimize Prompt</button>
					</div>
				</div>
			{/if}
		</div>
	{/if}
</div>

<style>
	.feedback-section {
		margin-top: 0.5rem;
		border-top: 1px solid var(--bg-hover);
		padding-top: 0.35rem;
	}

	.feedback-toggle {
		background: none; border: 1px solid var(--border); color: var(--accent);
		padding: 3px 10px; border-radius: 12px; font-size: 0.7rem;
		font-weight: 600; cursor: pointer;
	}
	.feedback-toggle:hover { background: #ede9fe; border-color: #a5b4fc; }

	.feedback-body {
		margin-top: 0.5rem; padding: 0.75rem;
		background: #fafafa; border: 1px solid var(--border); border-radius: 8px;
	}

	/* Form */
	.feedback-form { display: flex; flex-direction: column; gap: 0.5rem; }
	.form-field { display: flex; flex-direction: column; gap: 3px; }
	.form-field label { font-size: 0.72rem; font-weight: 600; color: var(--text-secondary); }
	.form-field textarea, .form-field input {
		font-size: 0.82rem; padding: 0.4rem 0.5rem;
		border: 1px solid var(--border); border-radius: 6px;
		font-family: inherit; color: var(--text-heading);
	}
	.form-field textarea { resize: vertical; min-height: 60px; }
	.form-field input[type='number'] { width: 80px; }

	.relevance-note {
		font-size: 0.72rem; color: #059669; font-weight: 600;
		background: #d1fae5; padding: 3px 8px; border-radius: 6px;
		align-self: flex-start;
	}

	.form-actions { display: flex; justify-content: flex-end; gap: 0.5rem; margin-top: 0.25rem; }

	.fb-btn {
		padding: 0.3rem 0.75rem; border-radius: 6px; font-size: 0.78rem;
		font-weight: 600; cursor: pointer; border: none;
	}
	.fb-btn.primary { background: var(--accent); color: white; }
	.fb-btn.primary:hover { background: #4f46e5; }
	.fb-btn.secondary { background: var(--bg-hover); color: var(--text-secondary); }
	.fb-btn.secondary:hover { background: var(--border); }

	/* Progress */
	.progress-area { display: flex; flex-direction: column; gap: 0.5rem; }
	.progress-header {
		display: flex; justify-content: space-between; align-items: center;
		font-size: 0.82rem; color: var(--text-heading); font-weight: 600;
	}
	.progress-count { color: var(--accent); font-variant-numeric: tabular-nums; }
	.progress-bar-bg {
		height: 6px; background: var(--border); border-radius: 3px; overflow: hidden;
	}
	.progress-bar-fill {
		height: 100%; background: var(--accent); border-radius: 3px;
		transition: width 0.5s ease;
	}

	/* Results */
	.results-area { display: flex; flex-direction: column; gap: 0.5rem; }
	.results-title { margin: 0; font-size: 0.85rem; color: var(--text-heading); }

	.iteration-card {
		background: var(--bg-surface); border: 1px solid var(--border); border-radius: 6px;
		padding: 0.5rem 0.65rem;
	}
	.iter-header { font-size: 0.72rem; font-weight: 700; color: var(--accent); margin-bottom: 0.35rem; }

	.scores-grid {
		display: grid; grid-template-columns: 1fr 1fr;
		gap: 0.2rem 1rem;
	}
	.score-item { display: flex; justify-content: space-between; }
	.score-label { font-size: 0.7rem; color: var(--text-secondary); text-transform: capitalize; }
	.score-val { font-size: 0.72rem; font-weight: 700; font-variant-numeric: tabular-nums; }

	.overall-score {
		font-size: 0.82rem; color: var(--text-heading); margin-top: 0.25rem;
		padding-top: 0.25rem; border-top: 1px solid var(--bg-hover);
	}

	.optimized-prompt { margin-top: 0.5rem; }
	.optimized-prompt label { font-size: 0.72rem; font-weight: 600; color: var(--text-secondary); }
	.optimized-prompt pre {
		font-size: 0.75rem; background: var(--bg-surface); border: 1px solid var(--border);
		border-radius: 6px; padding: 0.5rem; white-space: pre-wrap;
		margin: 0.25rem 0 0; max-height: 150px; overflow-y: auto;
	}
</style>
