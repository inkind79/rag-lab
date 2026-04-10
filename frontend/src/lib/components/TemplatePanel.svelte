<script lang="ts">
	import { activeSessionId, optimizedQuery } from '$lib/stores/session';
	import {
		getTemplates, selectTemplate, getSessionId, getSessionData,
		createTemplate as apiCreateTemplate,
		updateTemplate as apiUpdateTemplate,
		deleteTemplate as apiDeleteTemplate,
		setDefaultTemplate as apiSetDefault,
		getOptimizationResults,
		generateWithModel,
	} from '$lib/api/client';
	import { toasts } from '$lib/stores/toast';

	interface Template {
		id: string;
		name: string;
		template_type?: string;
		description?: string;
		system_prompt?: string;
		query_prefix?: string;
		query_suffix?: string;
		is_default?: boolean;
		optimized_query?: string;
		optimization_run_id?: string;
	}

	// Optimization results modal state
	let optResultsModal = $state<{ open: boolean; loading: boolean; data: any | null }>({ open: false, loading: false, data: null });

	async function viewOptimizationResults(runId: string) {
		optResultsModal = { open: true, loading: true, data: null };
		try {
			const resp = await getOptimizationResults(runId);
			optResultsModal.data = resp?.results || resp;
			optResultsModal.loading = false;
		} catch {
			toasts.error('Failed to load optimization results');
			optResultsModal = { open: false, loading: false, data: null };
		}
	}

	function scoreColor(score: number): string {
		if (score >= 8) return '#22c55e';
		if (score >= 6) return '#eab308';
		if (score >= 4) return '#f97316';
		return 'var(--danger)';
	}

	function formatDate(dateStr: string | null): string {
		if (!dateStr) return '';
		try { return new Date(dateStr).toLocaleDateString(); } catch { return dateStr; }
	}

	let templates = $state<Template[]>([]);
	let selectedId = $state<string>('');
	let expandedId = $state<string>('');
	let collapsed = $state(typeof window !== 'undefined' ? localStorage.getItem('panel_workflow_collapsed') === 'true' : false);

	$effect(() => { if (typeof window !== 'undefined') localStorage.setItem('panel_workflow_collapsed', String(collapsed)); });
	let savingId = $state<string | null>(null);
	let deleteConfirmId = $state<string | null>(null);

	// Editing state — track edits per template
	let editName = $state('');
	let editDescription = $state('');
	let editSystemPrompt = $state('');
	let editQueryPrefix = $state('');
	let editQuerySuffix = $state('');

	$effect(() => {
		if ($activeSessionId) loadTemplates();
	});

	async function loadTemplates() {
		try {
			const data = await getTemplates();
			templates = data as Template[];
			try {
				const resp = await getSessionData();
				const sessionData = resp?.session_data || resp;
				if (sessionData?.selected_template_id) {
					selectedId = sessionData.selected_template_id;
				} else if (templates.length > 0) {
					selectedId = templates[0].id;
				}
				const activeTpl = templates.find(t => t.id === selectedId);
				// selectedTemplateType removed — batch mode is now client-driven
void(activeTpl?.template_type || 'general');
				optimizedQuery.set(activeTpl?.optimized_query || '');
			} catch {}
		} catch {
			templates = [];
		}
	}

	async function handleSelect(templateId: string) {
		const sessionId = getSessionId();
		if (!sessionId) return;
		selectedId = templateId;
		const tpl = templates.find(t => t.id === templateId);
		// selectedTemplateType removed — batch mode is now client-driven
void(tpl?.template_type || 'general');
		optimizedQuery.set(tpl?.optimized_query || '');
		try {
			await selectTemplate(sessionId, templateId);
		} catch (e) {
			console.error('Failed to select template:', e);
		}
	}

	let discardConfirmTarget = $state<string | null>(null);

	function hasUnsavedEdits(): boolean {
		if (!expandedId) return false;
		const tpl = templates.find(t => t.id === expandedId);
		if (!tpl || tpl.is_default) return false;
		return (
			editName !== (tpl.name || '') ||
			editDescription !== (tpl.description || '') ||
			editSystemPrompt !== (tpl.system_prompt || '') ||
			editQueryPrefix !== (tpl.query_prefix || '') ||
			editQuerySuffix !== (tpl.query_suffix || '')
		);
	}

	function populateEditFields(tpl: Template) {
		editName = tpl.name || '';
		editDescription = tpl.description || '';
		editSystemPrompt = tpl.system_prompt || '';
		editQueryPrefix = tpl.query_prefix || '';
		editQuerySuffix = tpl.query_suffix || '';
	}

	function toggleExpand(id: string) {
		if (expandedId === id) {
			expandedId = '';
		} else {
			// Check for unsaved edits before switching
			if (hasUnsavedEdits()) {
				discardConfirmTarget = id;
				return;
			}
			expandedId = id;
			const tpl = templates.find(t => t.id === id);
			if (tpl) populateEditFields(tpl);
		}
	}

	function confirmDiscardAndExpand() {
		const id = discardConfirmTarget;
		discardConfirmTarget = null;
		if (!id) return;
		expandedId = id;
		const tpl = templates.find(t => t.id === id);
		if (tpl) populateEditFields(tpl);
	}

	async function handleSave(tpl: Template) {
		savingId = tpl.id;
		try {
			await apiUpdateTemplate(tpl.id, {
				name: editName,
				template_type: 'general',
				description: editDescription,
				system_prompt: editSystemPrompt,
				query_prefix: editQueryPrefix,
				query_suffix: editQuerySuffix,
			});
			templates = templates.map(t =>
				t.id === tpl.id
					? { ...t, name: editName, description: editDescription, system_prompt: editSystemPrompt, query_prefix: editQueryPrefix, query_suffix: editQuerySuffix }
					: t
			);
			toasts.success('Template saved');
		} catch (e) {
			console.error('Save template failed:', e);
			toasts.error('Failed to save template');
		} finally {
			savingId = null;
		}
	}

	async function handleDelete(templateId: string) {
		deleteConfirmId = null;
		try {
			await apiDeleteTemplate(templateId);
			templates = templates.filter(t => t.id !== templateId);
			if (expandedId === templateId) expandedId = '';
			if (selectedId === templateId && templates.length > 0) {
				selectedId = templates[0].id;
				const sessionId = getSessionId();
				if (sessionId) await selectTemplate(sessionId, selectedId);
			}
			toasts.success('Template deleted');
		} catch (e) {
			console.error('Delete template failed:', e);
			toasts.error('Failed to delete template');
		}
	}

	async function handleSetDefault(templateId: string) {
		try {
			await apiSetDefault(templateId);
			templates = templates.map(t => ({ ...t, is_default: t.id === templateId }));
			toasts.success('Default template updated');
		} catch {
			toasts.error('Failed to set default');
		}
	}

	// AI template generation
	let showCreateForm = $state(false);
	let createDescription = $state('');
	let generating = $state(false);

	async function handleGenerate() {
		if (!createDescription.trim()) return;
		generating = true;
		try {
			// Get current model from session
			let model = 'ollama-gemma4:latest';
			try {
				const resp = await getSessionData();
				const data = resp?.session_data || resp;
				if (data?.generation_model) model = data.generation_model;
			} catch {}

			const result = await generateWithModel(
				`Generate a prompt template for this task: ${createDescription.trim()}\n\nCreate a concise system_prompt (100-150 words) with:\n- A clear AI role definition\n- 4-5 specific guidelines\n- Anti-hallucination reminder\n\nAlso provide a short name and description.`,
				model,
				{
					type: 'object',
					properties: {
						name: { type: 'string', description: 'Short descriptive title (3-6 words)' },
						description: { type: 'string', description: 'One sentence description' },
						system_prompt: { type: 'string', description: 'System prompt with role, guidelines, and anti-hallucination reminder (100-150 words)' },
						query_prefix: { type: 'string', description: 'Short phrase before user query (5-10 words)' },
						query_suffix: { type: 'string', description: 'Short instruction after user query (10-15 words)' },
					},
					required: ['name', 'system_prompt'],
				}
			);

			const responseText = result?.response || '';
			const templateData = JSON.parse(responseText);

			const created = await apiCreateTemplate({
				name: templateData.name || 'AI Template',
				description: templateData.description || createDescription.trim(),
				system_prompt: templateData.system_prompt || '',
				query_prefix: templateData.query_prefix || '',
				query_suffix: templateData.query_suffix || '',
			});

			if (created?.template_id) {
				await loadTemplates();
				expandedId = created.template_id;
				const tpl = templates.find(t => t.id === created.template_id);
				if (tpl) populateEditFields(tpl);
				toasts.success(`Template "${templateData.name}" created`);
			}
		} catch (e) {
			console.error('Template generation failed:', e);
			toasts.error('Failed to generate template');
		} finally {
			generating = false;
			showCreateForm = false;
			createDescription = '';
		}
	}
</script>

<div class="panel-section">
	<button class="section-header" onclick={() => (collapsed = !collapsed)}>
		<span class="section-title">Prompt Templates</span>
		<div class="section-actions">
			{#if !collapsed}
				<button class="add-btn" onclick={(e) => { e.stopPropagation(); showCreateForm = !showCreateForm; }} title="New template">+</button>
			{/if}
			<span class="collapse-icon">{collapsed ? '▸' : '▾'}</span>
		</div>
	</button>

	{#if !collapsed}
		<div class="section-body">
			{#if showCreateForm}
				<div class="create-form">
					<textarea
						bind:value={createDescription}
						placeholder="Describe your task, e.g. 'Extract financial data from K-1 tax forms' or 'Summarize meeting notes with action items'"
						rows={3}
						disabled={generating}
					></textarea>
					<div class="create-actions">
						<button class="tpl-btn" onclick={() => { showCreateForm = false; createDescription = ''; }} disabled={generating}>Cancel</button>
						<button class="tpl-btn primary" onclick={handleGenerate} disabled={generating || !createDescription.trim()}>
							{generating ? 'Generating...' : 'Create with AI'}
						</button>
					</div>
				</div>
			{/if}
			{#if templates.length === 0 && !showCreateForm}
				<p class="empty-hint">No templates — click + to create one</p>
			{:else}
				<div class="template-list">
					{#each templates as tpl}
						<div class="template-item" class:selected={selectedId === tpl.id}>
							<!-- Template header row -->
							<div class="tpl-header">
								<button
									class="tpl-radio"
									class:checked={selectedId === tpl.id}
									onclick={() => handleSelect(tpl.id)}
								>
									<span class="radio-dot"></span>
								</button>
								<button class="tpl-name-btn" onclick={() => toggleExpand(tpl.id)}>
									<span class="tpl-name">{tpl.name}</span>
									{#if selectedId === tpl.id}
										<span class="selected-badge">Selected</span>
									{/if}
								</button>
								<span class="expand-icon">{expandedId === tpl.id ? '▾' : '▸'}</span>
							</div>

							<!-- Expanded edit form -->
							{#if expandedId === tpl.id}
								{@const ro = !!tpl.is_default}
								<div class="tpl-details">
									<div class="detail-field">
										<label>Name</label>
										{#if ro}<input type="text" value={tpl.name} readonly />{:else}<input type="text" bind:value={editName} />{/if}
									</div>
									<div class="detail-field">
										<label>Description</label>
										{#if ro}<input type="text" value={tpl.description || ''} readonly />{:else}<input type="text" bind:value={editDescription} placeholder="Brief description" />{/if}
									</div>
									<div class="detail-field">
										<label>System Prompt</label>
										{#if ro}<textarea readonly rows={3}>{tpl.system_prompt || ''}</textarea>{:else}<textarea bind:value={editSystemPrompt} rows={3}></textarea>{/if}
									</div>
									{#if tpl.optimization_run_id}
										<button class="tpl-btn opt-results-btn" onclick={() => viewOptimizationResults(tpl.optimization_run_id!)}>
											View Optimization Results
										</button>
									{/if}
									{#if ro}
										<p class="readonly-note">Default template — read-only.</p>
									{:else}
										<div class="tpl-actions">
											<button class="tpl-btn danger" onclick={() => (deleteConfirmId = tpl.id)}>Delete</button>
											<div class="tpl-actions-right">
												<button class="tpl-btn" onclick={() => handleSetDefault(tpl.id)}>Set Default</button>
												<button class="tpl-btn primary" onclick={() => handleSave(tpl)} disabled={savingId === tpl.id}>
													{savingId === tpl.id ? 'Saving...' : 'Save'}
												</button>
											</div>
										</div>
									{/if}
								</div>
							{/if}
						</div>
					{/each}
				</div>
			{/if}
		</div>
	{/if}
</div>

<!-- Unsaved edits confirmation -->
{#if discardConfirmTarget}
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
	<div class="confirm-overlay" onclick={() => (discardConfirmTarget = null)} role="dialog" tabindex="-1">
		<div class="confirm-box" onclick={(e) => e.stopPropagation()}>
			<p>You have unsaved template edits. Discard changes?</p>
			<div class="confirm-btns">
				<button class="tpl-btn" onclick={() => (discardConfirmTarget = null)}>Keep Editing</button>
				<button class="tpl-btn danger" onclick={confirmDiscardAndExpand}>Discard</button>
			</div>
		</div>
	</div>
{/if}

<!-- Delete confirmation -->
{#if deleteConfirmId}
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
	<div class="confirm-overlay" onclick={() => (deleteConfirmId = null)} role="dialog" tabindex="-1">
		<div class="confirm-box" onclick={(e) => e.stopPropagation()}>
			<p>Delete this template?</p>
			<div class="confirm-btns">
				<button class="tpl-btn" onclick={() => (deleteConfirmId = null)}>Cancel</button>
				<button class="tpl-btn danger" onclick={() => handleDelete(deleteConfirmId!)}>Delete</button>
			</div>
		</div>
	</div>
{/if}

<!-- Optimization Results Modal -->
{#if optResultsModal.open}
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
	<div class="opt-modal-backdrop" onclick={() => (optResultsModal = { open: false, loading: false, data: null })} onkeydown={(e) => { if (e.key === 'Escape') optResultsModal = { open: false, loading: false, data: null }; }} role="dialog" tabindex="-1">
		<div class="opt-modal" onclick={(e) => e.stopPropagation()}>
			<div class="opt-modal-header">
				<h3>Optimization Results</h3>
				<button class="close-btn" onclick={() => (optResultsModal = { open: false, loading: false, data: null })}>&times;</button>
			</div>

			{#if optResultsModal.loading}
				<div class="opt-modal-loading">
					<div class="opt-spinner"></div>
					<p>Loading optimization results...</p>
				</div>
			{:else if optResultsModal.data}
				{@const data = optResultsModal.data}
				{@const runInfo = data.run_info || {}}
				{@const iterations = data.iterations || []}
				{@const summary = data.summary || {}}

				<!-- Summary Header -->
				<div class="opt-summary">
					<div class="opt-summary-left">
						<div class="opt-template-name">{runInfo.template_name || 'Template'}</div>
						<div class="opt-meta">
							{formatDate(runInfo.created_at)}
							{#if runInfo.iteration_count} &middot; {runInfo.iteration_count} iterations{/if}
						</div>
					</div>
					<div class="opt-summary-right">
						{#if summary.best_score != null}
							<span class="opt-best-score" style="background: {scoreColor(summary.best_score)}">
								Best: {summary.best_score.toFixed(1)}/10
							</span>
						{/if}
						<span class="opt-status" class:completed={runInfo.status === 'completed'}>
							{runInfo.status || 'Unknown'}
						</span>
					</div>
				</div>

				<!-- Iterations -->
				{#if iterations.length > 0}
					<div class="opt-section">
						<h4>Iterations</h4>
						<div class="opt-iterations">
							{#each iterations as iter}
								<div class="opt-iter">
									<div class="opt-iter-header">
										<span class="opt-iter-num">#{iter.iteration_number}</span>
										<span class="opt-iter-score" style="color: {scoreColor(iter.evaluation_score || 0)}">
											{(iter.evaluation_score || 0).toFixed(1)}/10
										</span>
									</div>
									<div class="opt-score-bar">
										<div class="opt-score-fill" style="width: {Math.min((iter.evaluation_score || 0) * 10, 100)}%; background: {scoreColor(iter.evaluation_score || 0)}"></div>
									</div>
									{#if iter.optimized_query}
										<div class="opt-iter-query">{iter.optimized_query}</div>
									{/if}
									{#if iter.evaluation_notes}
										<div class="opt-iter-notes">{iter.evaluation_notes}</div>
									{/if}
								</div>
							{/each}
						</div>
					</div>
				{/if}

				<!-- Best Iteration Detail -->
				{#if summary.best_iteration}
					{@const best = summary.best_iteration}
					<div class="opt-section">
						<h4>Best Result</h4>
						<div class="opt-best-detail">
							{#if best.optimized_query}
								<div class="opt-detail-field">
									<label>Optimized Query</label>
									<div class="opt-detail-value">{best.optimized_query}</div>
								</div>
							{/if}
							{#if best.prompt_variant?.system_prompt}
								<div class="opt-detail-field">
									<label>System Prompt</label>
									<div class="opt-detail-value">{best.prompt_variant.system_prompt}</div>
								</div>
							{/if}
							{#if best.prompt_variant?.query_prefix}
								<div class="opt-detail-field">
									<label>Query Prefix</label>
									<div class="opt-detail-value">{best.prompt_variant.query_prefix}</div>
								</div>
							{/if}
							{#if best.prompt_variant?.query_suffix}
								<div class="opt-detail-field">
									<label>Query Suffix</label>
									<div class="opt-detail-value">{best.prompt_variant.query_suffix}</div>
								</div>
							{/if}
						</div>
					</div>
				{/if}
			{:else}
				<div class="opt-modal-loading">
					<p>No results available.</p>
				</div>
			{/if}

			<div class="opt-modal-footer">
				<button class="tpl-btn" onclick={() => (optResultsModal = { open: false, loading: false, data: null })}>Close</button>
			</div>
		</div>
	</div>
{/if}

<style>
	.panel-section { border-bottom: 1px solid var(--border); }

	.section-header {
		display: flex; justify-content: space-between; align-items: center;
		padding: 0.6rem 0.75rem; background: none; border: none; width: 100%;
		cursor: pointer; color: var(--text-heading); font-weight: 700; font-size: 0.82rem;
		text-align: left;
	}
	.section-header:hover { background: var(--bg-hover); }
	.section-actions { display: flex; align-items: center; gap: 0.4rem; }
	.collapse-icon { font-size: 0.7rem; color: var(--text-muted); }

	.add-btn {
		background: none; border: 1px solid var(--border); color: var(--text-secondary);
		width: 18px; height: 18px; border-radius: 4px; cursor: pointer;
		font-size: 0.8rem; display: flex; align-items: center; justify-content: center;
		line-height: 1;
	}
	.add-btn:hover { background: var(--accent); color: white; border-color: var(--accent); }

	.section-body { padding: 0 0.75rem 0.75rem; }

	.template-list { display: flex; flex-direction: column; gap: 4px; }

	.template-item {
		border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
		transition: border-color 0.15s;
	}
	.template-item.selected { border-color: var(--accent); background: var(--accent-bg); }

	.tpl-header {
		display: flex; align-items: center; gap: 0.4rem; padding: 0.45rem 0.6rem;
	}

	.tpl-radio {
		width: 16px; height: 16px; border: 2px solid var(--border); border-radius: 50%;
		background: none; cursor: pointer; display: flex; align-items: center;
		justify-content: center; flex-shrink: 0; padding: 0;
	}
	.tpl-radio.checked { border-color: var(--accent); }
	.radio-dot {
		width: 8px; height: 8px; border-radius: 50%; background: transparent;
	}
	.tpl-radio.checked .radio-dot { background: var(--accent); }

	.tpl-name-btn {
		flex: 1; background: none; border: none; cursor: pointer;
		display: flex; align-items: center; gap: 0.4rem; text-align: left; padding: 0;
	}
	.tpl-name { font-size: 0.78rem; color: var(--text-heading); font-weight: 500; }

	.selected-badge {
		font-size: 0.58rem; font-weight: 700; color: var(--success); background: rgba(92, 173, 111, 0.15);
		padding: 1px 6px; border-radius: 8px; text-transform: uppercase;
	}
	.type-badge {
		font-size: 0.58rem; font-weight: 700; color: var(--accent); background: var(--accent-bg);
		padding: 1px 6px; border-radius: 8px;
	}
	.expand-icon { font-size: 0.6rem; color: var(--text-muted); }

	/* Expanded details */
	.tpl-details {
		padding: 0.5rem 0.6rem; border-top: 1px solid var(--border);
		background: var(--bg-surface-alt); display: flex; flex-direction: column; gap: 0.5rem;
	}
	.detail-field { display: flex; flex-direction: column; gap: 2px; }
	.detail-field label { font-size: 0.68rem; font-weight: 600; color: var(--text-secondary); }
	.detail-field input, .detail-field select, .detail-field textarea {
		font-size: 0.78rem; padding: 0.3rem 0.5rem; border: 1px solid var(--border);
		border-radius: 5px; color: var(--text-heading); background: var(--bg-input); font-family: inherit;
		color-scheme: dark;
	}
	.detail-field input:focus, .detail-field select:focus, .detail-field textarea:focus {
		outline: none; border-color: var(--accent);
	}
	.detail-field textarea { resize: vertical; min-height: 60px; }
	.detail-field input:read-only, .detail-field textarea:read-only {
		background: var(--bg-hover); color: var(--text-secondary);
	}

	.readonly-note {
		font-size: 0.7rem; color: var(--text-muted); font-style: italic; margin: 0;
	}

	.detail-row { display: flex; gap: 0.5rem; }
	.detail-row .flex1 { flex: 1; min-width: 0; }

	.tpl-actions {
		display: flex; justify-content: space-between; align-items: center; gap: 0.5rem;
		margin-top: 0.25rem;
	}
	.tpl-actions-right { display: flex; gap: 0.35rem; }

	.tpl-btn {
		padding: 0.3rem 0.7rem; border-radius: 5px; font-size: 0.72rem;
		font-weight: 600; cursor: pointer; border: 1px solid var(--border);
		background: var(--bg-hover); color: var(--text-heading);
	}
	.tpl-btn:hover { background: var(--border); }
	.tpl-btn.primary { background: var(--accent); color: white; border-color: var(--accent); }
	.tpl-btn.primary:hover:not(:disabled) { filter: brightness(1.15); }
	.tpl-btn.primary:disabled { opacity: 0.5; }
	.tpl-btn.danger { background: transparent; color: var(--danger); border-color: var(--danger); }
	.tpl-btn.danger:hover { background: rgba(196, 92, 92, 0.15); }

	/* Delete confirmation overlay */
	.confirm-overlay {
		position: fixed; inset: 0; z-index: 1100;
		background: rgba(0,0,0,0.3);
		display: flex; align-items: center; justify-content: center;
	}
	.confirm-box {
		background: var(--bg-surface); border-radius: 10px; padding: 1.25rem 1.5rem;
		box-shadow: 0 16px 48px rgba(0,0,0,0.15); max-width: 300px;
	}
	.confirm-box p { margin: 0 0 1rem; font-size: 0.88rem; color: var(--text-heading); }
	.confirm-btns { display: flex; gap: 0.5rem; justify-content: flex-end; }

	.empty-hint { text-align: center; color: var(--text-muted); font-size: 0.78rem; margin: 0.75rem 0; }

	/* Create form */
	.create-form {
		border: 1px solid var(--border); border-radius: 8px;
		padding: 0.5rem; margin-bottom: 0.5rem; background: var(--bg-surface);
	}
	.create-form textarea {
		width: 100%; border: 1px solid var(--border); border-radius: 6px;
		padding: 0.4rem 0.5rem; font-size: 0.78rem; font-family: var(--font-sans);
		color: var(--text-primary); background: var(--bg-input);
		resize: vertical; outline: none; min-height: 60px;
	}
	.create-form textarea:focus { border-color: var(--text-muted); }
	.create-form textarea::placeholder { color: var(--text-muted); }
	.create-actions { display: flex; justify-content: flex-end; gap: 0.3rem; margin-top: 0.4rem; }

	/* Optimization results button */
	.opt-results-btn {
		width: 100%; text-align: center; background: var(--accent-bg); color: var(--accent-light);
		border-color: var(--accent-border); font-size: 0.72rem;
	}
	.opt-results-btn:hover { filter: brightness(1.15); }

	/* Optimization results modal */
	.opt-modal-backdrop {
		position: fixed; inset: 0; background: rgba(0,0,0,0.4);
		display: flex; align-items: center; justify-content: center; z-index: 1200;
	}
	.opt-modal {
		background: var(--bg-surface); border-radius: 12px; width: 560px; max-width: 90vw;
		max-height: 80vh; display: flex; flex-direction: column;
		box-shadow: 0 20px 60px rgba(0,0,0,0.15);
	}
	.opt-modal-header {
		display: flex; justify-content: space-between; align-items: center;
		padding: 0.75rem 1rem; border-bottom: 1px solid var(--border);
	}
	.opt-modal-header h3 { margin: 0; font-size: 0.95rem; color: var(--text-heading); letter-spacing: 0.05em; }
	.close-btn { background: none; border: none; font-size: 1.4rem; color: var(--text-muted); cursor: pointer; line-height: 1; }

	.opt-modal-loading {
		padding: 2rem; text-align: center; color: var(--text-secondary);
	}
	.opt-spinner {
		width: 28px; height: 28px; border: 3px solid var(--border); border-top-color: var(--accent);
		border-radius: 50%; animation: spin 0.7s linear infinite; margin: 0 auto 0.75rem;
	}
	@keyframes spin { to { transform: rotate(360deg); } }

	/* Summary */
	.opt-summary {
		display: flex; justify-content: space-between; align-items: center;
		padding: 0.75rem 1rem; background: var(--bg-hover); border-bottom: 1px solid var(--border);
	}
	.opt-template-name { font-size: 0.85rem; font-weight: 600; color: var(--text-heading); }
	.opt-meta { font-size: 0.7rem; color: var(--text-secondary); margin-top: 2px; }
	.opt-summary-right { display: flex; align-items: center; gap: 0.4rem; }
	.opt-best-score {
		font-size: 0.72rem; font-weight: 700; color: white;
		padding: 2px 8px; border-radius: 6px;
	}
	.opt-status {
		font-size: 0.65rem; font-weight: 600; color: var(--text-secondary);
		padding: 2px 6px; border-radius: 4px; background: var(--bg-hover); text-transform: capitalize;
	}
	.opt-status.completed { color: #059669; background: #d1fae5; }

	/* Sections */
	.opt-section { padding: 0.75rem 1rem; overflow-y: auto; }
	.opt-section h4 { font-size: 0.78rem; font-weight: 700; color: var(--accent); margin: 0 0 0.5rem; text-transform: uppercase; letter-spacing: 0.03em; }

	/* Iterations */
	.opt-iterations { display: flex; flex-direction: column; gap: 0.5rem; }
	.opt-iter {
		border: 1px solid var(--border); border-radius: 8px; padding: 0.5rem 0.6rem;
		background: var(--bg-surface);
	}
	.opt-iter-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
	.opt-iter-num { font-size: 0.72rem; font-weight: 700; color: var(--text-heading); }
	.opt-iter-score { font-size: 0.78rem; font-weight: 700; }
	.opt-score-bar { height: 4px; background: var(--bg-hover); border-radius: 2px; overflow: hidden; margin-bottom: 4px; }
	.opt-score-fill { height: 100%; border-radius: 2px; transition: width 0.3s; }
	.opt-iter-query { font-size: 0.72rem; color: var(--text-heading); margin-top: 4px; }
	.opt-iter-notes { font-size: 0.68rem; color: var(--text-secondary); margin-top: 2px; font-style: italic; }

	/* Best result detail */
	.opt-best-detail { display: flex; flex-direction: column; gap: 0.5rem; }
	.opt-detail-field { display: flex; flex-direction: column; gap: 2px; }
	.opt-detail-field label { font-size: 0.68rem; font-weight: 600; color: var(--text-secondary); }
	.opt-detail-value {
		font-size: 0.75rem; color: var(--text-heading); background: var(--bg-hover);
		border: 1px solid var(--border); border-radius: 6px; padding: 0.35rem 0.5rem;
		white-space: pre-wrap; word-break: break-word;
	}

	/* Footer */
	.opt-modal-footer {
		display: flex; justify-content: flex-end; padding: 0.6rem 1rem;
		border-top: 1px solid var(--border);
	}
</style>
