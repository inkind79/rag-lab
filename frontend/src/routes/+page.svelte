<script lang="ts">
	import { tick } from 'svelte';
	import { messages, isStreaming, currentModel, currentModelDisplay, chatHistoryCursor } from '$lib/stores/chat';
	import { activeSession, activeSessionId, optimizedQuery } from '$lib/stores/session';
	import { streamChat, getSessionId, saveSettings, getAvailableModels, getSessionData, getChatHistoryPage } from '$lib/api/client';
	import Markdown from '$lib/components/Markdown.svelte';
	import ModelBadge from '$lib/components/ModelBadge.svelte';
	import RetrievedImages from '$lib/components/RetrievedImages.svelte';
	import FeedbackSection from '$lib/components/FeedbackSection.svelte';
	import { toasts } from '$lib/stores/toast';
	import type { ChatMessage, RetrievedImage } from '$lib/stores/chat';

	// Per-message relevance tracking: messageIndex → selected image paths
	let relevanceMap = $state<Record<number, string[]>>({});

	function getRelevantPaths(index: number): string[] {
		return relevanceMap[index] || [];
	}

	function setRelevantPaths(index: number, paths: string[]) {
		relevanceMap[index] = paths;
	}

	let query = $state('');
	let chatContainer: HTMLDivElement;
	let textareaEl: HTMLTextAreaElement;

	// Pasted images
	let pastedImages = $state<string[]>([]);

	// Message bar toggles — 'chat' | 'rag' | 'batch'
	let chatMode = $state<'chat' | 'rag' | 'batch'>('chat');
	let ragMode = $derived(chatMode !== 'chat');
	let hasDocs = $derived(($activeSession?.indexed_files?.length ?? 0) > 0);
	// Auto-downgrade to Chat if no documents are available
	$effect(() => { if (!hasDocs && chatMode !== 'chat') chatMode = 'chat'; });
	let ocrEnabled = $state(false);
	let adaptiveEnabled = $state(false);

	// Model selector
	interface ModelOpt { value: string; label: string }
	let selectedModel = $state('');
	let modelOptions = $state<ModelOpt[]>([]);

	$effect(() => {
		if ($activeSessionId) loadModelState();
	});

	async function loadModelState() {
		try {
			const [modelsData, sessionResp] = await Promise.all([
				getAvailableModels(),
				getSessionData(),
			]);
			const all: ModelOpt[] = [];
			for (const m of modelsData.cloud || []) all.push(m);
			for (const m of modelsData.huggingface || []) all.push(m);
			for (const m of modelsData.ollama || []) all.push(m);
			modelOptions = all;
			const data = sessionResp?.session_data || sessionResp;
			if (data?.generation_model) selectedModel = data.generation_model;
		} catch { /* ignore */ }
	}

	async function onModelChange() {
		if (!selectedModel) return;
		try {
			await saveSettings({ generation_model: selectedModel });
		} catch { /* ignore */ }
	}

	// Streaming stage tracking
	let streamStage = $state('');
	// Batch processing tracking
	let batchInfo = $state<{total: number; current: string; index: number} | null>(null);

	// Abort controller for cancelling in-flight streams on session switch
	let streamAbort: AbortController | null = null;

	// Load chat history and abort stream when session changes
	$effect(() => {
		const sid = $activeSessionId;
		if (streamAbort) {
			streamAbort.abort();
			streamAbort = null;
			isStreaming.set(false);
		}
		if (sid) loadChatHistory();
	});

	async function loadChatHistory() {
		try {
			const resp = await getSessionData();
			const data = resp?.session_data || resp;
			const history = data?.chat_history || [];
			if (history.length > 0) {
				const mapped = history.map((msg: any, i: number) => ({
					id: msg.id || `hist-${i}`,
					role: msg.role || 'user',
					type: msg.type,
					content: msg.content || '',
					timestamp: msg.timestamp || 0,
					model: msg.model,
					modelDisplayName: msg.modelDisplayName || msg.model_display_name,
					reasoning: msg.reasoning,
					images: msg.images,
					template_name: msg.template_name,
				}));
				messages.set(mapped);
			} else {
				messages.set([]);
			}
			// Sync chat bar toggles with session settings
			if (data) {
				adaptiveEnabled = data.use_score_slope ?? false;
				ocrEnabled = data.use_ocr ?? false;
			}
		} catch {
			messages.set([]);
		}
	}

	let loadingEarlier = $state(false);

	async function loadEarlier() {
		// Walk one page backward and prepend the older messages onto the store.
		// We preserve the user's scroll position visually by capturing the
		// scrollHeight before the prepend and adjusting scrollTop after, so the
		// in-view conversation doesn't jump.
		const uuid = $activeSessionId;
		if (!uuid || loadingEarlier || !$chatHistoryCursor.has_more) return;

		loadingEarlier = true;
		const prevScrollHeight = chatContainer?.scrollHeight ?? 0;
		const prevScrollTop = chatContainer?.scrollTop ?? 0;

		try {
			const page = await getChatHistoryPage(uuid, 50, $chatHistoryCursor.first_index);
			const normalized = page.messages.map((msg: any) => ({
				...msg,
				timestamp: typeof msg.timestamp === 'string' ? new Date(msg.timestamp).getTime() : (msg.timestamp || Date.now()),
				images: Array.isArray(msg.images) ? msg.images : [],
			}));
			messages.update((current) => [...normalized, ...current]);
			chatHistoryCursor.set({
				first_index: page.first_index,
				total: page.total,
				has_more: page.has_more,
			});
			// Restore scroll position so the user's view doesn't jump.
			await tick();
			if (chatContainer) {
				const newScrollHeight = chatContainer.scrollHeight;
				chatContainer.scrollTop = prevScrollTop + (newScrollHeight - prevScrollHeight);
			}
		} catch (e) {
			console.error('Failed to load earlier messages:', e);
		} finally {
			loadingEarlier = false;
		}
	}

	async function handleSubmit() {
		const effectiveQuery = $optimizedQuery || query;
		if ((!effectiveQuery.trim() && pastedImages.length === 0) || $isStreaming) return;

		const userQuery = effectiveQuery;
		const userImages = [...pastedImages];
		query = '';
		pastedImages = [];
		resetTextarea();

		messages.update((msgs) => [
			...msgs,
			{ role: 'user', content: userQuery, timestamp: Date.now(), images: userImages.length > 0 ? userImages.map(src => ({ path: src, score: 0 })) : undefined },
		]);

		await runStream(userQuery, userImages, chatMode === 'batch');
	}

	/** Retry a failed assistant message by re-running the original request. */
	async function retryMessage(index: number) {
		if ($isStreaming) return;
		const msg = $messages[index];
		if (!msg?.retryContext) return;
		const ctx = msg.retryContext;
		// Drop the failed assistant message before retrying.
		messages.update((msgs) => msgs.filter((_, i) => i !== index));
		await runStream(ctx.query, ctx.images || [], !!ctx.isBatch);
	}

	async function runStream(userQuery: string, userImages: string[], isBatch: boolean) {
		isStreaming.set(true);
		streamStage = 'connecting';
		batchInfo = null;

		// Per-document state — reset on each doc_start in batch mode
		let assistantContent = '';
		let reasoning = '';
		let retrievedImages: RetrievedImage[] = [];

		function flushMessages() {
			messages.update((msgs) => {
				const last = msgs[msgs.length - 1];
				if (last?.role === 'assistant') {
					msgs[msgs.length - 1] = {
						...last,
						content: assistantContent,
						reasoning,
						model: $currentModel,
						modelDisplayName: $currentModelDisplay,
						images: retrievedImages.length > 0 ? [...retrievedImages] : last.images,
					};
				}
				return [...msgs];
			});
			if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight;
		}

		// Non-batch: create one assistant message up front
		// Batch: a new message is created per document on doc_start
		if (!isBatch) {
			messages.update((msgs) => [
				...msgs,
				{ role: 'assistant', id: `msg-${Date.now()}`, content: '', timestamp: Date.now(), images: [] },
			]);
		}

		const retryContext = {
			query: userQuery,
			images: userImages.length > 0 ? userImages : undefined,
			isBatch,
		};

		try {
			const sessionId = getSessionId() || '';
			streamAbort = new AbortController();
			const resp = await streamChat(userQuery, sessionId, { isRagMode: ragMode, isBatchMode: isBatch, pastedImages: userImages.length > 0 ? userImages : undefined, signal: streamAbort.signal });
			if (!resp.ok) {
				const text = await resp.text();
				messages.update((msgs) => [
					...msgs,
					{
						role: 'assistant',
						id: `msg-${Date.now()}`,
						content: '',
						timestamp: Date.now(),
						images: [],
						error: `Server returned ${resp.status}: ${text || resp.statusText}`,
						retryContext,
					},
				]);
				return;
			}
			if (!resp.body) return;
			const reader = resp.body.getReader();
			const decoder = new TextDecoder();
			let buffer = '';

			while (true) {
				const { done, value } = await reader.read();
				if (done) break;

				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split('\n');
				buffer = lines.pop() || '';
				for (const line of lines) {
					if (!line.startsWith('data: ')) continue;
					try {
						const data = JSON.parse(line.slice(6));

						switch (data.type) {
							case 'response':
								assistantContent += data.content;
								streamStage = 'generating';
								break;
							case 'reasoning':
								reasoning += data.content;
								streamStage = 'reasoning';
								break;
							case 'reasoning_complete':
								break;
							case 'model_info':
								currentModel.set(data.model || '');
								currentModelDisplay.set(data.display_name || data.model || '');
								break;
							case 'images':
								if (Array.isArray(data.images)) {
									retrievedImages = data.images;
								}
								break;
							case 'stage':
								streamStage = data.stage || '';
								break;
							case 'batch_start':
								batchInfo = { total: data.total_docs || 0, current: '', index: 0 };
								break;
							case 'doc_start': {
								if (batchInfo) {
									batchInfo = { total: batchInfo.total, current: data.doc_name || '', index: data.doc_index || 0 };
								}
								// Reset per-document state
								assistantContent = `### ${data.doc_name || 'Document'}\n\n`;
								reasoning = '';
								retrievedImages = [];
								// Create a new assistant message for this document
								messages.update((msgs) => [
									...msgs,
									{ role: 'assistant', id: `msg-${Date.now()}-doc${data.doc_index ?? 0}`, content: '', timestamp: Date.now(), images: [] },
								]);
								break;
							}
							case 'doc_complete':
								// Finalize this document's message
								flushMessages();
								break;
							case 'summary_processing':
								streamStage = 'extracting';
								break;
							case 'complete':
								break;
							case 'error':
								assistantContent += `\n\n**Error:** ${data.message || data.error || 'Unknown error'}`;
								break;
						}

						// Flush after each event and yield to browser paint cycle
						// for word-by-word streaming (tick only waits for Svelte, not paint)
						flushMessages();
						await new Promise(r => requestAnimationFrame(r));
					} catch {
						// Skip non-JSON lines
					}
				}
			}
			// Final flush
			flushMessages();
		} catch (error: any) {
			if (error?.name !== 'AbortError') {
				console.error('Stream error:', error);
				const errMsg = error?.message || String(error) || 'Connection lost';
				messages.update((msgs) => {
					const last = msgs[msgs.length - 1];
					// Attach the error to the in-flight assistant message (preserving
					// any partial content the user already saw) rather than overwriting it.
					if (last?.role === 'assistant') {
						msgs[msgs.length - 1] = { ...last, error: errMsg, retryContext };
					} else {
						msgs.push({
							role: 'assistant',
							id: `msg-err-${Date.now()}`,
							content: '',
							timestamp: Date.now(),
							images: [],
							error: errMsg,
							retryContext,
						});
					}
					return [...msgs];
				});
			}
		} finally {
			flushMessages();
			streamAbort = null;
			isStreaming.set(false);
			streamStage = '';
			batchInfo = null;
			if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight;
		}
	}

	function handleFormSubmit(e: SubmitEvent) {
		e.preventDefault();
		e.stopPropagation();
		handleSubmit();
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			handleSubmit();
		}
	}

	function handleInput() {
		if (textareaEl) {
			textareaEl.style.height = 'auto';
			textareaEl.style.height = Math.min(textareaEl.scrollHeight, 200) + 'px';
		}
	}

	function resetTextarea() {
		if (textareaEl) textareaEl.style.height = 'auto';
	}

	function handlePaste(e: ClipboardEvent) {
		const items = e.clipboardData?.items;
		if (!items) return;
		for (const item of items) {
			if (item.type.startsWith('image/')) {
				e.preventDefault();
				const file = item.getAsFile();
				if (!file) continue;
				const reader = new FileReader();
				reader.onload = () => {
					if (typeof reader.result === 'string') {
						pastedImages = [...pastedImages, reader.result];
					}
				};
				reader.readAsDataURL(file);
			}
		}
	}

	function removePastedImage(index: number) {
		pastedImages = pastedImages.filter((_, i) => i !== index);
	}

	let composerDragOver = $state(false);

	function handleDrop(e: DragEvent) {
		e.preventDefault();
		composerDragOver = false;
		const files = e.dataTransfer?.files;
		if (!files) return;
		for (const file of files) {
			if (file.type.startsWith('image/')) {
				const reader = new FileReader();
				reader.onload = () => {
					if (typeof reader.result === 'string') {
						pastedImages = [...pastedImages, reader.result];
					}
				};
				reader.readAsDataURL(file);
			}
		}
	}

	function handleDragOver(e: DragEvent) {
		e.preventDefault();
		composerDragOver = true;
	}

	function handleDragLeave() {
		composerDragOver = false;
	}

	async function toggleOcr() {
		ocrEnabled = !ocrEnabled;
		try {
			await saveSettings({ use_ocr: ocrEnabled });
		} catch {
			ocrEnabled = !ocrEnabled; // Revert on failure
			toasts.error('Failed to update OCR setting');
		}
	}

	async function toggleAdaptive() {
		adaptiveEnabled = !adaptiveEnabled;
		try {
			await saveSettings({ use_score_slope: adaptiveEnabled });
		} catch {
			adaptiveEnabled = !adaptiveEnabled; // Revert on failure
			toasts.error('Failed to update adaptive setting');
		}
	}

	function getStageLabel(stage: string): string {
		if (batchInfo) {
			return `Processing ${batchInfo.current || `document ${batchInfo.index + 1}`} (${batchInfo.index + 1}/${batchInfo.total})...`;
		}
		switch (stage) {
			case 'connecting': return 'Preparing...';
			case 'analyzing': return 'Analyzing query...';
			case 'searching': return 'Searching documents...';
			case 'extracting': return 'Extracting context...';
			case 'reasoning': return 'Thinking...';
			case 'generating': return 'Generating response...';
			default: return 'Processing...';
		}
	}

	function getVendorColor(model?: string): string {
		if (!model) return '#e5e7eb';
		const m = model.toLowerCase();
		if (m.includes('phi') || m.includes('devstral')) return '#0078D4';
		if (m.includes('llama') || m.includes('meta')) return '#8C9EFF';
		if (m.includes('gemma')) return '#4285F4';
		if (m.includes('mistral') || m.includes('magistral')) return '#5546FF';
		if (m.includes('qwen') || m.includes('qwq')) return '#6B46C1';
		if (m.includes('deepseek')) return '#008B7A';
		return '#9CA3AF';
	}

	function formatTime(ts: number): string {
		if (!ts) return '';
		const d = new Date(ts);
		return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
	}

	// Normalize images from chat history (may have different format than SSE)
	function getMessageImages(msg: ChatMessage): RetrievedImage[] {
		if (!msg.images || msg.images.length === 0) return [];
		return msg.images.map(img => {
			if (typeof img === 'string') {
				// Old format: just a path string
				return { path: img, score: 0 };
			}
			return img;
		});
	}
</script>

<div class="chat-page" class:empty={$messages.length === 0}>
	{#if $messages.length === 0}
		<!-- Empty state: centered hero with input -->
		<div class="hero">
			<div class="hero-content">
				<h1>What would you like to explore?</h1>
			</div>

			<div class="hero-input">
				<form class="composer" class:drag-over={composerDragOver} onsubmit={handleFormSubmit} ondrop={handleDrop} ondragover={handleDragOver} ondragleave={handleDragLeave}>
					{#if pastedImages.length > 0}
						<div class="paste-previews">
							{#each pastedImages as src, idx}
								<div class="paste-thumb">
									<img src={src} alt="Pasted" />
									<button class="paste-remove" onclick={() => removePastedImage(idx)}>&times;</button>
								</div>
							{/each}
						</div>
					{/if}
					<textarea
						bind:this={textareaEl}
						bind:value={query}
						placeholder="Ask anything..."
						onkeydown={handleKeydown}
						oninput={handleInput}
						onpaste={handlePaste}
						disabled={$isStreaming}
						rows={3}
					></textarea>
					<div class="composer-footer">
						<div class="composer-actions">
							<div class="mode-switch">
								<button type="button" class="mode-opt" class:selected={chatMode === 'chat'} onclick={() => { chatMode = 'chat'; }}>Chat</button>
								<button type="button" class="mode-opt" class:selected={chatMode === 'rag'} disabled={!hasDocs} onclick={() => { chatMode = 'rag'; }}>RAG</button>
								<button type="button" class="mode-opt" class:selected={chatMode === 'batch'} disabled={!hasDocs} onclick={() => { chatMode = 'batch'; }}>Batch</button>
							</div>
							{#if modelOptions.length > 0}
								<select class="model-select" bind:value={selectedModel} onchange={onModelChange}>
									{#each modelOptions as m}<option value={m.value}>{m.label}</option>{/each}
								</select>
							{/if}
							<button type="button" class="action-pill" class:active={ocrEnabled}
								onclick={toggleOcr}>OCR</button>
							<button type="button" class="action-pill" class:active={adaptiveEnabled}
								onclick={toggleAdaptive}>Adaptive</button>
						</div>
						<button type="submit" class="send-btn" disabled={$isStreaming || (!query.trim() && pastedImages.length === 0)}>
							{#if $isStreaming}
								<span class="send-icon spinning">⏳</span>
							{:else}
								<span class="send-icon">↑</span>
							{/if}
						</button>
					</div>
				</form>
			</div>

		</div>
	{/if}

	<div class="messages-area" bind:this={chatContainer} class:hidden={$messages.length === 0}>

		{#if $chatHistoryCursor.has_more}
			<div class="load-earlier">
				<button
					type="button"
					class="load-earlier-btn"
					onclick={loadEarlier}
					disabled={loadingEarlier}
				>
					{loadingEarlier ? 'Loading...' : `Load earlier messages (${$chatHistoryCursor.first_index} of ${$chatHistoryCursor.total} hidden)`}
				</button>
			</div>
		{/if}

		{#each $messages as msg, i}
			<div class="message {msg.role}" class:upload={msg.type === 'document_upload'} style={msg.role === 'assistant' ? `--vendor-color: ${getVendorColor(msg.model)}` : ''}>
				{#if msg.type === 'document_upload'}
					<div class="upload-msg">
						{#if Array.isArray(msg.content)}
							{#each msg.content as file}
								<div class="upload-file">
									{#if file.file_type === 'image'}📷{:else}📄{/if}
									<span class="upload-name">{file.filename}</span>
									{#if file.page_count}<span class="upload-pages">{file.page_count} pages</span>{/if}
								</div>
							{/each}
						{:else}
							<span>Documents uploaded</span>
						{/if}
					</div>

				{:else if msg.role === 'user'}
					<div class="msg-body user-body">{typeof msg.content === 'string' ? msg.content : ''}</div>

				{:else if msg.role === 'assistant'}
					<div class="assistant-header">
						{#if msg.model}
							<ModelBadge model={msg.model} displayName={msg.modelDisplayName || ''} />
						{/if}
						{#if msg.content && typeof msg.content === 'string'}
							<button class="copy-btn" onclick={() => { navigator.clipboard.writeText(msg.content as string); toasts.success('Copied'); }} title="Copy response">
								Copy
							</button>
						{/if}
					</div>

					{#if msg.reasoning}
						<details class="reasoning-block">
							<summary>Reasoning</summary>
							<Markdown content={msg.reasoning} />
						</details>
					{/if}

					<div class="assistant-content">
						{#if msg.content && typeof msg.content === 'string'}
							<Markdown content={msg.content} streaming={$isStreaming && i === $messages.length - 1} />
						{/if}
						{#if $isStreaming && i === $messages.length - 1 && (!msg.content || msg.content === '')}
							<div class="thinking-indicator">
								<div class="thinking-dots"><span></span><span></span><span></span></div>
							</div>
						{/if}
						{#if msg.error && !$isStreaming}
							<div class="stream-error" role="alert">
								<div class="stream-error-text">
									<strong>Response interrupted.</strong>
									<span class="stream-error-detail">{msg.error}</span>
								</div>
								{#if msg.retryContext}
									<button type="button" class="stream-error-retry" onclick={() => retryMessage(i)}>
										Retry
									</button>
								{/if}
							</div>
						{/if}
					</div>

					{@const images = getMessageImages(msg)}
					{#if images.length > 0}
						<RetrievedImages
							{images}
							showRelevance={!!msg.id}
							relevantPaths={getRelevantPaths(i)}
							onrelevancechange={(paths) => setRelevantPaths(i, paths)}
						/>
					{/if}
					{#if msg.id && !$isStreaming && images.length > 0}
						<FeedbackSection
							messageId={msg.id}
							relevantImages={relevanceMap[i] || []}
						/>
					{/if}
				{/if}
			</div>
		{/each}

		{#if $isStreaming && batchInfo && batchInfo.total > 0}
			<div class="batch-progress">
				<div class="batch-progress-bar">
					<div class="batch-progress-fill" style="width: {((batchInfo.index + 1) / batchInfo.total) * 100}%"></div>
				</div>
				<span class="batch-progress-text">{batchInfo.index + 1} / {batchInfo.total} documents</span>
			</div>
		{/if}
	</div>


	{#if $messages.length > 0}
		<div class="bottom-bar">
			<form class="composer" class:drag-over={composerDragOver} onsubmit={handleFormSubmit} ondrop={handleDrop} ondragover={handleDragOver} ondragleave={handleDragLeave}>
				{#if pastedImages.length > 0}
					<div class="paste-previews">
						{#each pastedImages as src, idx}
							<div class="paste-thumb">
								<img src={src} alt="Pasted" />
								<button class="paste-remove" onclick={() => removePastedImage(idx)}>&times;</button>
							</div>
						{/each}
					</div>
				{/if}
				{#if $optimizedQuery}
					<div class="optimized-lock">
						<span class="lock-badge">Optimized</span>
						<span class="lock-text">{$optimizedQuery}</span>
					</div>
				{/if}
				<textarea
					bind:this={textareaEl}
					bind:value={query}
					placeholder={$optimizedQuery ? 'Using optimized query...' : pastedImages.length > 0 ? 'Describe the image...' : 'Ask anything...'}
					onkeydown={handleKeydown}
					oninput={handleInput}
					onpaste={handlePaste}
					disabled={$isStreaming}
					readonly={!!$optimizedQuery}
					rows={1}
				></textarea>
				<div class="composer-footer">
					<div class="composer-actions">
						<div class="mode-switch">
							<button type="button" class="mode-opt" class:selected={chatMode === 'chat'} onclick={() => { chatMode = 'chat'; }}>Chat</button>
							<button type="button" class="mode-opt" class:selected={chatMode === 'rag'} onclick={() => { chatMode = 'rag'; }}>RAG</button>
							<button type="button" class="mode-opt" class:selected={chatMode === 'batch'} onclick={() => { chatMode = 'batch'; }}>Batch</button>
						</div>
						{#if modelOptions.length > 0}
							<select class="model-select" bind:value={selectedModel} onchange={onModelChange}>
								{#each modelOptions as m}<option value={m.value}>{m.label}</option>{/each}
							</select>
						{/if}
						<button type="button" class="action-pill" class:active={ocrEnabled}
							onclick={toggleOcr}>OCR</button>
						<button type="button" class="action-pill" class:active={adaptiveEnabled}
							onclick={toggleAdaptive}>Adaptive</button>
					</div>
					<button type="submit" class="send-btn" disabled={$isStreaming || (!query.trim() && !$optimizedQuery && pastedImages.length === 0)}>
						{#if $isStreaming}
							<span class="send-icon spinning">⏳</span>
						{:else}
							<span class="send-icon">↑</span>
						{/if}
					</button>
				</div>
			</form>
		</div>
	{/if}
</div>

<style>
	.chat-page { display: flex; flex-direction: column; height: 100%; background: var(--bg-surface); }
	.chat-page.empty { justify-content: center; }

	/* ── Hero (empty state) ── */
	.hero {
		display: flex; flex-direction: column; align-items: center;
		padding: 0 1.5rem 2rem; max-width: 680px; width: 100%; margin: 0 auto;
		animation: heroIn 0.4s ease;
	}
	@keyframes heroIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

	.hero-content { text-align: center; margin-bottom: 1.5rem; }
	.hero-content h1 {
		font-size: 1.75rem; font-weight: 600; color: var(--text-heading);
		margin: 0 0 0.5rem; line-height: 1.3;
	}
	.hero-input { width: 100%; margin-bottom: 1rem; }

	/* ── Composer (shared between hero and bottom bar) ── */
	.composer {
		background: var(--bg-surface); border: 1px solid var(--border);
		border-radius: 16px; overflow: hidden;
		transition: border-color 0.15s, box-shadow 0.15s;
	}
	.composer:focus-within {
		border-color: var(--text-muted);
		box-shadow: 0 0 0 1px var(--border);
	}
	.composer.drag-over {
		border-color: var(--accent); border-style: dashed;
		background: var(--accent-bg);
	}
	.composer textarea {
		width: 100%; background: transparent; border: none;
		color: var(--text-primary); padding: 0.85rem 1rem 0.4rem;
		font-size: 0.92rem; outline: none; resize: none; font-family: var(--font-sans);
		min-height: 52px; max-height: 200px; line-height: 1.5;
	}
	.composer textarea::placeholder { color: var(--text-muted); }

	.composer-footer {
		display: flex; align-items: center; justify-content: space-between;
		padding: 0.3rem 0.5rem 0.5rem 0.65rem;
	}
	.composer-actions { display: flex; gap: 4px; }
	/* Mode switch (Chat / RAG) */
	.mode-switch {
		display: flex; background: var(--bg-active); border-radius: 8px; padding: 2px; gap: 1px;
	}
	.mode-opt {
		background: none; border: none; color: var(--text-muted);
		padding: 3px 10px; border-radius: 6px; font-size: 0.7rem; font-weight: 600;
		cursor: pointer; font-family: var(--font-sans);
		transition: all 0.12s; display: flex; align-items: center; gap: 3px;
	}
	.mode-opt:hover:not(:disabled) { color: var(--text-secondary); }
	.mode-opt:disabled { opacity: 0.35; cursor: not-allowed; }
	.mode-opt.selected {
		background: var(--bg-surface); color: var(--text-heading);
		box-shadow: 0 1px 2px var(--shadow);
	}

	.model-select {
		background: var(--bg-active); border: 1px solid transparent; color: var(--text-muted);
		padding: 3px 6px; border-radius: 8px; font-size: 0.7rem; font-weight: 500;
		cursor: pointer; font-family: var(--font-sans); outline: none;
		max-width: 160px; color-scheme: dark light;
		background-image: url("data:image/svg+xml,%3Csvg width='8' height='5' viewBox='0 0 8 5' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M1 1L4 4L7 1' stroke='%239ca3af' stroke-width='1.2' stroke-linecap='round'/%3E%3C/svg%3E");
		background-repeat: no-repeat; background-position: right 6px center;
		padding-right: 18px;
		transition: all 0.1s;
	}
	.model-select:hover { border-color: var(--border); color: var(--text-secondary); }
	.model-select:focus { border-color: var(--text-muted); color: var(--text-heading); }

	.action-pill {
		background: none; border: 1px solid transparent; color: var(--text-muted);
		padding: 3px 10px; border-radius: 12px; font-size: 0.7rem; font-weight: 600;
		cursor: pointer; font-family: var(--font-sans);
		transition: all 0.1s;
	}
	.action-pill:hover { color: var(--text-secondary); background: var(--bg-active); }
	.action-pill.active { background: var(--bg-active); color: var(--text-heading); border-color: var(--border); }
	.action-pill.memory-toggle.active { background: var(--text-heading); color: var(--bg-surface); border-color: var(--text-heading); }

	/* ── Messages area ── */
	.messages-area {
		flex: 1; overflow-y: auto;
		padding: 1.5rem 1.5rem; scroll-behavior: smooth;
		max-width: 800px; width: 100%; margin: 0 auto;
	}
	.messages-area.hidden { display: none; }

	.load-earlier {
		display: flex; justify-content: center;
		padding: 0.5rem 0 1rem;
	}
	.load-earlier-btn {
		background: var(--bg-card, transparent);
		color: var(--text-secondary);
		border: 1px solid var(--border);
		border-radius: 999px;
		padding: 0.4rem 0.9rem;
		font-size: 0.78rem; font-weight: 500;
		font-family: var(--font-sans, inherit);
		cursor: pointer;
		transition: background 0.15s, border-color 0.15s, color 0.15s;
	}
	.load-earlier-btn:hover:not(:disabled) {
		background: var(--bg-hover);
		color: var(--text-primary);
		border-color: var(--text-secondary);
	}
	.load-earlier-btn:disabled { opacity: 0.55; cursor: progress; }
	.load-earlier-btn:focus-visible {
		outline: 2px solid var(--accent, #6366f1); outline-offset: 2px;
	}

	/* ── Bottom bar (when messages exist) ── */
	.bottom-bar {
		padding: 0.5rem 1rem 1rem;
		max-width: 800px; width: 100%; margin: 0 auto;
	}

	.message { margin-bottom: 1.5rem; max-width: 100%; animation: msgIn 0.15s ease; }
	@keyframes msgIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
	.message.user { display: flex; justify-content: flex-end; }
	.message.assistant { }

	/* Upload messages */
	.message.upload { max-width: 70%; margin: 0 auto 0.75rem; }
	.upload-msg { background: var(--bg-hover); border: 1px solid var(--border); border-radius: 10px; padding: 0.45rem 0.7rem; }
	.upload-file { display: flex; align-items: center; gap: 0.35rem; font-size: 0.78rem; color: var(--text-secondary); }
	.upload-name { font-weight: 600; color: var(--text-heading); }
	.upload-pages { font-size: 0.66rem; color: var(--text-muted); font-family: var(--font-mono); }

	/* User: simple bubble, no labels */
	.msg-body { line-height: 1.6; font-size: 0.9rem; }
	.user-body {
		background: var(--bg-active); color: var(--text-heading);
		padding: 0.65rem 0.9rem; border-radius: 18px; border-bottom-left-radius: 4px;
		display: inline-block; max-width: 85%;
		border: 1px solid var(--border);
		box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
	}

	/* Assistant: no bubble, flowing text */
	.assistant-header {
		display: flex; align-items: center; gap: 0.4rem;
		margin-bottom: 0.25rem;
	}
	.copy-btn {
		background: none; border: 1px solid var(--border); color: var(--text-muted);
		padding: 1px 8px; border-radius: 10px; font-size: 0.6rem; font-weight: 600;
		cursor: pointer; opacity: 0; transition: opacity 0.12s; font-family: var(--font-sans);
	}
	.message.assistant:hover .copy-btn { opacity: 1; }
	.copy-btn:hover { border-color: var(--text-secondary); color: var(--text-heading); }

	.assistant-content {
		color: var(--text-primary); font-size: 0.9rem; line-height: 1.7;
		padding-left: 0.1rem;
	}

	.reasoning-block {
		margin-bottom: 0.5rem; background: var(--bg-hover);
		border: 1px solid var(--border-light); border-radius: 10px;
		padding: 0.5rem 0.75rem; font-size: 0.82rem; color: var(--text-secondary);
	}
	.reasoning-block summary { cursor: pointer; font-weight: 600; font-size: 0.75rem; color: var(--text-muted); }

	.stream-error {
		display: flex; align-items: flex-start; gap: 0.75rem;
		margin-top: 0.5rem; padding: 0.6rem 0.75rem;
		background: color-mix(in srgb, var(--danger, #c43d3d) 8%, var(--bg-hover));
		border: 1px solid color-mix(in srgb, var(--danger, #c43d3d) 30%, transparent);
		border-left: 3px solid var(--danger, #c43d3d);
		border-radius: 8px;
		font-size: 0.8rem;
	}
	.stream-error-text { flex: 1; min-width: 0; line-height: 1.45; color: var(--text-primary); }
	.stream-error-text strong { color: var(--text-heading); margin-right: 0.4rem; }
	.stream-error-detail { color: var(--text-secondary); word-break: break-word; }
	.stream-error-retry {
		flex-shrink: 0;
		padding: 0.3rem 0.7rem;
		background: var(--bg-card); color: var(--text-primary);
		border: 1px solid var(--border); border-radius: 6px;
		font-family: var(--font-sans); font-size: 0.78rem; font-weight: 600;
		cursor: pointer; transition: all 0.12s;
	}
	.stream-error-retry:hover { background: var(--bg-hover); border-color: var(--text-secondary); }
	.stream-error-retry:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

	.thinking-indicator { padding: 0.25rem 0; }
	.thinking-dots { display: flex; gap: 4px; }
	.thinking-dots span {
		width: 6px; height: 6px; border-radius: 50%; background: var(--text-muted);
		animation: think 1.4s ease-in-out infinite;
	}
	.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
	.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
	@keyframes think { 0%, 80%, 100% { opacity: 0.2; } 40% { opacity: 1; } }

	.batch-progress { padding: 0.25rem 0.75rem 0.5rem; display: flex; align-items: center; gap: 0.5rem; }
	.batch-progress-bar { flex: 1; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
	.batch-progress-fill { height: 100%; background: var(--accent); border-radius: 2px; transition: width 0.5s ease; }
	.batch-progress-text { font-size: 0.68rem; color: var(--text-secondary); font-weight: 600; white-space: nowrap; font-variant-numeric: tabular-nums; font-family: var(--font-mono); }

	/* (old message-bar removed — replaced by composer) */

	.optimized-lock {
		display: flex; align-items: center; gap: 0.4rem; width: 100%;
		padding: 0.3rem 0.6rem; background: #fef7ed; border: 1px solid var(--accent-border);
		border-radius: 8px; font-size: 0.75rem;
	}
	.lock-badge {
		font-size: 0.58rem; font-weight: 700; background: var(--accent); color: white;
		padding: 1px 7px; border-radius: 6px; text-transform: uppercase; flex-shrink: 0;
		letter-spacing: 0.04em;
	}
	.lock-text { color: #92400e; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

	.paste-previews { display: flex; gap: 0.4rem; padding: 0.3rem 0; width: 100%; }
	.paste-thumb {
		position: relative; width: 48px; height: 48px;
		border: 1.5px solid var(--border); border-radius: 7px; overflow: hidden;
	}
	.paste-thumb img { width: 100%; height: 100%; object-fit: cover; }
	.paste-remove {
		position: absolute; top: -4px; right: -4px;
		width: 16px; height: 16px; border-radius: 50%;
		background: var(--danger); color: white; border: none;
		font-size: 0.6rem; cursor: pointer;
		display: flex; align-items: center; justify-content: center; line-height: 1;
	}

	.send-btn {
		width: 32px; height: 32px; border-radius: 50%; background: var(--text-heading);
		color: var(--bg-surface); border: none; cursor: pointer;
		display: flex; align-items: center; justify-content: center;
		font-size: 0.95rem; font-weight: 700; flex-shrink: 0;
		transition: all 0.1s;
	}
	.send-btn:hover:not(:disabled) { opacity: 0.8; }
	.send-btn:disabled { opacity: 0.15; cursor: not-allowed; }
	.send-icon { line-height: 1; }
	.spinning { animation: spin 0.8s linear infinite; }
	@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

	@media (max-width: 768px) {
		.messages-area { padding: 0.75rem; }
		.message { max-width: 98%; }
		.msg-body { padding: 0.5rem 0.65rem; font-size: 0.85rem; }
		.message-bar { padding: 0.4rem 0.5rem; gap: 0.3rem; }
		.bar-toggles { gap: 2px; }
		.toggle-btn { padding: 3px 7px; font-size: 0.6rem; }
		.empty-hints { flex-direction: column; align-items: center; }
	}
</style>
