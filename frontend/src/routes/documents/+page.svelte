<script lang="ts">
	import { onMount } from 'svelte';
	import { activeSessionId } from '$lib/stores/session';
	import {
		getIndexedFiles,
		uploadDocuments,
		saveSelectedDocs,
		reindexDocuments,
		getSessionId,
	} from '$lib/api/client';

	interface IndexedFile {
		filename: string;
		file_type: string;
		page_count?: number;
	}

	let indexedFiles = $state<IndexedFile[]>([]);
	let selectedDocs = $state<string[]>([]);
	let uploading = $state(false);
	let reindexing = $state(false);
	let statusMessage = $state('');
	let dragOver = $state(false);

	onMount(async () => {
		if ($activeSessionId) {
			await loadDocuments();
		}
	});

	$effect(() => {
		if ($activeSessionId) {
			loadDocuments();
		}
	});

	async function loadDocuments() {
		if (!$activeSessionId) return;
		try {
			const result = await getIndexedFiles($activeSessionId);
			if (result.data) {
				indexedFiles = (result.data.indexed_files || []) as IndexedFile[];
				selectedDocs = (result.data.selected_docs || []) as string[];
			}
		} catch (e) {
			console.error('Failed to load documents:', e);
		}
	}

	async function handleUpload(files: FileList | null) {
		if (!files || files.length === 0) return;
		uploading = true;
		statusMessage = `Uploading ${files.length} file(s)...`;

		try {
			const result = await uploadDocuments(files, $activeSessionId || '');
			const data = result.data as Record<string, any> | undefined;
			if (data) {
				indexedFiles = (data.indexed_files || []) as IndexedFile[];
				selectedDocs = (data.selected_docs || []) as string[];
				statusMessage = data.message || `${files.length} file(s) indexed`;
			}
		} catch (e: unknown) {
			statusMessage = `Upload failed: ${e instanceof Error ? e.message : String(e)}`;
		} finally {
			uploading = false;
			setTimeout(() => (statusMessage = ''), 4000);
		}
	}

	async function handleFileInput(event: Event) {
		const input = event.target as HTMLInputElement;
		await handleUpload(input.files);
		input.value = ''; // Reset for re-upload of same file
	}

	async function handleDrop(event: DragEvent) {
		event.preventDefault();
		dragOver = false;
		await handleUpload(event.dataTransfer?.files ?? null);
	}

	function handleDragOver(event: DragEvent) {
		event.preventDefault();
		dragOver = true;
	}

	function handleDragLeave() {
		dragOver = false;
	}

	async function toggleDoc(filename: string) {
		if (selectedDocs.includes(filename)) {
			selectedDocs = selectedDocs.filter((d) => d !== filename);
		} else {
			selectedDocs = [...selectedDocs, filename];
		}
		await saveSelectedDocs(selectedDocs);
	}

	async function selectAll() {
		selectedDocs = indexedFiles.map((f) => f.filename);
		await saveSelectedDocs(selectedDocs);
	}

	async function deselectAll() {
		selectedDocs = [];
		await saveSelectedDocs(selectedDocs);
	}

	async function handleReindex() {
		reindexing = true;
		statusMessage = 'Reindexing documents...';
		try {
			const result = await reindexDocuments();
			const rdata = result.data as Record<string, any> | undefined;
			statusMessage = rdata?.message || 'Reindex complete';
		} catch (e: unknown) {
			statusMessage = `Reindex failed: ${e instanceof Error ? e.message : String(e)}`;
		} finally {
			reindexing = false;
			setTimeout(() => (statusMessage = ''), 4000);
		}
	}

	function getFileIcon(fileType: string): string {
		switch (fileType) {
			case 'pdf': return '📄';
			case 'image': return '🖼️';
			default: return '📎';
		}
	}
</script>

<div class="documents-page">
	<header class="page-header">
		<h2>Documents</h2>
		<div class="header-actions">
			{#if statusMessage}
				<span class="status-message">{statusMessage}</span>
			{/if}
			{#if indexedFiles.length > 0}
				<button class="action-btn secondary" onclick={handleReindex} disabled={reindexing}>
					{reindexing ? 'Reindexing...' : 'Reindex All'}
				</button>
			{/if}
		</div>
	</header>

	<!-- Upload Zone -->
	<div
		class="upload-zone"
		class:drag-over={dragOver}
		class:uploading
		ondrop={handleDrop}
		ondragover={handleDragOver}
		ondragleave={handleDragLeave}
		role="button"
		tabindex="0"
	>
		{#if uploading}
			<div class="upload-spinner">Indexing documents...</div>
		{:else}
			<div class="upload-content">
				<span class="upload-icon">+</span>
				<p>Drag & drop files here, or <label class="file-label">
					browse
					<input type="file" multiple accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp" onchange={handleFileInput} />
				</label></p>
				<span class="upload-hint">PDF, PNG, JPG, TIFF, BMP</span>
			</div>
		{/if}
	</div>

	<!-- Document List -->
	{#if indexedFiles.length > 0}
		<div class="doc-list-header">
			<span class="doc-count">{indexedFiles.length} document{indexedFiles.length === 1 ? '' : 's'}</span>
			<div class="selection-actions">
				<button class="text-btn" onclick={selectAll}>Select All</button>
				<button class="text-btn" onclick={deselectAll}>Deselect All</button>
			</div>
		</div>

		<div class="doc-list">
			{#each indexedFiles as file}
				<label class="doc-item" class:selected={selectedDocs.includes(file.filename)}>
					<input
						type="checkbox"
						checked={selectedDocs.includes(file.filename)}
						onchange={() => toggleDoc(file.filename)}
					/>
					<span class="doc-icon">{getFileIcon(file.file_type)}</span>
					<div class="doc-info">
						<span class="doc-name">{file.filename}</span>
						<span class="doc-meta">
							{file.file_type.toUpperCase()}
							{#if file.page_count}
								 · {file.page_count} page{file.page_count === 1 ? '' : 's'}
							{/if}
						</span>
					</div>
				</label>
			{/each}
		</div>
	{:else}
		<div class="empty-state">
			<p>No documents uploaded yet</p>
			<p class="empty-hint">Upload PDFs or images to start querying with RAG</p>
		</div>
	{/if}
</div>

<style>
	.documents-page {
		padding: 1.5rem 2rem;
		max-width: 900px;
		overflow-y: auto;
		height: 100%;
	}

	.page-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 1.5rem;
		padding-bottom: 1rem;
		border-bottom: 1px solid #e5e7eb;
	}
	.page-header h2 { margin: 0; font-size: 1.4rem; color: #1f2937; }

	.header-actions {
		display: flex;
		align-items: center;
		gap: 1rem;
	}

	.status-message {
		font-size: 0.85rem;
		color: #22c55e;
	}

	.action-btn {
		padding: 0.4rem 1rem;
		border-radius: 6px;
		font-size: 0.8rem;
		font-weight: 600;
		cursor: pointer;
		border: none;
	}
	.action-btn.secondary {
		background: #e5e7eb;
		color: #1f2937;
	}
	.action-btn.secondary:hover:not(:disabled) {
		background: #d1d5db;
	}
	.action-btn:disabled { opacity: 0.5; }

	/* Upload zone */
	.upload-zone {
		border: 2px dashed #e5e7eb;
		border-radius: 12px;
		padding: 2rem;
		text-align: center;
		transition: all 0.2s;
		margin-bottom: 1.5rem;
	}
	.upload-zone.drag-over {
		border-color: #6366f1;
		background: rgba(99, 102, 241, 0.05);
	}
	.upload-zone.uploading {
		border-color: #6366f1;
		background: rgba(99, 102, 241, 0.03);
	}

	.upload-content p { margin: 0.5rem 0; color: #9ca3af; font-size: 0.9rem; }
	.upload-icon { font-size: 2rem; color: #4f46e5; }
	.upload-hint { font-size: 0.75rem; color: #9ca3af; }
	.upload-spinner { color: #6366f1; font-size: 0.9rem; padding: 1rem; }

	.file-label {
		color: #6366f1;
		cursor: pointer;
		text-decoration: underline;
	}
	.file-label input { display: none; }

	/* Document list */
	.doc-list-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.75rem;
	}
	.doc-count { font-size: 0.85rem; color: #9ca3af; }
	.selection-actions { display: flex; gap: 0.5rem; }
	.text-btn {
		background: none;
		border: none;
		color: #6366f1;
		font-size: 0.8rem;
		cursor: pointer;
		padding: 2px 6px;
	}
	.text-btn:hover { text-decoration: underline; }

	.doc-list {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.doc-item {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.6rem 0.75rem;
		border-radius: 8px;
		border: 1px solid #e5e7eb;
		cursor: pointer;
		transition: all 0.15s;
	}
	.doc-item:hover { border-color: #d1d5db; }
	.doc-item.selected {
		border-color: #6366f1;
		background: rgba(99, 102, 241, 0.05);
	}
	.doc-item input[type='checkbox'] {
		accent-color: #6366f1;
		width: 16px;
		height: 16px;
	}
	.doc-icon { font-size: 1.2rem; }
	.doc-info { display: flex; flex-direction: column; }
	.doc-name { font-size: 0.85rem; color: #1f2937; }
	.doc-meta { font-size: 0.7rem; color: #9ca3af; }

	.empty-state {
		text-align: center;
		padding: 3rem 1rem;
		color: #9ca3af;
	}
	.empty-state p { margin: 0.25rem 0; }
	.empty-hint { font-size: 0.8rem; }
</style>
