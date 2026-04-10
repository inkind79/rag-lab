<script lang="ts">
	import { activeSessionId, sessions } from '$lib/stores/session';
	import { getIndexedFiles, uploadDocuments, saveSelectedDocs, getSessionId, deleteDocuments, getSessionData } from '$lib/api/client';
	import { toasts } from '$lib/stores/toast';

	interface IndexedFile {
		filename: string;
		file_type: string;
		page_count?: number;
	}

	let indexedFiles = $state<IndexedFile[]>([]);
	let selectedDocs = $state<string[]>([]);
	let uploading = $state(false);
	let dragOver = $state(false);
	let statusMsg = $state('');
	let collapsed = $state(typeof window !== 'undefined' ? localStorage.getItem('panel_collection_collapsed') === 'true' : false);
	const storedView = typeof window !== 'undefined' ? localStorage.getItem('panel_collection_view') : null;
	let viewMode = $state<'tiles' | 'list'>(storedView === 'list' ? 'list' : 'tiles');

	$effect(() => { if (typeof window !== 'undefined') localStorage.setItem('panel_collection_collapsed', String(collapsed)); });
	$effect(() => { if (typeof window !== 'undefined') localStorage.setItem('panel_collection_view', viewMode); });

	$effect(() => {
		if ($activeSessionId) {
			// Clear stale data immediately on session switch, then reload
			indexedFiles = [];
			selectedDocs = [];
			loadDocs();
		}
	});

	function inferFileType(filename: string): string {
		const ext = filename.split('.').pop()?.toLowerCase() || '';
		if (ext === 'pdf') return 'pdf';
		if (['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'].includes(ext)) return 'image';
		return 'unknown';
	}

	async function loadDocs() {
		if (!$activeSessionId) return;
		try {
			// Get indexed files list
			const result = await getIndexedFiles($activeSessionId);
			if (result) {
				const files = (result.indexed_files || result.files || []) as any[];
				// Ensure file_type is set (backend may not send it)
				indexedFiles = files.map(f => ({
					filename: f.filename,
					file_type: f.file_type || inferFileType(f.filename || ''),
					page_count: f.page_count,
				}));
				// Update the sessions store so hasDocs reactivity works across components
				const sid = $activeSessionId;
				sessions.update(list => list.map(s =>
					s.session_id === sid ? { ...s, indexed_files: indexedFiles } : s
				));
			}
			// Get selected_docs from session data (not in indexed_files response)
			try {
				const resp = await getSessionData();
				const sd = resp?.session_data || resp;
				selectedDocs = (sd?.selected_docs || []) as string[];
			} catch {}
		} catch {}
	}

	let uploadElapsed = $state(0);
	let uploadFileCount = $state(0);
	let uploadTimer: ReturnType<typeof setInterval> | null = null;

	async function handleUpload(files: FileList | null) {
		if (!files || files.length === 0 || !$activeSessionId) return;
		uploading = true;
		uploadFileCount = files.length;
		uploadElapsed = 0;
		statusMsg = '';
		uploadTimer = setInterval(() => { uploadElapsed++; }, 1000);
		try {
			const result = await uploadDocuments(files, $activeSessionId);
			if (result.success !== false) {
				statusMsg = '';
				toasts.success(`${uploadFileCount} document${uploadFileCount > 1 ? 's' : ''} indexed in ${uploadElapsed}s`);
				await loadDocs();
			} else {
				statusMsg = (result as any).message || 'Failed';
				toasts.error(statusMsg);
			}
		} catch (e: unknown) {
			statusMsg = `Error: ${e instanceof Error ? e.message : String(e)}`;
			toasts.error(statusMsg);
		} finally {
			uploading = false;
			if (uploadTimer) { clearInterval(uploadTimer); uploadTimer = null; }
			setTimeout(() => (statusMsg = ''), 3000);
		}
	}

	function handleFileInput(e: Event) {
		const input = e.target as HTMLInputElement;
		handleUpload(input.files);
		input.value = '';
	}

	function handleDrop(e: DragEvent) {
		e.preventDefault();
		dragOver = false;
		handleUpload(e.dataTransfer?.files ?? null);
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

	async function handleDelete(filename: string) {
		try {
			await deleteDocuments([filename]);
			toasts.success('Document deleted');
			await loadDocs();
		} catch {
			toasts.error('Failed to delete document');
		}
	}

	async function handleDeleteSelected() {
		if (selectedDocs.length === 0) return;
		try {
			await deleteDocuments(selectedDocs);
			toasts.success(`${selectedDocs.length} document${selectedDocs.length > 1 ? 's' : ''} deleted`);
			selectedDocs = [];
			await loadDocs();
		} catch {
			toasts.error('Failed to delete documents');
		}
	}

	function getFileIcon(ft: string): { icon: string; color: string } {
		switch (ft) {
			case 'pdf': return { icon: 'PDF', color: 'var(--danger)' };
			case 'image': return { icon: 'IMG', color: 'var(--accent)' };
			default: return { icon: 'DOC', color: 'var(--text-secondary)' };
		}
	}

	let allSelected = $derived(indexedFiles.length > 0 && selectedDocs.length === indexedFiles.length);
	let someSelected = $derived(selectedDocs.length > 0 && selectedDocs.length < indexedFiles.length);
</script>

<div class="panel-section">
	<button class="section-header" onclick={() => (collapsed = !collapsed)}>
		<span class="section-title">Documents</span>
		<div class="section-actions">
			{#if statusMsg}
				<span class="status-badge">{statusMsg}</span>
			{/if}
			{#if !collapsed}
				<span class="view-toggles" onclick={(e) => e.stopPropagation()}>
					<button class="vt" class:active={viewMode === 'tiles'} onclick={() => (viewMode = 'tiles')} title="Tile view">&#9638;</button>
					<button class="vt" class:active={viewMode === 'list'} onclick={() => (viewMode = 'list')} title="List view">&#9776;</button>
				</span>
			{/if}
			<span class="collapse-icon">{collapsed ? '▸' : '▾'}</span>
		</div>
	</button>

	{#if !collapsed}
		<div class="section-body">
			<!-- Upload zone -->
			<div
				class="upload-zone"
				class:drag-over={dragOver}
				class:uploading
				ondrop={handleDrop}
				ondragover={(e) => { e.preventDefault(); dragOver = true; }}
				ondragleave={() => (dragOver = false)}
				role="button"
				tabindex="0"
			>
				{#if dragOver}
					<div class="drop-overlay">
						<div class="drop-icon">&#9729;</div>
						<span>Drop files to upload</span>
					</div>
				{:else if uploading}
					<div class="upload-progress">
						<div class="progress-pulse"></div>
						<div class="progress-info">
							<span class="progress-title">Indexing {uploadFileCount} file{uploadFileCount > 1 ? 's' : ''}</span>
							<span class="progress-elapsed">{Math.floor(uploadElapsed / 60)}:{String(uploadElapsed % 60).padStart(2, '0')}</span>
						</div>
					</div>
				{:else}
					<label class="upload-label">
						<span class="upload-plus">📎</span>
						<span class="upload-text">Attach Documents</span>
						<input type="file" multiple accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp" onchange={handleFileInput} />
					</label>
				{/if}
			</div>

			<!-- Documents -->
			{#if indexedFiles.length > 0}
				{#if viewMode === 'tiles'}
					<div class="doc-tiles">
						{#each indexedFiles as file}
							{@const icon = getFileIcon(file.file_type)}
							<div
								class="doc-tile"
								class:selected={selectedDocs.includes(file.filename)}
								onclick={() => toggleDoc(file.filename)}
								role="button"
								tabindex="0"
							>
								{#if selectedDocs.includes(file.filename)}
									<div class="tile-check">✓</div>
								{/if}
								<div class="tile-icon" style="background-color: {icon.color}">{icon.icon}</div>
								<div class="tile-name" title={file.filename}>{file.filename}</div>
								{#if file.page_count}
									<div class="tile-pages">{file.page_count}p</div>
								{/if}
								<button class="tile-delete" onclick={(e) => { e.stopPropagation(); handleDelete(file.filename); }} title="Delete">×</button>
							</div>
						{/each}
					</div>
				{:else}
					<div class="doc-list">
						{#each indexedFiles as file}
							{@const icon = getFileIcon(file.file_type)}
							<div class="doc-list-item" class:selected={selectedDocs.includes(file.filename)} onclick={() => toggleDoc(file.filename)} role="button" tabindex="0">
								<span class="list-icon" style="background-color: {icon.color}">{icon.icon}</span>
								<span class="list-name" title={file.filename}>{file.filename}</span>
								{#if file.page_count}<span class="list-pages">{file.page_count}p</span>{/if}
								{#if file.file_type === 'pdf'}
									<a class="list-view" href="/document/view/{$activeSessionId}/{file.filename}" target="_blank" onclick={(e) => e.stopPropagation()} title="View PDF">View</a>
								{/if}
								<button class="list-del" onclick={(e) => { e.stopPropagation(); handleDelete(file.filename); }} title="Delete">×</button>
							</div>
						{/each}
					</div>
				{/if}

				<div class="doc-actions">
					<label class="select-all-label">
						<input type="checkbox"
							checked={allSelected}
							indeterminate={someSelected}
							onchange={() => { allSelected ? deselectAll() : selectAll(); }}
						/>
						<span>{allSelected ? 'Deselect' : 'Select All'}</span>
					</label>
					{#if selectedDocs.length > 0}
						<span class="doc-counter">{selectedDocs.length} / {indexedFiles.length}</span>
						<button class="link-btn danger" onclick={handleDeleteSelected}>
							Delete selected
						</button>
					{/if}
				</div>
			{:else if !uploading}
				<p class="empty-hint">No documents yet</p>
			{/if}
		</div>
	{/if}
</div>

<style>
	.panel-section { border-bottom: 1px solid var(--border); }

	.section-header {
		display: flex; justify-content: space-between; align-items: center;
		padding: 0.6rem 0.75rem; background: none; border: none; width: 100%;
		cursor: pointer; color: var(--text-heading); font-weight: 700; font-size: 0.82rem;
		text-align: left;
	}
	.section-header:hover { background: var(--bg-hover); }
	.section-title { letter-spacing: 0.02em; }
	.section-actions { display: flex; align-items: center; gap: 0.5rem; }
	.status-badge {
		font-size: 0.65rem; font-weight: 600; color: var(--accent);
		background: var(--accent-bg); padding: 1px 6px; border-radius: 8px;
	}
	.collapse-icon { font-size: 0.7rem; color: var(--text-muted); }

	.section-body { padding: 0 0.75rem 0.75rem; }

	/* Upload zone */
	.upload-zone {
		border: 1.5px dashed var(--border); border-radius: 8px; padding: 0.6rem;
		text-align: center; transition: all 0.2s; margin-bottom: 0.6rem;
		cursor: pointer;
	}
	.upload-zone:hover { border-color: var(--text-muted); background: var(--bg-hover); }
	.upload-zone.drag-over { border-color: var(--accent); background: rgba(107, 159, 200, 0.08); min-height: 80px; }

	.drop-overlay {
		display: flex; flex-direction: column; align-items: center; justify-content: center;
		gap: 0.25rem; padding: 0.5rem; color: var(--accent); pointer-events: none;
	}
	.drop-icon { font-size: 1.8rem; opacity: 0.7; }
	.drop-overlay span { font-size: 0.78rem; font-weight: 600; }
	.upload-zone.uploading { border-color: var(--accent); }

	.upload-label {
		cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 0.4rem;
	}
	.upload-label input { display: none; }
	.upload-plus { font-size: 0.9rem; }
	.upload-text { font-size: 0.78rem; color: var(--text-secondary); }

	.upload-progress {
		display: flex; flex-direction: column; align-items: center; gap: 0.5rem;
		padding: 0.5rem 0;
	}
	.progress-pulse {
		width: 100%; height: 3px; border-radius: 2px; background: var(--border);
		overflow: hidden; position: relative;
	}
	.progress-pulse::after {
		content: ''; position: absolute; inset: 0;
		background: var(--text-heading);
		animation: pulse-slide 1.5s ease-in-out infinite;
		border-radius: 2px;
	}
	@keyframes pulse-slide {
		0% { transform: translateX(-100%); width: 40%; }
		50% { transform: translateX(60%); width: 40%; }
		100% { transform: translateX(200%); width: 40%; }
	}
	.progress-info { display: flex; justify-content: space-between; width: 100%; }
	.progress-title { font-weight: 600; color: var(--text-heading); font-size: 0.74rem; }
	.progress-elapsed { font-size: 0.72rem; color: var(--text-muted); font-family: var(--font-mono); }
	.spinner {
		width: 14px; height: 14px; border: 2px solid var(--border); border-top-color: var(--accent);
		border-radius: 50%; animation: spin 0.6s linear infinite;
	}
	@keyframes spin { to { transform: rotate(360deg); } }

	/* Document tiles — grid of cards matching Flask UI */
	.doc-tiles {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
		gap: 0.5rem;
	}

	.doc-tile {
		position: relative;
		background: var(--bg-surface); border: 1.5px solid var(--border); border-radius: 8px;
		padding: 0.5rem; cursor: pointer; text-align: center;
		transition: all 0.15s; min-height: 80px;
		display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 0.25rem;
	}
	.doc-tile:hover { border-color: var(--accent-light); box-shadow: 0 2px 8px rgba(99, 102, 241, 0.1); }
	.doc-tile.selected { border-color: var(--accent); background: var(--accent-bg); }

	.tile-check {
		position: absolute; top: 4px; right: 4px;
		width: 18px; height: 18px; background: var(--accent); color: white;
		border-radius: 50%; font-size: 0.6rem; font-weight: 700;
		display: flex; align-items: center; justify-content: center;
	}

	.tile-icon {
		width: 32px; height: 22px; border-radius: 4px;
		color: white; font-size: 0.55rem; font-weight: 800;
		display: flex; align-items: center; justify-content: center;
		letter-spacing: 0.03em;
	}

	.tile-name {
		font-size: 0.65rem; color: var(--text-heading); line-height: 1.2;
		overflow: hidden; text-overflow: ellipsis; display: -webkit-box;
		-webkit-line-clamp: 2; -webkit-box-orient: vertical;
		word-break: break-all; max-width: 100%;
	}

	.tile-pages { font-size: 0.55rem; color: var(--text-muted); }

	.tile-delete {
		position: absolute; top: 2px; left: 4px;
		background: none; border: none; color: var(--border); font-size: 0.85rem;
		cursor: pointer; opacity: 0; transition: opacity 0.15s; line-height: 1;
	}
	.doc-tile:hover .tile-delete { opacity: 1; }
	.tile-delete:hover { color: var(--danger); }

	/* View toggle buttons */
	.view-toggles { display: flex; gap: 1px; }
	.vt {
		background: none; border: 1px solid var(--border); color: var(--text-muted);
		width: 18px; height: 18px; font-size: 0.6rem; cursor: pointer;
		display: flex; align-items: center; justify-content: center;
		padding: 0; line-height: 1;
	}
	.vt:first-child { border-radius: 3px 0 0 3px; }
	.vt:last-child { border-radius: 0 3px 3px 0; }
	.vt.active { background: var(--accent); color: white; border-color: var(--accent); }

	/* List view */
	.doc-list { display: flex; flex-direction: column; gap: 2px; }
	.doc-list-item {
		display: flex; align-items: center; gap: 0.4rem;
		padding: 0.3rem 0.4rem; border-radius: 5px; cursor: pointer;
		font-size: 0.75rem; color: var(--text-heading);
	}
	.doc-list-item:hover { background: var(--bg-hover); }
	.doc-list-item.selected { background: var(--accent-bg); }

	.list-icon {
		width: 20px; height: 14px; border-radius: 2px; color: white;
		font-size: 0.45rem; font-weight: 800; display: flex;
		align-items: center; justify-content: center; flex-shrink: 0;
	}
	.list-name {
		flex: 1; min-width: 0; overflow: hidden;
		text-overflow: ellipsis; white-space: nowrap;
	}
	.list-pages { font-size: 0.6rem; color: var(--text-muted); flex-shrink: 0; }
	.list-view {
		font-size: 0.6rem; color: var(--accent); text-decoration: none;
		flex-shrink: 0;
	}
	.list-view:hover { text-decoration: underline; }
	.list-del {
		background: none; border: none; color: var(--border); font-size: 0.8rem;
		cursor: pointer; opacity: 0; transition: opacity 0.1s; line-height: 1;
		flex-shrink: 0; padding: 0;
	}
	.doc-list-item:hover .list-del { opacity: 1; }
	.list-del:hover { color: var(--danger); }

	.doc-actions { margin-top: 0.4rem; display: flex; align-items: center; justify-content: center; gap: 0.5rem; }
	.select-all-label {
		display: flex; align-items: center; gap: 0.3rem; cursor: pointer;
		font-size: 0.72rem; font-weight: 600; color: var(--text-secondary);
	}
	.select-all-label input { accent-color: var(--accent); width: 13px; height: 13px; }
	.doc-counter { font-size: 0.65rem; color: var(--text-muted); font-variant-numeric: tabular-nums; }
	.link-btn {
		background: none; border: none; color: var(--accent); font-size: 0.72rem;
		cursor: pointer; font-weight: 600;
	}
	.link-btn:hover { text-decoration: underline; }
	.link-btn.danger { color: var(--danger); }

	.empty-hint { text-align: center; color: var(--text-muted); font-size: 0.78rem; margin: 0.75rem 0; }
</style>
