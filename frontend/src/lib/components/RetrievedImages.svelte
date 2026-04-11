<script lang="ts">
	import type { RetrievedImage } from '$lib/stores/chat';

	let {
		images = [],
		showRelevance = false,
		relevantPaths = [],
		onrelevancechange,
	}: {
		images: RetrievedImage[];
		showRelevance?: boolean;
		relevantPaths?: string[];
		onrelevancechange?: (paths: string[]) => void;
	} = $props();

	let expanded = $state(true);
	let zoomSrc = $state('');
	let zoomScale = $state(1);
	let panX = $state(0);
	let panY = $state(0);
	let isPanning = $state(false);
	let panStart = { x: 0, y: 0 };

	function openZoom(src: string) {
		zoomSrc = src;
		zoomScale = 1;
		panX = 0;
		panY = 0;
	}

	function closeZoom() {
		zoomSrc = '';
		zoomScale = 1;
		panX = 0;
		panY = 0;
	}

	function handleZoomWheel(e: WheelEvent) {
		e.preventDefault();
		const delta = e.deltaY > 0 ? -0.15 : 0.15;
		zoomScale = Math.max(0.5, Math.min(3, zoomScale + delta));
	}

	function handleZoomKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') closeZoom();
		else if (e.key === '+' || e.key === '=') zoomScale = Math.min(3, zoomScale + 0.2);
		else if (e.key === '-') zoomScale = Math.max(0.5, zoomScale - 0.2);
		else if (e.key === '0') { zoomScale = 1; panX = 0; panY = 0; }
	}

	function handlePanStart(e: MouseEvent) {
		if (zoomScale <= 1) return;
		isPanning = true;
		panStart = { x: e.clientX - panX, y: e.clientY - panY };
	}

	function handlePanMove(e: MouseEvent) {
		if (!isPanning) return;
		panX = e.clientX - panStart.x;
		panY = e.clientY - panStart.y;
	}

	function handlePanEnd() {
		isPanning = false;
	}

	function getScoreColor(score: number): string {
		if (score >= 0.8) return '#22c55e';
		if (score >= 0.6) return '#eab308';
		if (score >= 0.4) return '#f97316';
		return '#ef4444';
	}

	function getImageUrl(img: RetrievedImage): string {
		const p = img.path || '';
		if (!p) return '';
		if (p.startsWith('/static/')) return p;
		if (p.startsWith('static/')) return `/${p}`;
		if (p.startsWith('images/')) return `/static/${p}`;
		return `/static/${p}`;
	}

	function getSourceName(images: RetrievedImage[]): string {
		for (const img of images) {
			const path = img.path || img.full_path || '';
			const basename = path.split('/').pop() || '';
			const withoutPrefix = basename.replace(/^\d+_/, '');
			if (withoutPrefix) return withoutPrefix.replace(/\.(png|jpg|jpeg)$/i, '');
		}
		return 'document';
	}

	function toggleRelevant(path: string) {
		let updated: string[];
		if (relevantPaths.includes(path)) {
			updated = relevantPaths.filter(p => p !== path);
		} else {
			updated = [...relevantPaths, path];
		}
		onrelevancechange?.(updated);
	}

	let imageItems = $derived(images.filter(i => i.result_type !== 'text'));
	let textItems = $derived(images.filter(i => i.result_type === 'text'));
	let sourceName = $derived(getSourceName(images));
	let bestMatchPath = $derived(imageItems.length > 1 ? imageItems.reduce((a, b) => a.score > b.score ? a : b).path : '');

	// Group images by document source (for batch mode)
	interface ImageGroup { name: string; images: RetrievedImage[] }
	let imageGroups = $derived.by(() => {
		const groups: ImageGroup[] = [];
		const bySource = new Map<string, RetrievedImage[]>();
		for (const img of imageItems) {
			const src = img.source || 'default';
			if (!bySource.has(src)) bySource.set(src, []);
			bySource.get(src)!.push(img);
		}
		if (bySource.size <= 1) return []; // No grouping needed
		for (const [name, imgs] of bySource) {
			groups.push({ name, images: imgs });
		}
		return groups;
	});
</script>

{#if images.length > 0}
	<div class="retrieved-section">
		{#if imageItems.length > 0}
			<button class="source-link" onclick={() => (expanded = !expanded)}>
				Retrieved Pages from {sourceName}
				<span class="expand-arrow">{expanded ? '▾' : '▸'}</span>
			</button>

			{#if expanded}
				{#if imageGroups.length > 0}
					{#each imageGroups as group}
						<div class="image-group">
							<div class="group-header">{group.name}</div>
							<div class="image-grid">
								{#each group.images as img}
									<div class="image-card">
										<div class="image-frame">
											<img src={getImageUrl(img)} alt="Retrieved page" loading="lazy" onclick={() => openZoom(getImageUrl(img))} onerror={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }} style="cursor: zoom-in;" />
											<div class="sim-hover-bar" style="background: linear-gradient(90deg, {getScoreColor(img.score)} {Math.round(img.score * 100)}%, transparent {Math.round(img.score * 100)}%)"></div>
											<div class="score-overlay" style="background-color: {getScoreColor(img.score)}">{img.score.toFixed(3)}</div>
										</div>
									</div>
								{/each}
							</div>
						</div>
					{/each}
				{:else}
				<div class="image-grid">
					{#each imageItems as img}
						<div class="image-card">
							<div class="image-frame">
								<img
									src={getImageUrl(img)}
									alt="Retrieved page"
									loading="lazy"
									onclick={() => openZoom(getImageUrl(img))}
									onerror={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
									style="cursor: zoom-in;"
								/>
								<div class="sim-hover-bar" style="background: linear-gradient(90deg, {getScoreColor(img.score)} {Math.round(img.score * 100)}%, transparent {Math.round(img.score * 100)}%)"></div>
								<div class="score-overlay" style="background-color: {getScoreColor(img.score)}">
									{img.score.toFixed(3)}
								</div>
								{#if img.path === bestMatchPath}
									<div class="best-match">Best</div>
								{/if}
								{#if showRelevance}
									<label class="relevance-overlay" class:checked={relevantPaths.includes(img.path)}>
										<input
											type="checkbox"
											checked={relevantPaths.includes(img.path)}
											onchange={() => toggleRelevant(img.path)}
										/>
										<span class="rel-label">Relevant</span>
									</label>
								{/if}
							</div>
							{#if img.text_preview}
								<div class="citation-snippet">
									<button class="snippet-toggle" onclick={(e) => {
										const card = (e.currentTarget as HTMLElement).parentElement;
										if (card) card.classList.toggle('open');
									}}>
										<span class="snippet-label">Page {img.page_num || '?'}</span>
										<span class="snippet-arrow">▸</span>
									</button>
									<div class="snippet-body">
										<p>{img.text_preview}</p>
									</div>
								</div>
							{/if}
						</div>
					{/each}
				</div>
				{/if}
			{/if}
		{/if}

		{#if textItems.length > 0}
			<div class="text-results">
				{#each textItems as item}
					<div class="text-chunk">
						<div class="chunk-header">
							<span class="chunk-source">{item.source || item.path || 'Text chunk'}</span>
							<span class="chunk-score" style="color: {getScoreColor(item.score)}">{item.score.toFixed(3)}</span>
						</div>
						{#if item.text_preview}
							<p class="chunk-preview">{item.text_preview}</p>
						{/if}
					</div>
				{/each}
			</div>
		{/if}
	</div>
{/if}

{#if zoomSrc}
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
	<div class="zoom-overlay"
		onclick={(e) => { if (!isPanning && e.target === e.currentTarget) closeZoom(); }}
		onkeydown={handleZoomKeydown}
		onwheel={handleZoomWheel}
		onmousedown={handlePanStart}
		onmousemove={handlePanMove}
		onmouseup={handlePanEnd}
		onmouseleave={handlePanEnd}
		role="dialog"
		tabindex="-1"
		style="cursor: {zoomScale > 1 ? (isPanning ? 'grabbing' : 'grab') : 'zoom-out'}"
	>
		<img
			src={zoomSrc}
			alt="Zoomed page"
			class="zoom-img"
			style="transform: scale({zoomScale}) translate({panX / zoomScale}px, {panY / zoomScale}px)"
			draggable="false"
		/>
		<div class="zoom-controls">
			<button onclick={(e) => { e.stopPropagation(); zoomScale = Math.min(3, zoomScale + 0.2); }}>+</button>
			<span class="zoom-level">{Math.round(zoomScale * 100)}%</span>
			<button onclick={(e) => { e.stopPropagation(); zoomScale = Math.max(0.5, zoomScale - 0.2); }}>-</button>
			<button onclick={(e) => { e.stopPropagation(); zoomScale = 1; panX = 0; panY = 0; }}>Reset</button>
		</div>
		<button class="zoom-close" onclick={closeZoom}>&times;</button>
	</div>
{/if}

<style>
	.retrieved-section {
		margin-top: 0.5rem;
		padding-top: 0.5rem;
	}

	.source-link {
		display: flex; align-items: center; gap: 0.35rem;
		background: none; border: none; color: var(--accent);
		font-size: 0.78rem; font-weight: 600; cursor: pointer;
		padding: 0.25rem 0; margin-bottom: 0.5rem;
	}
	.source-link:hover { text-decoration: underline; }
	.expand-arrow { font-size: 0.6rem; color: var(--text-muted); }

	.image-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; }
	.image-card { flex-shrink: 0; }
	.citation-snippet {
		width: 140px;
		margin-top: 0.25rem;
		background: var(--bg-hover);
		border: 1px solid var(--border);
		border-radius: 4px;
		overflow: hidden;
	}
	.snippet-toggle {
		display: flex;
		align-items: center;
		justify-content: space-between;
		width: 100%;
		padding: 0.25rem 0.4rem;
		background: none;
		border: none;
		cursor: pointer;
		color: var(--text-muted);
		font-size: 0.65rem;
		font-family: inherit;
	}
	.snippet-toggle:hover { color: var(--text-heading); }
	.snippet-label { font-weight: 600; }
	.snippet-arrow { transition: transform 0.2s; font-size: 0.55rem; display: inline-block; }
	:global(.citation-snippet.open) .snippet-arrow { transform: rotate(90deg); }
	.snippet-body {
		max-height: 0;
		overflow: hidden;
		transition: max-height 0.25s ease;
	}
	:global(.citation-snippet.open) .snippet-body {
		max-height: 300px;
		overflow-y: auto;
	}
	.snippet-body p {
		font-size: 0.68rem;
		line-height: 1.45;
		color: var(--text-muted);
		margin: 0;
		padding: 0 0.4rem 0.35rem;
		white-space: pre-wrap;
		word-break: break-word;
	}
	.image-group { margin-bottom: 0.5rem; }
	.group-header {
		font-size: 0.72rem; font-weight: 700; color: var(--text-heading);
		padding: 0.2rem 0; margin-bottom: 0.3rem;
		border-bottom: 1px solid var(--border);
	}

	.image-frame {
		position: relative; width: 140px; max-height: 200px;
		border: 2px solid var(--border); border-radius: 6px; overflow: hidden;
		background: var(--bg-hover); transition: border-color 0.2s, box-shadow 0.2s;
	}
	.image-frame:hover { border-color: #a5b4fc; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); }

	.image-frame img {
		display: block; width: 100%; height: auto;
		max-height: 196px; object-fit: cover; object-position: top;
	}

	.sim-hover-bar {
		position: absolute; bottom: 0; left: 0; right: 0;
		height: 4px; opacity: 0; transition: opacity 0.2s;
	}
	.image-frame:hover .sim-hover-bar { opacity: 1; height: 6px; }

	.score-overlay {
		position: absolute; bottom: 0; right: 0;
		color: white; font-size: 0.6rem; font-weight: 800;
		padding: 2px 6px; border-top-left-radius: 6px; letter-spacing: 0.02em;
	}

	.best-match {
		position: absolute; top: 0; left: 0; right: 0;
		background: rgba(34, 197, 94, 0.85); color: white;
		font-size: 0.55rem; font-weight: 800; text-align: center;
		padding: 1px 0; letter-spacing: 0.05em; text-transform: uppercase;
	}

	/* Relevance checkbox overlay */
	.relevance-overlay {
		position: absolute; top: 4px; left: 4px;
		display: flex; align-items: center; gap: 3px;
		background: rgba(0,0,0,0.5); border-radius: 4px;
		padding: 2px 6px; cursor: pointer;
		opacity: 0.4; transition: opacity 0.15s;
	}
	.image-frame:hover .relevance-overlay { opacity: 1; }
	.relevance-overlay.checked { opacity: 1; background: rgba(99, 102, 241, 0.8); }

	.relevance-overlay input { accent-color: var(--accent); width: 13px; height: 13px; cursor: pointer; }
	.rel-label { font-size: 0.58rem; color: white; font-weight: 600; user-select: none; }

	/* Text chunk results */
	.text-results { display: flex; flex-direction: column; gap: 0.5rem; }
	.text-chunk {
		background: var(--bg-hover); border: 1px solid var(--border);
		border-radius: 6px; padding: 0.5rem 0.75rem;
	}
	.chunk-header {
		display: flex; justify-content: space-between;
		align-items: center; margin-bottom: 0.25rem;
	}
	.chunk-source { font-size: 0.72rem; font-weight: 600; color: var(--text-secondary); }
	.chunk-score { font-size: 0.68rem; font-weight: 700; }
	.chunk-preview {
		margin: 0; font-size: 0.78rem; color: var(--text-heading);
		line-height: 1.4; max-height: 4.2em; overflow: hidden;
	}

	/* Lightbox zoom with advanced controls */
	.zoom-overlay {
		position: fixed; inset: 0; z-index: 2000;
		background: rgba(0,0,0,0.85);
		display: flex; align-items: center; justify-content: center;
		user-select: none;
	}
	.zoom-img {
		max-width: 90vw; max-height: 90vh;
		border-radius: 6px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
		object-fit: contain; transition: transform 0.1s ease-out;
		pointer-events: none;
	}
	.zoom-close {
		position: absolute; top: 1rem; right: 1.5rem;
		background: none; border: none; color: rgba(255,255,255,0.7);
		font-size: 2rem; cursor: pointer; line-height: 1;
	}
	.zoom-close:hover { color: white; }

	.zoom-controls {
		position: absolute; bottom: 1.5rem; left: 50%; transform: translateX(-50%);
		display: flex; align-items: center; gap: 0.5rem;
		background: rgba(0,0,0,0.6); padding: 0.35rem 0.75rem;
		border-radius: 20px; backdrop-filter: blur(4px);
	}
	.zoom-controls button {
		background: rgba(255,255,255,0.15); border: none; color: white;
		width: 28px; height: 28px; border-radius: 50%; cursor: pointer;
		font-size: 0.9rem; font-weight: 700;
		display: flex; align-items: center; justify-content: center;
	}
	.zoom-controls button:hover { background: rgba(255,255,255,0.3); }
	.zoom-level {
		font-size: 0.72rem; color: rgba(255,255,255,0.8); font-weight: 600;
		min-width: 3em; text-align: center; font-variant-numeric: tabular-nums;
	}
</style>
