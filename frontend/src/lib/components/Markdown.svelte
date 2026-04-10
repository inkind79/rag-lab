<script lang="ts">
	import { marked } from 'marked';
	import DOMPurify from 'dompurify';

	let { content = '', streaming = false }: { content: string; streaming?: boolean } = $props();

	// Configure marked for safe rendering
	marked.setOptions({
		breaks: true,
		gfm: true,
	});

	function processLatex(text: string): string {
		// Convert \text{...} to plain text
		text = text.replace(/\\text\{([^}]+)\}/g, '$1');
		// Convert $$...$$ block math to styled divs
		text = text.replace(/\$\$([^$]+)\$\$/g, '<div class="math-block">$1</div>');
		// Convert $...$ inline math to styled spans (avoid matching $$)
		text = text.replace(/(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)/g, '<span class="math-inline">$1</span>');
		return text;
	}

	function renderMarkdown(text: string): string {
		if (!text) return '';
		text = processLatex(text);
		const raw = marked.parse(text, { async: false }) as string;
		return DOMPurify.sanitize(raw, {
			ADD_ATTR: ['class'],
		});
	}

	let html = $derived(renderMarkdown(content));

	let contentNode: HTMLElement;

	// Add copy buttons to <pre> blocks only when NOT streaming.
	// During streaming, innerHTML swaps on every flush would cause the
	// MutationObserver to destroy + recreate wrappers each time — DOM storm.
	$effect(() => {
		// Subscribe to html so this re-runs when content changes
		if (html && contentNode && !streaming) {
			// Tick ensures the DOM has the latest {@html} applied
			requestAnimationFrame(() => injectCopyButtons(contentNode));
		}
	});

	function injectCopyButtons(node: HTMLElement) {
		node.querySelectorAll('pre').forEach((pre) => {
			if (pre.querySelector('.code-copy-btn')) return;
			const wrapper = document.createElement('div');
			wrapper.style.position = 'relative';
			pre.parentNode?.insertBefore(wrapper, pre);
			wrapper.appendChild(pre);

			const btn = document.createElement('button');
			btn.className = 'code-copy-btn';
			btn.textContent = 'Copy';
			btn.onclick = async () => {
				const code = pre.querySelector('code')?.textContent || pre.textContent || '';
				try {
					await navigator.clipboard.writeText(code);
					btn.textContent = 'Copied!';
					btn.classList.add('copied');
					setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
				} catch {
					btn.textContent = 'Failed';
					setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
				}
			};
			wrapper.appendChild(btn);
		});
	}
</script>

<div class="markdown-content" bind:this={contentNode}>
	{@html html}
</div>

<style>
	.markdown-content {
		line-height: 1.65;
		word-wrap: break-word;
	}

	.markdown-content :global(h1),
	.markdown-content :global(h2),
	.markdown-content :global(h3) {
		color: var(--text-heading, #dcdee3);
		margin: 1rem 0 0.5rem;
		border-bottom: 1px solid var(--border, #33343a);
		padding-bottom: 0.3rem;
	}
	.markdown-content :global(h1) { font-size: 1.3rem; }
	.markdown-content :global(h2) { font-size: 1.15rem; }
	.markdown-content :global(h3) { font-size: 1rem; }

	.markdown-content :global(p) {
		margin: 0.4rem 0;
	}

	.markdown-content :global(ul),
	.markdown-content :global(ol) {
		padding-left: 1.5rem;
		margin: 0.4rem 0;
	}

	.markdown-content :global(li) {
		margin: 0.2rem 0;
	}

	.markdown-content :global(strong) {
		color: var(--text-heading, #dcdee3);
	}

	.markdown-content :global(code) {
		background: var(--bg-active, #2e2f35);
		padding: 2px 6px;
		border-radius: 4px;
		font-size: 0.85em;
		color: var(--accent-light, #7eb3d6);
		border: 1px solid var(--border-light, #2a2b30);
	}

	.markdown-content :global(pre) {
		background: var(--bg-input, #222328);
		border: 1px solid var(--border, #33343a);
		border-radius: 8px;
		padding: 0.75rem;
		overflow-x: auto;
		margin: 0.5rem 0;
		box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
	}

	.markdown-content :global(pre code) {
		background: none;
		padding: 0;
	}

	.markdown-content :global(.code-copy-btn) {
		position: absolute; top: 4px; right: 4px;
		background: var(--bg-active, #2e2f35); border: 1px solid var(--border, #33343a); color: var(--text-muted, #55565e);
		padding: 2px 8px; border-radius: 4px; font-size: 0.65rem;
		font-weight: 600; cursor: pointer; opacity: 0;
		transition: opacity 0.15s;
	}
	.markdown-content :global(div:hover > .code-copy-btn) { opacity: 1; }
	.markdown-content :global(.code-copy-btn:hover) { background: var(--bg-hover, #26272c); color: var(--text-heading, #dcdee3); }
	.markdown-content :global(.code-copy-btn.copied) { background: var(--success, #5cad6f); color: white; }

	.markdown-content :global(blockquote) {
		border-left: 3px solid var(--accent, #6b9fc8);
		padding-left: 0.75rem;
		margin: 0.5rem 0;
		color: var(--text-secondary, #8b8d95);
	}

	.markdown-content :global(hr) {
		border: none;
		border-top: 1px solid var(--border, #33343a);
		margin: 1rem 0;
	}

	.markdown-content :global(table) {
		border-collapse: collapse;
		width: 100%;
		margin: 0.5rem 0;
	}

	.markdown-content :global(th),
	.markdown-content :global(td) {
		border: 1px solid var(--border, #33343a);
		padding: 0.4rem 0.6rem;
		text-align: left;
		font-size: 0.85rem;
	}

	.markdown-content :global(th) {
		background: var(--bg-active, #2e2f35);
		color: var(--text-heading, #dcdee3);
		font-weight: 600;
	}

	.markdown-content :global(a) {
		color: var(--accent-light, #7eb3d6);
		text-decoration: none;
	}
	.markdown-content :global(a:hover) {
		text-decoration: underline;
	}

	.markdown-content :global(.math-inline) {
		font-family: 'Cambria Math', 'Times New Roman', serif;
		font-style: italic; color: var(--accent, #6b9fc8); padding: 0 2px;
	}
	.markdown-content :global(.math-block) {
		font-family: 'Cambria Math', 'Times New Roman', serif;
		font-style: italic; color: var(--accent, #6b9fc8);
		text-align: center; padding: 0.5rem; margin: 0.5rem 0;
		background: var(--bg-input, #222328); border-radius: 6px;
		border: 1px solid var(--border-light, #2a2b30);
	}
</style>
