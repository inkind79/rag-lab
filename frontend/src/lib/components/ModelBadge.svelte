<script lang="ts">
	let { model = '', displayName = '' }: { model: string; displayName: string } = $props();

	const providerColors: Record<string, string> = {
		microsoft: '#0078D4',
		google: '#4285F4',
		mistral: '#5546FF',
		agentica: '#FF5A5F',
		ibm: '#6b7280',
		alibaba: '#6B46C1',
		deepseek: '#008B7A',
		meta: '#8C9EFF',
		lg: '#A50034',
		default: '#9ca3af',
	};

	function getProvider(model: string): { name: string; color: string } {
		const m = model.toLowerCase();
		if (m.includes('gemma')) return { name: 'Google', color: providerColors.google };
		if (m.includes('phi') || m.includes('kosmos')) return { name: 'Microsoft', color: providerColors.microsoft };
		if (m.includes('llama') && !m.includes('ollama')) return { name: 'Meta', color: providerColors.meta };
		if (m.includes('mistral') || m.includes('magistral') || m.includes('devstral')) return { name: 'Mistral', color: providerColors.mistral };
		if (m.includes('deepcoder')) return { name: 'Agentica', color: providerColors.agentica };
		if (m.includes('granite')) return { name: 'IBM', color: providerColors.ibm };
		if (m.includes('qwen') || m.includes('qwq')) return { name: 'Alibaba', color: providerColors.alibaba };
		if (m.includes('deepseek')) return { name: 'DeepSeek', color: providerColors.deepseek };
		if (m.includes('exaone')) return { name: 'LG', color: providerColors.lg };
		if (m.includes('glm')) return { name: 'Zhipu', color: '#3b82f6' };
		return { name: '', color: providerColors.default };
	}

	function cleanName(display: string, m: string): string {
		let name = display || m;
		// Remove ollama- prefix
		if (name.startsWith('ollama-')) name = name.slice(7);
		// Remove "Vendor - " prefix
		const dash = name.indexOf(' - ');
		if (dash > 0 && dash < 20) name = name.slice(dash + 3);
		// Clean up common suffixes
		name = name.replace(/:latest$/, '');
		return name;
	}

	let provider = $derived(getProvider(model));
	let modelName = $derived(cleanName(displayName, model));
</script>

<span class="model-label">
	<span class="model-dot" style="background: {provider.color}"></span>
	<span class="model-name">{modelName}</span>
</span>

<style>
	.model-label {
		display: inline-flex; align-items: center; gap: 5px;
		font-size: 0.72rem; color: var(--text-muted);
	}
	.model-dot {
		width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
	}
	.model-name {
		font-weight: 500; font-family: var(--font-sans);
		white-space: nowrap;
	}
</style>
