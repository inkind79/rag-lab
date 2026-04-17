<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { goto } from '$app/navigation';
	import { sessions, activeSessionId } from '$lib/stores/session';
	import { messages, chatHistoryCursor } from '$lib/stores/chat';
	import {
		checkAuth, listSessions, getSessionData, getSessionById, switchSession as apiSwitchSession,
		createNewSession, setSessionId, getSessionId, logout as apiLogout,
		renameSession as apiRenameSession, deleteSession as apiDeleteSession,
		deleteAllSessions as apiDeleteAllSessions,
		getIndexedFiles, cleanupMemory, emergencyMemoryCleanup,
		getChatHistoryPage,
	} from '$lib/api/client';
	import DocumentPanel from '$lib/components/DocumentPanel.svelte';
	import TemplatePanel from '$lib/components/TemplatePanel.svelte';
	import SettingsModal from '$lib/components/SettingsModal.svelte';
	import ToastContainer from '$lib/components/ToastContainer.svelte';
	import { toasts } from '$lib/stores/toast';

	let { children } = $props();

	// Theme toggle — persisted in localStorage, overrides prefers-color-scheme
	let darkMode = $state(true);
	if (typeof window !== 'undefined') {
		const saved = localStorage.getItem('theme');
		darkMode = saved ? saved === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
	}
	$effect(() => {
		if (typeof document !== 'undefined') {
			document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
			localStorage.setItem('theme', darkMode ? 'dark' : 'light');
		}
	});

	let settingsOpen = $state(false);
	let sidebarCollapsed = $state(false);
	let centerPanelOpen = $state(false);
	let memoryPercent = $state(0);
	let authenticated = $state(false);
	let loading = $state(true);
	let switchingSession = $state(false);
	let switchingName = $state('');
	let username = $state('');

	// Session management state
	let menuOpenId = $state<string | null>(null);
	let renamingId = $state<string | null>(null);
	let renameValue = $state('');
	let deleteConfirmId = $state<string | null>(null);

	onMount(async () => {
		const isLoginPage = window.location.pathname === '/login';

		if (isLoginPage) {
			authenticated = false;
			loading = false;
			return;
		}

		// Check if we're authenticated with Flask
		const authed = await checkAuth();
		if (!authed) {
			goto('/login');
			return;
		}

		authenticated = true;
		username = localStorage.getItem('username') || '';

		// Auto-collapse sidebar on mobile
		if (window.innerWidth < 768) sidebarCollapsed = true;

		// Load sessions and session data before showing the UI
		await loadSessions();
		loading = false;
		pollMemory();
	});

	async function loadSessions() {
		try {
			const sessionList = await listSessions();
			sessions.set(sessionList);

			// Try to restore active session from localStorage
			const savedId = getSessionId();
			// Validate savedId exists in the session list
			const validSaved = savedId && sessionList.some((s) => s.session_id === savedId);

			if (validSaved) {
				activeSessionId.set(savedId);
			} else if (sessionList.length > 0) {
				// Use the first session from the list
				const first = sessionList[0].session_id;
				activeSessionId.set(first);
				setSessionId(first);
				await apiSwitchSession(first);
			} else {
				// New user with no sessions — auto-create one
				await handleNewSession();
			}

			// Load session data to get chat history
			if ($activeSessionId) {
				await loadSessionData();
			}
		} catch (e) {
			console.error('Failed to load sessions:', e);
		}
	}

	function _normalizeChatMessage(msg: any) {
		return {
			...msg,
			timestamp: typeof msg.timestamp === 'string'
				? new Date(msg.timestamp).getTime()
				: (msg.timestamp || Date.now()),
			images: Array.isArray(msg.images) ? msg.images : [],
		};
	}

	async function loadSessionData() {
		// Loads only the most recent CHAT_PAGE_SIZE messages so a session with
		// hundreds of turns doesn't blow open-time. Older messages can be pulled
		// in via chatHistoryCursor's has_more flag (the message list shows a
		// "Load earlier" affordance).
		const CHAT_PAGE_SIZE = 50;
		try {
			const uuid = $activeSessionId;
			if (!uuid) {
				messages.set([]);
				chatHistoryCursor.set({ first_index: 0, total: 0, has_more: false });
				return;
			}
			const page = await getChatHistoryPage(uuid, CHAT_PAGE_SIZE);
			messages.set(page.messages.map(_normalizeChatMessage));
			chatHistoryCursor.set({
				first_index: page.first_index,
				total: page.total,
				has_more: page.has_more,
			});
		} catch {
			messages.set([]);
			chatHistoryCursor.set({ first_index: 0, total: 0, has_more: false });
		}
	}

	async function handleSwitchSession(sessionId: string) {
		// Look up session name for the overlay
		const target = $sessions.find((s) => s.session_id === sessionId);
		switchingName = target?.session_name || '';
		switchingSession = true;
		activeSessionId.set(sessionId);
		setSessionId(sessionId);
		messages.set([]); // Clear immediately for visual feedback
		try {
			// Activate first (sets Flask cookie), then load data
			await apiSwitchSession(sessionId);
			await loadSessionData();
		} finally {
			switchingSession = false;
			switchingName = '';
		}
	}

	async function handleNewSession() {
		try {
			const newUuid = await createNewSession();
			if (newUuid) {
				activeSessionId.set(newUuid);
				setSessionId(newUuid);
				// Auto-collapse panels for fresh session
				localStorage.setItem('panel_collection_collapsed', 'true');
				localStorage.setItem('panel_workflow_collapsed', 'true');
			} else {
				toasts.error('Failed to create session');
				return;
			}
			// Reload session list
			const sessionList = await listSessions();
			sessions.set(sessionList);
			// Load the new session's data
			messages.set([]);
			await loadSessionData();
		} catch (e) {
			console.error('Failed to create session:', e);
			toasts.error('Failed to create session');
		}
	}

	// --- Session management ---
	function openMenu(sessionId: string, e: MouseEvent) {
		e.stopPropagation();
		menuOpenId = menuOpenId === sessionId ? null : sessionId;
	}

	function startRename(sessionId: string, currentName: string) {
		menuOpenId = null;
		renamingId = sessionId;
		renameValue = currentName;
	}

	async function confirmRename(sessionId: string) {
		const name = renameValue.trim();
		if (!name) { renamingId = null; return; }
		try {
			await apiRenameSession(sessionId, name);
			sessions.update((list) =>
				list.map((s) =>
					s.session_id === sessionId ? { ...s, session_name: name } : s
				)
			);
			toasts.success('Session renamed');
		} catch (e) {
			console.error('Rename failed:', e);
			toasts.error('Failed to rename session');
		}
		renamingId = null;
	}

	function cancelRename() {
		renamingId = null;
	}

	function promptDelete(sessionId: string) {
		menuOpenId = null;
		deleteConfirmId = sessionId;
	}

	async function confirmDelete() {
		if (!deleteConfirmId) return;
		const uuid = deleteConfirmId;
		deleteConfirmId = null;
		try {
			await apiDeleteSession(uuid);
			// Remove from list
			sessions.update((list) => list.filter((s) => s.session_id !== uuid));
			// If we deleted the active session, switch to another
			if ($activeSessionId === uuid) {
				const remaining = $sessions;
				if (remaining.length > 0) {
					await handleSwitchSession(remaining[0].session_id);
				} else {
					// No sessions left — create a fresh one
					messages.set([]);
					await handleNewSession();
				}
			}
			toasts.success('Session deleted');
		} catch (e) {
			console.error('Delete failed:', e);
			toasts.error('Failed to delete session');
		}
	}

	// Close menus on outside click
	function handleGlobalClick() {
		if (menuOpenId) menuOpenId = null;
	}

	// Long-press delete all sessions
	let deleteAllConfirm = $state(false);
	let longPressTimer: ReturnType<typeof setTimeout> | null = null;
	let longPressTriggered = false;

	function startLongPress() {
		longPressTriggered = false;
		longPressTimer = setTimeout(() => { longPressTriggered = true; deleteAllConfirm = true; }, 1500);
	}
	function cancelLongPress() {
		if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = null; }
	}
	async function confirmDeleteAll() {
		deleteAllConfirm = false;
		try {
			await apiDeleteAllSessions();
			sessions.set([]);
			activeSessionId.set('');
			messages.set([]);
			toasts.success('All sessions deleted');
			// Create a fresh session
			await handleNewSession();
		} catch {
			toasts.error('Failed to delete all sessions');
		}
	}

	// Indexed files modal
	let indexedFilesModal = $state<{ open: boolean; sessionName: string; files: any[] }>({ open: false, sessionName: '', files: [] });

	async function showIndexedFiles(sessionId: string, sessionName: string) {
		menuOpenId = null;
		indexedFilesModal = { open: true, sessionName, files: [] };
		try {
			const result = await getIndexedFiles(sessionId);
			indexedFilesModal.files = result?.indexed_files || result?.files || [];
		} catch {
			indexedFilesModal.files = [];
		}
	}

	// Editable app title
	let appTitle = $state(typeof window !== 'undefined' ? localStorage.getItem('app_title') || 'R A G + L A B' : 'R A G + L A B');
	function saveTitle(e: Event) {
		const el = e.target as HTMLElement;
		appTitle = el.textContent || 'R A G + L A B';
		localStorage.setItem('app_title', appTitle);
	}

	let memoryInterval: ReturnType<typeof setInterval> | null = null;
	let memoryCleaningUp = $state(false);
	let memoryLongPressTimer: ReturnType<typeof setTimeout> | null = null;
	let memoryLongPressTriggered = false;

	async function pollMemory() {
		// Initial poll
		await fetchMemory();
		// Set up recurring poll — store handle for cleanup
		memoryInterval = setInterval(fetchMemory, 15000);
	}

	async function fetchMemory() {
		try {
			const resp = await fetch('/api/v1/system/memory', { credentials: 'include' });
			if (resp.ok) {
				const json = await resp.json();
				const data = json?.data || json;
				memoryPercent = data.gpu_percentage ?? data.ram_percentage ?? 0;
			}
		} catch { /* ignore */ }
	}

	async function handleMemoryClick() {
		if (memoryLongPressTriggered || memoryCleaningUp) return;
		memoryCleaningUp = true;
		try {
			await cleanupMemory();
			toasts.success('Memory cleanup completed');
			await fetchMemory();
		} catch {
			toasts.error('Memory cleanup failed (admin only)');
		} finally {
			memoryCleaningUp = false;
		}
	}

	function startMemoryLongPress() {
		memoryLongPressTriggered = false;
		memoryLongPressTimer = setTimeout(async () => {
			memoryLongPressTriggered = true;
			memoryCleaningUp = true;
			try {
				await emergencyMemoryCleanup();
				toasts.success('Emergency cleanup completed');
				await fetchMemory();
			} catch {
				toasts.error('Emergency cleanup failed (admin only)');
			} finally {
				memoryCleaningUp = false;
			}
		}, 1500);
	}

	function cancelMemoryLongPress() {
		if (memoryLongPressTimer) { clearTimeout(memoryLongPressTimer); memoryLongPressTimer = null; }
	}

	onDestroy(() => {
		if (memoryInterval) clearInterval(memoryInterval);
	});

	// --- Global keyboard shortcuts ---
	let shortcutsOpen = $state(false);

	function isTyping(e: KeyboardEvent): boolean {
		const tag = (e.target as HTMLElement)?.tagName;
		if (tag === 'INPUT' || tag === 'TEXTAREA') return true;
		if ((e.target as HTMLElement)?.isContentEditable) return true;
		return false;
	}

	function handleGlobalKeydown(e: KeyboardEvent) {
		if (!authenticated) return;

		// Escape: close modals/overlays (highest priority)
		if (e.key === 'Escape') {
			if (shortcutsOpen) { shortcutsOpen = false; e.preventDefault(); return; }
			if (settingsOpen) { settingsOpen = false; e.preventDefault(); return; }
			if (deleteConfirmId) { deleteConfirmId = null; e.preventDefault(); return; }
			if (deleteAllConfirm) { deleteAllConfirm = false; e.preventDefault(); return; }
			if (indexedFilesModal.open) { indexedFilesModal.open = false; e.preventDefault(); return; }
			return;
		}

		// Don't intercept when user is typing in inputs
		if (isTyping(e)) return;

		const ctrl = e.ctrlKey || e.metaKey;

		if (e.key === '?' || (ctrl && e.key === '/')) {
			// ? or Ctrl+/ → toggle shortcuts help
			e.preventDefault();
			shortcutsOpen = !shortcutsOpen;
		} else if (ctrl && e.key === 'n') {
			// Ctrl+N → new session
			e.preventDefault();
			handleNewSession();
		} else if (ctrl && e.key === ',') {
			// Ctrl+, → settings
			e.preventDefault();
			settingsOpen = !settingsOpen;
		} else if (ctrl && e.key === 'b') {
			// Ctrl+B → toggle sidebar
			e.preventDefault();
			sidebarCollapsed = !sidebarCollapsed;
		}
	}
</script>

<svelte:window onkeydown={handleGlobalKeydown} />

<SettingsModal open={settingsOpen} onclose={() => (settingsOpen = false)} />
<ToastContainer />

{#if shortcutsOpen}
	<div class="confirm-backdrop" onclick={() => (shortcutsOpen = false)} role="dialog" tabindex="-1">
		<div class="confirm-dialog wide" onclick={(e) => e.stopPropagation()}>
			<div class="modal-mini-header">
				<strong>Keyboard Shortcuts</strong>
				<button class="close-sm" onclick={() => (shortcutsOpen = false)}>&times;</button>
			</div>
			<div class="shortcuts-grid">
				<kbd>?</kbd><span>Show shortcuts</span>
				<kbd>Ctrl N</kbd><span>New session</span>
				<kbd>Ctrl ,</kbd><span>Settings</span>
				<kbd>Ctrl B</kbd><span>Toggle sidebar</span>
				<kbd>Esc</kbd><span>Close modal/dialog</span>
				<kbd>Enter</kbd><span>Send message</span>
				<kbd>Shift Enter</kbd><span>New line in message</span>
			</div>
		</div>
	</div>
{/if}

{#if deleteConfirmId}
	<div class="confirm-backdrop" onclick={() => (deleteConfirmId = null)} role="dialog" tabindex="-1">
		<div class="confirm-dialog" onclick={(e) => e.stopPropagation()}>
			<p>Delete this session and all its documents?</p>
			<div class="confirm-actions">
				<button class="cbtn cancel" onclick={() => (deleteConfirmId = null)}>Cancel</button>
				<button class="cbtn delete" onclick={confirmDelete}>Delete</button>
			</div>
		</div>
	</div>
{/if}

{#if indexedFilesModal.open}
	<div class="confirm-backdrop" onclick={() => (indexedFilesModal.open = false)} role="dialog" tabindex="-1">
		<div class="confirm-dialog wide" onclick={(e) => e.stopPropagation()}>
			<div class="modal-mini-header">
				<strong>{indexedFilesModal.sessionName}</strong> — Collection
				<button class="close-sm" onclick={() => (indexedFilesModal.open = false)}>&times;</button>
			</div>
			{#if indexedFilesModal.files.length === 0}
				<p class="empty-files">No documents indexed.</p>
			{:else}
				<ul class="files-list">
					{#each indexedFilesModal.files as file}
						<li>
							<span>{file.filename}</span>
							{#if file.page_count}<span class="file-pages">{file.page_count} pages</span>{/if}
						</li>
					{/each}
				</ul>
			{/if}
		</div>
	</div>
{/if}

{#if deleteAllConfirm}
	<div class="confirm-backdrop" onclick={() => (deleteAllConfirm = false)} role="dialog" tabindex="-1">
		<div class="confirm-dialog" onclick={(e) => e.stopPropagation()}>
			<p>Delete <strong>ALL</strong> sessions and their documents? This cannot be undone.</p>
			<div class="confirm-actions">
				<button class="cbtn cancel" onclick={() => (deleteAllConfirm = false)}>Cancel</button>
				<button class="cbtn delete" onclick={confirmDeleteAll}>Delete All</button>
			</div>
		</div>
	</div>
{/if}

{#if loading}
	<div class="loading">Loading...</div>
{:else if !authenticated}
	{@render children()}
{:else}
<!-- svelte-ignore a11y_click_events_have_key_events -->
<div class="app" onclick={handleGlobalClick}>
	<nav class="sidebar" class:collapsed={sidebarCollapsed}>
		<div class="sidebar-top">
			<div class="logo"
				contenteditable="true"
				spellcheck="false"
				onblur={saveTitle}
				role="textbox"
				tabindex="0"
			>{appTitle}</div>
			<button class="sidebar-toggle" onclick={() => (sidebarCollapsed = true)} title="Collapse sidebar">
				<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><rect x="1" y="1" width="14" height="14" rx="3" stroke="currentColor" stroke-width="1.3"/><rect x="1" y="1" width="5" height="14" rx="3" fill="currentColor" opacity="0.3"/><path d="M9.5 6L7.5 8L9.5 10" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>
			</button>
		</div>

		<div class="sessions-section">
			<div class="sessions-header">
				<span>Sessions</span>
				<button class="new-btn"
					onclick={() => { if (!longPressTriggered) handleNewSession(); }}
					onmousedown={startLongPress}
					onmouseup={cancelLongPress}
					onmouseleave={cancelLongPress}
					title="New session (long-press to delete all)"
				>+</button>
			</div>
			<div class="sessions-list">
				{#each $sessions as session}
					{@const sid = session.session_id}
					{@const sname = session.session_name || 'Untitled'}
					<div class="session-row" class:active={sid === $activeSessionId}>
						{#if renamingId === sid}
							<input
								class="rename-input"
								type="text"
								bind:value={renameValue}
								onkeydown={(e) => { if (e.key === 'Enter') confirmRename(sid); if (e.key === 'Escape') cancelRename(); }}
								onblur={() => confirmRename(sid)}
								autofocus
							/>
						{:else}
							<button
								class="session-name"
								onclick={() => handleSwitchSession(sid)}
								ondblclick={() => startRename(sid, sname)}
								title="Double-click to rename"
							>
								{sname}
							</button>
							<button class="session-menu-btn" onclick={(e) => openMenu(sid, e)} title="Options">
								⋯
							</button>
						{/if}

						{#if menuOpenId === sid}
							<div class="session-popup">
								<button class="popup-item" onclick={() => startRename(sid, sname)}>Rename</button>
								<button class="popup-item" onclick={() => showIndexedFiles(sid, sname)}>Collection</button>
								<button class="popup-item danger" onclick={() => promptDelete(sid)}>Delete</button>
							</div>
						{/if}
					</div>
				{/each}
			</div>
		</div>
	</nav>

	<aside class="center-panel" class:collapsed={!centerPanelOpen}>
		<div class="center-panel-top">
			<span class="center-panel-title">Documents & Templates</span>
			<button class="center-panel-close" onclick={() => (centerPanelOpen = false)} title="Close panel">
				<svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M4 4L12 12M12 4L4 12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
			</button>
		</div>
		<DocumentPanel />
		<TemplatePanel />
	</aside>

	<div class="main-area">
		<header class="top-bar">
			{#if sidebarCollapsed}
				<button class="hamburger" onclick={() => (sidebarCollapsed = false)} title="Show sidebar">
					<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><rect x="1" y="1" width="14" height="14" rx="3" stroke="currentColor" stroke-width="1.3"/><rect x="1" y="1" width="5" height="14" rx="3" fill="currentColor" opacity="0.3"/><path d="M8.5 6L10.5 8L8.5 10" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>
				</button>
			{/if}
			{#if !centerPanelOpen}
				<button class="top-btn" onclick={() => (centerPanelOpen = true)}>
					<svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M2 3h12M2 6h8M2 9h10M2 12h6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
					Docs
				</button>
			{/if}
			<button class="top-btn" onclick={() => (settingsOpen = true)}>
				<svg width="14" height="14" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="2.5" stroke="currentColor" stroke-width="1.3"/><path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.05 3.05l1.41 1.41M11.54 11.54l1.41 1.41M3.05 12.95l1.41-1.41M11.54 4.46l1.41-1.41" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>
				Settings
			</button>
			<button class="theme-toggle" onclick={() => (darkMode = !darkMode)} title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}>
				{#if darkMode}
					<svg width="14" height="14" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="3.5" stroke="currentColor" stroke-width="1.3"/><path d="M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1.06 1.06M11.54 11.54l1.06 1.06M3.4 12.6l1.06-1.06M11.54 4.46l1.06-1.06" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>
				{:else}
					<svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M13.5 9.5a5.5 5.5 0 01-7-7A5.5 5.5 0 1013.5 9.5z" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>
				{/if}
			</button>
			<div class="top-spacer"></div>
			<button class="memory-bar"
				title="GPU memory — click: cleanup, long-press: emergency"
				onclick={handleMemoryClick}
				onmousedown={startMemoryLongPress}
				onmouseup={cancelMemoryLongPress}
				onmouseleave={cancelMemoryLongPress}
				disabled={memoryCleaningUp}
			>
				<div class="memory-fill" style="width: {Math.min(memoryPercent, 100)}%"></div>
				<span class="memory-text">{memoryCleaningUp ? '...' : `${memoryPercent.toFixed(0)}%`}</span>
			</button>
			<button class="top-btn logout" onclick={async () => { await apiLogout(); goto('/login'); }}>
				{username || 'Logout'}
			</button>
		</header>
		<main class="content">
			{#if switchingSession}
				<div class="switching-overlay">
					<div class="switch-spinner"></div>
					Loading{switchingName ? ` "${switchingName}"` : ''}...
				</div>
			{/if}
			{@render children()}
		</main>
	</div>
</div>
{/if}

<style>
	@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Geist+Mono:wght@400;500&display=swap');

	/* ── Monochrome Metallic Theme ──
	   Light: brushed aluminum with cool gray surfaces and steel-blue accent.
	   Dark: brushed titanium with layered charcoals and silver text. */
	:global(:root) {
		/* Light mode — brushed aluminum */
		--bg-page: #edeef1;
		--bg-surface: #f5f6f8;
		--bg-surface-alt: #f0f1f4;
		--bg-hover: #e6e7eb;
		--bg-active: #dcdde2;
		--bg-input: #f0f1f4;
		--border: #c8c9cf;
		--border-light: #d8d9de;
		--border-highlight: rgba(255, 255, 255, 0.65);
		--text-primary: #2c2d32;
		--text-secondary: #62636a;
		--text-muted: #8e8f96;
		--text-heading: #1a1b1f;
		--accent: #4a7a9e;
		--accent-bg: #e4eef5;
		--accent-border: #6b9fc8;
		--accent-light: #3a6a8e;
		--danger: #b34040;
		--success: #3d8a52;
		--shadow: rgba(0, 0, 0, 0.06);
		--shadow-lg: rgba(0, 0, 0, 0.1);
		--radius: 10px;
		--font-sans: 'DM Sans', -apple-system, system-ui, sans-serif;
		--font-mono: 'Geist Mono', 'JetBrains Mono', monospace;
		/* Sidebar — dark contrast panel */
		--sidebar-bg: #22232a;
		--sidebar-hover: #2c2d35;
		--sidebar-active: #37383f;
		--sidebar-text: #b0b2b8;
		--sidebar-text-muted: #62636a;
		--sidebar-border: #2c2d35;
	}

	:global(:root[data-theme="dark"]) {
			/* Dark mode — brushed titanium */
			--bg-page: #18191c;
			--bg-surface: #1e1f23;
			--bg-surface-alt: #1a1b1f;
			--bg-hover: #26272c;
			--bg-active: #2e2f35;
			--bg-input: #222328;
			--border: #33343a;
			--border-light: #2a2b30;
			--border-highlight: rgba(255, 255, 255, 0.06);
			--text-primary: #c8cad0;
			--text-secondary: #8b8d95;
			--text-muted: #55565e;
			--text-heading: #dcdee3;
			--accent: #6b9fc8;
			--accent-bg: #1c2630;
			--accent-border: #4a7a9e;
			--accent-light: #7eb3d6;
			--danger: #c45c5c;
			--success: #5cad6f;
			--shadow: rgba(0, 0, 0, 0.35);
			--shadow-lg: rgba(0, 0, 0, 0.55);
			--sidebar-bg: #121316;
			--sidebar-hover: #1c1d21;
			--sidebar-active: #26272c;
			--sidebar-text: #b0b2b8;
			--sidebar-text-muted: #55565e;
			--sidebar-border: #1e1f23;
	}

	:global(*) { box-sizing: border-box; }
	:global(body) {
		margin: 0;
		font-family: var(--font-sans);
		background: var(--bg-page); color: var(--text-primary); font-size: 14px;
		-webkit-font-smoothing: antialiased;
		-moz-osx-font-smoothing: grayscale;
		letter-spacing: 0.01em;
		color-scheme: light dark;
	}

	.loading {
		display: flex; align-items: center; justify-content: center; height: 100vh;
		color: var(--text-muted); font-weight: 500;
	}

	.app { display: flex; height: 100vh; }

	/* ── Sidebar ── */
	.sidebar {
		width: 260px; background: var(--sidebar-bg);
		display: flex; flex-direction: column; flex-shrink: 0;
		transition: width 0.2s ease, margin 0.2s ease;
		overflow: hidden;
		border-right: 1px solid var(--border-light);
		box-shadow: inset 0 1px 0 var(--border-highlight);
	}
	.sidebar.collapsed { width: 0; }

	.sidebar-top {
		display: flex; align-items: center; justify-content: space-between;
		padding: 0.65rem 0.65rem; border-bottom: 1px solid var(--sidebar-border);
	}
	.sidebar-toggle {
		background: none; border: none; color: var(--sidebar-text-muted);
		cursor: pointer; padding: 0.25rem; border-radius: 6px; display: flex;
		align-items: center; justify-content: center; flex-shrink: 0;
		transition: all 0.1s;
	}
	.sidebar-toggle:hover { color: #fff; background: var(--sidebar-hover); }

	.logo {
		font-size: 0.78rem; font-weight: 700; letter-spacing: 0.18em; color: #fff;
		outline: none; cursor: text; white-space: nowrap;
	}
	.logo:focus { opacity: 0.7; }

	.sessions-section { flex: 1; display: flex; flex-direction: column; min-height: 0; padding: 0.5rem; }
	.sessions-header {
		display: flex; justify-content: space-between; align-items: center;
		padding: 0.25rem 0.5rem; margin-bottom: 0.25rem;
	}
	.sessions-header span { font-size: 0.65rem; text-transform: uppercase; color: var(--sidebar-text-muted); font-weight: 600; letter-spacing: 0.08em; }
	.new-btn {
		background: none; border: 1px solid var(--sidebar-border); color: var(--sidebar-text-muted);
		width: 22px; height: 22px; border-radius: 6px; cursor: pointer; font-size: 0.9rem;
		display: flex; align-items: center; justify-content: center;
		transition: all 0.15s;
	}
	.new-btn:hover { background: var(--sidebar-hover); color: #fff; border-color: var(--sidebar-text-muted); }
	.sessions-list { display: flex; flex-direction: column; gap: 1px; overflow-y: auto; flex: 1; }

	.session-row {
		position: relative; display: flex; align-items: center;
		border-radius: 8px; padding: 0; gap: 0;
		transition: background 0.1s;
	}
	.session-row:hover { background: var(--sidebar-hover); }
	.session-row.active { background: var(--sidebar-active); }
	.session-row:hover .session-menu-btn { opacity: 1; }

	.session-name {
		flex: 1; background: none; border: none; color: var(--sidebar-text);
		padding: 0.5rem 0.65rem; text-align: left; border-radius: 8px;
		cursor: pointer; font-size: 0.82rem; font-family: var(--font-sans); font-weight: 400;
		overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
		min-width: 0;
	}
	.session-row.active .session-name { color: #fff; font-weight: 500; }

	.session-menu-btn {
		background: none; border: none; color: var(--sidebar-text-muted); cursor: pointer;
		padding: 0.25rem 0.4rem; font-size: 0.85rem; opacity: 0;
		transition: opacity 0.1s; border-radius: 4px; flex-shrink: 0; line-height: 1;
	}
	.session-menu-btn:hover { color: #fff; background: rgba(255,255,255,0.1); }

	.rename-input {
		flex: 1; font-size: 0.82rem; padding: 0.35rem 0.5rem; font-family: var(--font-sans);
		border: 1.5px solid var(--accent); border-radius: 6px; outline: none;
		background: var(--sidebar-bg); color: #fff; min-width: 0;
	}

	.session-popup {
		position: absolute; right: 4px; top: 100%; z-index: 20;
		background: #262626; border: 1px solid #333; border-radius: 10px;
		box-shadow: 0 8px 30px rgba(0,0,0,0.4); overflow: hidden; min-width: 120px;
	}
	.popup-item {
		display: block; width: 100%; background: none; border: none;
		padding: 0.5rem 0.85rem; text-align: left; font-size: 0.82rem;
		color: #d4d4d4; cursor: pointer; font-family: var(--font-sans);
		transition: background 0.1s;
	}
	.popup-item:hover { background: #333; }
	.popup-item.danger { color: #f87171; }
	.popup-item.danger:hover { background: rgba(248, 113, 113, 0.1); }

	/* Dialogs */
	.confirm-backdrop {
		position: fixed; inset: 0; z-index: 1100;
		background: rgba(28, 25, 23, 0.4);
		display: flex; align-items: center; justify-content: center;
		backdrop-filter: blur(2px);
		animation: fadeIn 0.15s ease;
	}
	@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
	.confirm-dialog {
		background: var(--bg-surface); border-radius: 12px; padding: 1.25rem 1.5rem;
		box-shadow: 0 20px 60px var(--shadow-lg); max-width: 340px;
		border: 1px solid var(--border);
		animation: scaleIn 0.15s ease;
	}
	@keyframes scaleIn { from { transform: scale(0.95); opacity: 0; } to { transform: scale(1); opacity: 1; } }
	.confirm-dialog p { margin: 0 0 1rem; font-size: 0.9rem; color: var(--text-heading); line-height: 1.5; }
	.confirm-actions { display: flex; gap: 0.5rem; justify-content: flex-end; }
	.cbtn {
		padding: 0.4rem 0.9rem; border-radius: 7px; font-size: 0.82rem;
		font-weight: 600; cursor: pointer; border: none; font-family: var(--font-sans);
		transition: all 0.12s;
	}
	.cbtn.cancel { background: var(--bg-hover); color: var(--text-secondary); }
	.cbtn.cancel:hover { background: var(--border); }
	.cbtn.delete { background: var(--danger); color: white; }
	.cbtn.delete:hover { filter: brightness(1.1); }

	/* Indexed files modal */
	.confirm-dialog.wide { max-width: 440px; }
	.modal-mini-header {
		display: flex; justify-content: space-between; align-items: center;
		margin-bottom: 0.75rem; font-size: 0.88rem; color: var(--text-heading); font-weight: 600;
	}
	.close-sm { background: none; border: none; font-size: 1.2rem; color: var(--text-muted); cursor: pointer; transition: color 0.1s; }
	.close-sm:hover { color: var(--text-heading); }
	.empty-files { color: var(--text-muted); font-size: 0.82rem; text-align: center; margin: 1rem 0; }
	.files-list {
		list-style: none; padding: 0; margin: 0;
		max-height: 300px; overflow-y: auto;
	}
	.files-list li {
		display: flex; justify-content: space-between; align-items: center;
		padding: 0.4rem 0; border-bottom: 1px solid var(--border-light);
		font-size: 0.82rem; color: var(--text-heading);
	}
	.file-pages { font-size: 0.68rem; color: var(--text-muted); font-family: var(--font-mono); }

	/* ── Center Panel (slide-over) ── */
	.center-panel {
		width: 300px; background: var(--bg-surface-alt); border-right: 1px solid var(--border);
		flex-shrink: 0; overflow: hidden;
		transition: width 0.2s ease;
		box-shadow: inset 0 1px 0 var(--border-highlight);
	}
	.center-panel.collapsed { width: 0; border-right: none; }
	.center-panel.collapsed * { visibility: hidden; }
	.center-panel-top {
		display: flex; align-items: center; padding: 0.5rem 0.75rem;
		border-bottom: 1px solid var(--border);
	}
	.center-panel-title {
		font-size: 0.78rem; font-weight: 600; color: var(--text-heading);
	}
	.center-panel-close {
		margin-left: auto; background: none; border: none; color: var(--text-muted);
		cursor: pointer; padding: 0.25rem; border-radius: 6px; line-height: 1;
		display: flex; align-items: center; transition: all 0.1s;
	}
	.center-panel-close:hover { color: var(--text-heading); background: var(--bg-hover); }
	/* ── Main Area ── */
	.main-area { flex: 1; display: flex; flex-direction: column; overflow: hidden; background: var(--bg-surface); }

	.top-bar {
		display: flex; align-items: center; gap: 0.5rem;
		padding: 0.5rem 0.75rem;
		border-bottom: 1px solid var(--border);
		background: linear-gradient(180deg, var(--bg-surface) 0%, var(--bg-surface-alt) 100%);
		box-shadow: inset 0 1px 0 var(--border-highlight);
	}
	.hamburger {
		background: none; border: none; color: var(--text-muted);
		cursor: pointer; padding: 0.3rem; line-height: 1; border-radius: 6px;
		transition: all 0.1s; display: flex; align-items: center;
	}
	.hamburger:hover { color: var(--text-heading); background: var(--bg-hover); }

	.top-btn {
		background: none; border: none; color: var(--text-secondary);
		padding: 0.35rem 0.65rem; border-radius: 8px; font-size: 0.8rem; font-weight: 500; cursor: pointer;
		font-family: var(--font-sans);
		transition: all 0.1s;
		display: flex; align-items: center; gap: 0.35rem;
	}
	.top-btn:hover { background: var(--bg-hover); color: var(--text-heading); }
	.top-btn.active { background: var(--bg-hover); color: var(--text-heading); }
	.theme-toggle {
		background: none; border: 1px solid var(--border); color: var(--text-muted);
		padding: 0.3rem; border-radius: 6px; cursor: pointer;
		display: flex; align-items: center; justify-content: center;
		transition: all 0.15s; line-height: 1;
	}
	.theme-toggle:hover { color: var(--text-heading); border-color: var(--text-muted); background: var(--bg-hover); }
	.top-spacer { flex: 1; }
	.top-btn.logout { color: var(--text-muted); font-size: 0.78rem; }
	.top-btn.logout:hover { color: var(--text-heading); background: var(--bg-hover); }

	.memory-bar {
		position: relative; width: 64px; height: 18px; background: var(--bg-hover);
		border-radius: 9px; overflow: hidden;
		border: none; padding: 0; cursor: pointer;
	}
	.memory-bar:hover { outline: 2px solid var(--border); outline-offset: 1px; }
	.memory-bar:disabled { cursor: wait; opacity: 0.5; }
	.memory-fill {
		height: 100%; background: linear-gradient(90deg, #22c55e, #eab308, #ef4444);
		border-radius: 9px; transition: width 1s ease;
	}
	.memory-text {
		position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
		font-size: 0.58rem; font-weight: 600; color: var(--text-secondary);
		font-family: var(--font-mono);
	}

	.content { flex: 1; display: flex; flex-direction: column; overflow: hidden; position: relative; }

	.switching-overlay {
		position: absolute; inset: 0; z-index: 10;
		background: rgba(255, 255, 255, 0.9);
		display: flex; flex-direction: column; align-items: center; justify-content: center;
		gap: 0.75rem; color: var(--text-secondary); font-size: 0.88rem; font-weight: 500;
		backdrop-filter: blur(4px);
	}
	.switch-spinner {
		width: 24px; height: 24px; border: 2px solid var(--border);
		border-top-color: var(--accent); border-radius: 50%;
		animation: spin 0.6s linear infinite;
	}
	@keyframes spin { to { transform: rotate(360deg); } }

	/* ── Responsive ── */
	@media (max-width: 768px) {
		.sidebar {
			position: fixed; top: 0; left: 0; bottom: 0; z-index: 100;
			width: 280px; box-shadow: 8px 0 24px rgba(0,0,0,0.3);
			transform: translateX(0);
			transition: transform 0.2s ease;
		}
		.sidebar.collapsed { transform: translateX(-100%); width: 280px; }

		.center-panel {
			position: fixed; top: 0; left: 0; bottom: 0; z-index: 90;
			width: 300px; box-shadow: 8px 0 24px rgba(0,0,0,0.15);
			transition: transform 0.2s ease;
			transform: translateX(0);
		}
		.center-panel.collapsed { transform: translateX(-100%); width: 300px; }

		.app { flex-direction: column; }
		.main-area { flex: 1; min-height: 0; }

		.top-bar { gap: 0.3rem; padding: 0.35rem 0.5rem; flex-wrap: wrap; }
		.top-btn { padding: 3px 8px; font-size: 0.72rem; }
		.memory-bar { width: 50px; height: 16px; }
	}

	/* Shortcuts help */
	.shortcuts-grid {
		display: grid; grid-template-columns: auto 1fr;
		gap: 0.4rem 1rem; align-items: center;
	}
	.shortcuts-grid kbd {
		display: inline-block; background: var(--bg-hover); border: 1px solid var(--border);
		border-radius: 5px; padding: 2px 8px; font-size: 0.7rem; font-weight: 600;
		font-family: var(--font-mono); color: var(--text-heading); white-space: nowrap;
		box-shadow: 0 1px 0 var(--border);
	}
	.shortcuts-grid span { font-size: 0.8rem; color: var(--text-secondary); }
</style>
