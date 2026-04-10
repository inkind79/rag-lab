/**
 * Session Store
 *
 * Reactive state for the active session and session list.
 */

import { writable, derived } from 'svelte/store';

export interface Session {
	session_id: string;
	session_name: string;
	user_id: string;
	generation_model: string;
	retrieval_method: string;
	indexed_files: Array<{ filename: string; file_type: string }>;
	selected_docs: string[];
}

export const sessions = writable<Session[]>([]);
export const activeSessionId = writable<string | null>(null);

export const activeSession = derived(
	[sessions, activeSessionId],
	([$sessions, $activeSessionId]) =>
		$sessions.find((s) => s.session_id === $activeSessionId) ?? null
);

// Track active template type so the chat page can detect batch mode
export const selectedTemplateType = writable<string>('general');
// Track optimized query lock (if set, chat input is locked to this query)
export const optimizedQuery = writable<string>('');
