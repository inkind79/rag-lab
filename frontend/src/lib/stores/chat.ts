/**
 * Chat Store
 *
 * Reactive state for chat messages and streaming status.
 */

import { writable } from 'svelte/store';

export interface RetrievedImage {
	path: string;
	full_path?: string;
	score: number;
	text_preview?: string;
	result_type?: string;
	source?: string;
	page_num?: number;
}

export interface ChatMessage {
	id?: string;
	type?: string;
	role: 'user' | 'assistant' | 'system';
	content: string | any[];
	timestamp: number;
	images?: RetrievedImage[];
	model?: string;
	modelDisplayName?: string;
	reasoning?: string;
	template_name?: string;
}

export const messages = writable<ChatMessage[]>([]);
export const isStreaming = writable(false);
export const currentModel = writable<string>('');
export const currentModelDisplay = writable<string>('');

/**
 * Pagination cursor for the chat history loaded into ``messages``.
 *
 * - ``first_index`` is the position of the first loaded message in the full
 *   server-side history. After ``loadEarlier()`` runs it shifts down toward 0.
 * - ``total`` is the size of the full history; lets the UI show "showing
 *   X of Y" without a second roundtrip.
 * - ``has_more`` is true when there are still older messages on the server.
 */
export interface ChatHistoryCursor {
	first_index: number;
	total: number;
	has_more: boolean;
}

export const chatHistoryCursor = writable<ChatHistoryCursor>({
	first_index: 0,
	total: 0,
	has_more: false,
});
