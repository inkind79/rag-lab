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
