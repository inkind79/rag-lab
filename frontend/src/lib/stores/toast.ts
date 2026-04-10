import { writable } from 'svelte/store';

export interface Toast {
	id: number;
	type: 'success' | 'error' | 'info' | 'warning';
	message: string;
}

let nextId = 0;

function createToastStore() {
	const { subscribe, update } = writable<Toast[]>([]);

	function add(type: Toast['type'], message: string, duration = 3500) {
		const id = nextId++;
		update((toasts) => [...toasts, { id, type, message }]);
		setTimeout(() => dismiss(id), duration);
	}

	function dismiss(id: number) {
		update((toasts) => toasts.filter((t) => t.id !== id));
	}

	return {
		subscribe,
		success: (msg: string) => add('success', msg),
		error: (msg: string) => add('error', msg, 5000),
		info: (msg: string) => add('info', msg),
		warning: (msg: string) => add('warning', msg, 4000),
		dismiss,
	};
}

export const toasts = createToastStore();
