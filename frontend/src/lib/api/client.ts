/**
 * RAG Lab API Client
 *
 * All endpoints target the backend server via Vite proxy.
 * Auth: Flask-Login session cookies (credentials: 'include').
 *
 * Endpoint reference (v1 API):
 *   Auth:      POST /login (form), GET /logout
 *   Sessions:  /api/v1/sessions/*
 *   Docs:      /api/v1/documents/*
 *   Chat:      POST /api/v1/chat/stream (SSE)
 *   Settings:  PUT /api/v1/settings
 *   Templates: /api/prompt-templates/*
 *   Feedback:  /api/feedback/*
 *   Memory:    /api/v1/system/memory, /api/v1/chat/memory/*
 */

class ApiError extends Error {
	status: number;
	constructor(message: string, status: number) {
		super(message);
		this.status = status;
	}
}

// --- Session UUID ---

export function getSessionId(): string | null {
	if (typeof window !== 'undefined') {
		const val = localStorage.getItem('active_session_uuid');
		// Guard against corrupted values
		if (!val || val === 'undefined' || val === 'null' || val.length < 10) {
			return null;
		}
		return val;
	}
	return null;
}

export function setSessionId(uuid: string): void {
	if (typeof window !== 'undefined') {
		// Never store undefined/null/empty values
		if (uuid && uuid !== 'undefined' && uuid !== 'null' && uuid.length > 10) {
			localStorage.setItem('active_session_uuid', uuid);
		}
	}
}

// --- Session normalization ---

import type { Session } from '$lib/stores/session';

/**
 * Normalize a raw session object from the Flask API into the canonical Session shape.
 * The backend returns { id, name } from get_all_sessions() but { session_id, session_name }
 * from create/activate endpoints. This normalizes both to the Session interface.
 */
function normalizeSession(raw: Record<string, any>): Session {
	return {
		session_id: raw.session_id || raw.id || '',
		session_name: raw.session_name || raw.name || 'Untitled',
		user_id: raw.user_id || '',
		generation_model: raw.generation_model || '',
		retrieval_method: raw.retrieval_method || '',
		indexed_files: raw.indexed_files || [],
		selected_docs: raw.selected_docs || [],
	};
}

// --- Fetch wrapper ---

async function fetchJson<T = any>(path: string, options: RequestInit = {}): Promise<T> {
	const sessionId = getSessionId();
	const headers: Record<string, string> = {
		...(options.headers as Record<string, string>),
	};
	if (sessionId) headers['X-Session-UUID'] = sessionId;
	if (!(options.body instanceof FormData) && !(options.body instanceof URLSearchParams)) {
		headers['Content-Type'] = 'application/json';
	}

	const response = await fetch(path, { ...options, headers, credentials: 'include' });

	if (!response.ok) {
		const text = await response.text();
		let msg = response.statusText;
		try { msg = JSON.parse(text).message || JSON.parse(text).error || msg; } catch {}
		throw new ApiError(msg, response.status);
	}

	const ct = response.headers.get('content-type') || '';
	if (ct.includes('application/json')) return response.json();
	return {} as T;
}

// ============================================================
//  AUTH — JWT cookie-based (FastAPI)
// ============================================================

export async function login(username: string, password: string): Promise<boolean> {
	try {
		const resp = await fetch('/auth/login', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ username, password }),
			credentials: 'include',
		});

		if (!resp.ok) return false;

		const data = await resp.json();
		if (!data?.success) return false;

		if (typeof window !== 'undefined') localStorage.setItem('username', username);

		// Fetch sessions list to get the active session UUID
		try {
			const sr = await fetch('/api/v1/sessions', { credentials: 'include' });
			if (sr.ok) {
				const sd = await sr.json();
				const raw: any[] = sd?.data?.sessions || [];
				const sessions = raw.map(normalizeSession);
				if (sessions.length > 0 && sessions[0].session_id) {
					setSessionId(sessions[0].session_id);
				}
			}
		} catch {}
		return true;
	} catch {
		return false;
	}
}

export async function checkAuth(): Promise<boolean> {
	try {
		const resp = await fetch('/auth/check', {
			credentials: 'include',
		});
		return resp.ok;
	} catch {
		return false;
	}
}

export async function logout(): Promise<void> {
	await fetch('/auth/logout', { method: 'POST', credentials: 'include' });
}

// ============================================================
//  SESSIONS — v1 API
// ============================================================

export async function listSessions(): Promise<Session[]> {
	try {
		const data = await fetchJson('/api/v1/sessions');
		const raw: any[] = data?.data?.sessions || [];
		return raw.map(normalizeSession);
	} catch {
		return [];
	}
}

export async function getSessionData(): Promise<any> {
	const uuid = getSessionId();
	if (!uuid) return {};
	const resp = await fetchJson(`/api/v1/sessions/${uuid}`);
	// Callers expect resp?.session_data || resp — wrap v1 response for compatibility
	return { session_data: resp?.data || resp };
}

export async function getSessionById(uuid: string): Promise<any> {
	return fetchJson(`/api/v1/sessions/${uuid}`);
}

export async function switchSession(uuid: string): Promise<void> {
	// Use our JSON API endpoint — avoids the Flask /switch_session redirect
	// which triggers a full /chat page render (208KB, 10s).
	await fetchJson(`/api/v1/sessions/${uuid}/activate`, { method: 'POST' });
	setSessionId(uuid);
}

export async function createNewSession(): Promise<string | null> {
	try {
		const resp = await fetchJson('/api/v1/sessions', { method: 'POST', body: JSON.stringify({}) });
		const uuid = resp?.data?.session_id;
		if (uuid) {
			setSessionId(uuid);
			return uuid;
		}
	} catch {}
	return null;
}

export async function renameSession(uuid: string, name: string): Promise<any> {
	return fetchJson(`/api/v1/sessions/${uuid}/rename`, {
		method: 'PUT',
		body: JSON.stringify({ name }),
	});
}

export async function deleteSession(uuid: string): Promise<any> {
	return fetchJson(`/api/v1/sessions/${uuid}`, { method: 'DELETE' });
}

export async function deleteAllSessions(): Promise<any> {
	return fetchJson('/api/v1/sessions/all', { method: 'DELETE' });
}

// ============================================================
//  DOCUMENTS — v1 API
// ============================================================

export async function getIndexedFiles(uuid: string): Promise<any> {
	const resp = await fetchJson(`/api/v1/documents/${uuid}`);
	// Callers expect { indexed_files: [...], selected_docs: [...] }
	return resp?.data || resp;
}

export async function uploadDocuments(files: FileList, sessionUuid: string): Promise<any> {
	const formData = new FormData();
	for (const file of files) formData.append('files', file);

	const resp = await fetchJson('/api/v1/documents/upload', { method: 'POST', body: formData });
	return resp?.data || resp;
}

export async function saveSelectedDocs(selectedDocs: string[]): Promise<any> {
	return fetchJson('/api/v1/documents/selection', {
		method: 'PUT',
		body: JSON.stringify({ selected_docs: selectedDocs }),
	});
}

export async function deleteDocuments(filenames: string[]): Promise<any> {
	return fetchJson('/api/v1/documents', {
		method: 'DELETE',
		body: JSON.stringify({ filenames }),
	});
}

export async function reindexDocuments(): Promise<any> {
	return fetchJson('/api/v1/documents/reindex', { method: 'POST' });
}

// ============================================================
//  CHAT — Flask SSE streaming
// ============================================================

export async function streamChat(
	query: string,
	sessionUuid: string,
	options: { isRagMode?: boolean; isBatchMode?: boolean; pastedImages?: string[]; signal?: AbortSignal } = {}
): Promise<Response> {
	const headers: Record<string, string> = { 'Content-Type': 'application/json' };
	if (sessionUuid) headers['X-Session-UUID'] = sessionUuid;

	const body: Record<string, any> = {
		query,
		is_rag_mode: options.isRagMode ?? true,
		is_batch_mode: options.isBatchMode ?? false,
		has_pasted_images: (options.pastedImages?.length ?? 0) > 0,
		pasted_images: options.pastedImages ?? [],
	};
	if (sessionUuid) body.session_uuid = sessionUuid;

	return fetch('/api/v1/chat/stream', {
		method: 'POST',
		headers,
		credentials: 'include',
		body: JSON.stringify(body),
		signal: options.signal,
	});
}

// ============================================================
//  SETTINGS — v1 API
// ============================================================

export async function saveSettings(settings: Record<string, any>): Promise<any> {
	// Use the /api/v1/settings endpoint (our Flask API layer)
	return fetchJson('/api/v1/settings', {
		method: 'PUT',
		body: JSON.stringify(settings),
	});
}

// ============================================================
//  TEMPLATES — v1 API
// ============================================================

export async function getTemplates(): Promise<any[]> {
	try {
		const data = await fetchJson('/api/v1/templates');
		if (data?.templates) return data.templates;
		if (Array.isArray(data)) return data;
		return [];
	} catch {
		return [];
	}
}

export async function selectTemplate(sessionUuid: string, templateId: string): Promise<any> {
	return fetchJson(`/api/v1/sessions/${sessionUuid}/template`, {
		method: 'POST',
		body: JSON.stringify({ template_id: templateId }),
	});
}

export async function createTemplate(data: Record<string, any>): Promise<any> {
	return fetchJson('/api/v1/templates', {
		method: 'POST',
		body: JSON.stringify(data),
	});
}

export async function updateTemplate(templateId: string, data: Record<string, any>): Promise<any> {
	return fetchJson(`/api/v1/templates/${templateId}`, {
		method: 'PUT',
		body: JSON.stringify(data),
	});
}

export async function deleteTemplate(templateId: string): Promise<any> {
	return fetchJson(`/api/v1/templates/${templateId}`, {
		method: 'DELETE',
	});
}

export async function setDefaultTemplate(templateId: string): Promise<any> {
	return fetchJson(`/api/v1/templates/${templateId}/set-default`, {
		method: 'POST',
	});
}

// ============================================================
//  FEEDBACK & EVALUATION — v1 API
export async function generateWithModel(query: string, model?: string, format?: any): Promise<any> {
	const resp = await fetchJson('/api/v1/generate', {
		method: 'POST',
		body: JSON.stringify({ query, model, format }),
	});
	return resp?.data || resp;
}

// ============================================================

export async function submitFeedback(data: {
	message_id: string;
	relevant_images?: string[];
	expected_response?: string;
}): Promise<any> {
	return fetchJson('/api/v1/feedback/submit', {
		method: 'POST',
		body: JSON.stringify(data),
	});
}

export async function startOptimization(data: {
	message_id: string;
	iteration_count?: number;
	expected_response?: string;
	relevant_images?: string[];
}): Promise<any> {
	return fetchJson('/api/v1/feedback/optimize', {
		method: 'POST',
		body: JSON.stringify(data),
	});
}

export async function getOptimizationStatus(runId: string): Promise<any> {
	return fetchJson(`/api/v1/feedback/optimization/${runId}/status`);
}

export async function getOptimizationResults(runId: string): Promise<any> {
	return fetchJson(`/api/v1/feedback/optimization/${runId}/results`);
}

// ============================================================
//  SYSTEM — v1 API
// ============================================================

export async function getMemoryUsage(): Promise<any> {
	const resp = await fetchJson('/api/v1/system/memory');
	return resp?.data || resp;
}

export async function cleanupMemory(): Promise<any> {
	return fetchJson('/api/v1/system/memory/cleanup', { method: 'POST' });
}

export async function emergencyMemoryCleanup(): Promise<any> {
	return fetchJson('/api/v1/system/memory/emergency', { method: 'POST' });
}

export async function getAvailableModels(): Promise<{ cloud: any[]; huggingface: any[]; ollama: any[] }> {
	try {
		const data = await fetchJson('/api/v1/system/models');
		return data?.data || { cloud: [], huggingface: [], ollama: [] };
	} catch {
		return { cloud: [], huggingface: [], ollama: [] };
	}
}
