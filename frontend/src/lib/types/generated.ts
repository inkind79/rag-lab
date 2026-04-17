// ──────────────────────────────────────────────────────────────────────────
// GENERATED FILE — DO NOT EDIT
// Regenerate with:  python scripts/gen_types.py
//
// Source: Pydantic request/response models in src/api/routers/*.py
// ──────────────────────────────────────────────────────────────────────────

export interface LoginRequest {
	username: string;
	password: string;
}

export interface CreateSessionRequest {
	name?: string | null;
}

export interface RenameRequest {
	name: string;
}

export interface ChatRequest {
	query: string;
	session_uuid?: string | null;
	is_rag_mode?: boolean;
	is_batch_mode?: boolean;
	has_pasted_images?: boolean;
	pasted_images?: Array<string>;
}

export interface SelectionUpdate {
	selected_docs?: Array<string>;
}

export interface DeleteDocsRequest {
	filenames: Array<string>;
}

export interface SettingsUpdate {
	generation_model?: string | null;
	indexer_model?: string | null;
	retrieval_count?: number | null;
	distance_metric?: string | null;
	similarity_threshold?: number | null;
	use_score_slope?: boolean | null;
	rel_drop_threshold?: number | null;
	abs_score_threshold?: number | null;
	use_ocr?: boolean | null;
	ocr_engine?: string | null;
	resized_height?: number | null;
	resized_width?: number | null;
	cloud_history_limit?: number | null;
	local_history_limit?: number | null;
	show_score_viz?: boolean | null;
	retrieval_method?: string | null;
	text_embedding_model?: string | null;
	chunk_size?: number | null;
	chunk_overlap?: number | null;
	hybrid_visual_weight?: number | null;
	use_hyde?: boolean | null;
	hyde_model?: string | null;
	use_llm_rerank?: boolean | null;
	llm_rerank_model?: string | null;
	model_params?: Record<string, any> | null;
}

export interface OCRSettings {
	use_ocr?: boolean | null;
	ocr_engine?: string | null;
}

export interface OllamaKeyUpdate {
	api_key?: string;
}

export interface SaveScoresRequest {
	scores: any;
}

export interface SlopeToggle {
	enabled: boolean;
}

export interface SlopeParams {
	rel_drop_threshold?: number | null;
	abs_score_threshold?: number | null;
}
