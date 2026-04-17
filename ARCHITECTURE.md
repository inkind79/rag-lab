# Architecture

A short tour of the codebase for contributors. Tells you where any given
behavior lives so you don't have to grep from scratch.

## Shape at 10,000 feet

```
   Browser                      FastAPI (uvicorn)                 Local deps
┌─────────────┐   SSE / JSON  ┌──────────────────────┐       ┌─────────────────┐
│ SvelteKit   │ ───────────►  │  Middlewares         │       │ Ollama          │
│ SPA (build) │ ◄───────────  │   CORS               │       │ (localhost:11434)│
│             │   httpOnly    │   CSRF origin check  │       └─────────────────┘
└─────────────┘   auth cookie │   Security headers   │       ┌─────────────────┐
                              │   Prometheus meter   │       │ LanceDB          │
                              │   Cache headers      │       │ (.lancedb/)      │
                              │   Rate limit (slowapi)│      └─────────────────┘
                              └──────────────────────┘       ┌─────────────────┐
                                          │                  │ BM25 index       │
                                          ▼                  │ (.bm25/)         │
                              ┌──────────────────────┐       └─────────────────┘
                              │  Routers (src/api/   │       ┌─────────────────┐
                              │  routers/*.py)       │       │ ColPali / OCR    │
                              │  auth  sessions      │       │ (HF weights)     │
                              │  documents  chat     │       └─────────────────┘
                              │  settings  scores    │       ┌─────────────────┐
                              │  templates feedback  │       │ Mem0 / ChromaDB  │
                              │  system              │       │ (optional)       │
                              └──────────────────────┘       └─────────────────┘
                                          │
                                          ▼
                              ┌──────────────────────┐
                              │  Services + models   │
                              │  src/services/       │
                              │  src/models/         │
                              └──────────────────────┘
                                          │
                                          ▼
                              ┌──────────────────────┐
                              │  Disk state          │
                              │  sessions/*.json     │
                              │  uploaded_documents/ │
                              │  .lancedb/  .bm25/   │
                              │  cache/ (OCR, search)│
                              │  data/users.db       │
                              └──────────────────────┘
```

Single-process FastAPI app. SPA is built once (`npm run build`) and served
out of `frontend/build/` by the same process. No separate web server.

## Entry point

`fastapi_app.py` is the top of the stack:

1. Platform shim (Windows/Linux differences) — `src/utils/platform_shim.py`
2. Optional Sentry init — `src/api/observability.py`
3. FastAPI app construction with a lifespan that:
   - creates required dirs
   - initializes the FastAPI-Users SQLite DB
   - warms `RAGRetriever` + `OCRRAGRetriever` via `src/models/retriever_manager.py`
   - seeds default prompt templates
4. Middlewares stacked in this order (outermost first):
   - `CORSMiddleware` — origin allowlist from `CORS_ORIGINS`
   - `CSRFOriginMiddleware` — rejects cross-origin writes with the auth cookie
   - `SecurityHeadersMiddleware` — X-Frame-Options, Permissions-Policy, opt-in CSP
   - `StaticAssetCacheMiddleware` — per-path Cache-Control
   - `PrometheusMiddleware` — HTTP volume + latency counters
   - `SlowAPIMiddleware` + `PathRateLimitMiddleware` — per-IP rate limits
5. Routers — one module per resource under `src/api/routers/`
6. SPA catch-all — serves `frontend/build/index.html` for anything else

## Auth

- **Storage**: `data/users.db` (SQLite via `aiosqlite`)
- **Library**: fastapi-users with a username column (`src/api/users.py`)
- **Token**: JWT in an `httpOnly` `auth_token` cookie (`SameSite=lax`)
- **Secret**: `JWT_SECRET` env var; in dev it persists to `data/.jwt_secret`
  so sessions survive restarts. In production (`APP_ENV=production`) the
  env var is **required** — the app refuses to start otherwise.
- **Login hardening**: `/auth/login` always runs one bcrypt round (against
  a dummy hash when the user isn't found) so missing-user and wrong-password
  cases are timing-equivalent. Returns the same "Invalid credentials" 401.
- **Password policy**: `UserManager.validate_password` requires 8–128
  chars, one letter + one digit, no username substring.
- **Rate limiting**: 5 login attempts/min, 3 registrations/min (and 20/hour)
  per IP. Configured via `RATE_LIMIT_*` env vars.

## Sessions

Each user has many sessions; a session is one conversation with one uploaded
document set. Storage is **one JSON file per session** at
`sessions/{uuid}.json`, not a relational table — lets the whole thing ship
without a server.

- **Manager**: `src/services/session_manager/manager.py`
- **Concurrency**: every read/write goes through `src/utils/file_lock.py`
  which combines an in-process `threading.Lock` with `fcntl.flock()` on an
  adjacent `.lock` file. Writes are tmp-file + atomic rename. Stale locks
  (crashed writer) are cleared after a timeout.
- **Defaults**: global defaults live in `DEFAULT_GLOBAL_SETTINGS` at the
  top of `manager.py`. Per-user overrides live in `user_defaults/{user_id}.json`.
- **Listing**: `get_all_sessions()` iterates the session folder and filters
  by the `user_id` field inside each file.
- **Pagination**: `GET /api/v1/sessions/{uuid}/chat_history?limit=50&before=N`
  walks the chat history backward in 50-message pages.

## Documents + retrieval

Upload flow (`src/api/routers/documents.py` → `src/services/document_processor/`):

1. User uploads PDFs / images → saved to `uploaded_documents/{session_uuid}/`
   with filenames sanitized via `src/utils/path_safety.safe_filename`.
2. Background pipeline extracts pages, optionally runs OCR, and hands pages
   to the active retriever.
3. Indexing writes to:
   - `.lancedb/{session}/...` for ColPali visual embeddings
   - `.bm25/{session}/...` for BM25 text index
   - `static/images/{session}/...` for PNG thumbnails served to the UI

Retrieval flow (`src/models/rag_retriever.py` → `retrieve_documents`):

1. Optional query expansion via `_expand_query_with_context` — if the
   query has pronouns like "it/this", Ollama rewrites it with chat history
   context.
2. Optional HyDE expansion — generate a hypothetical answer, concat with
   the query. Opt-in via `session.use_hyde`.
3. Fan-out to the configured retriever (ColPali / BM25 / dense / hybrid).
4. Score-slope filter (`apply_score_slope`) trims the tail when scores
   drop off sharply. Opt-in via `session.use_score_slope`.
5. Token budget filter (`apply_token_budget_filter`) keeps the top-N that
   fit in the generation model's context window.
6. Optional LLM rerank — the retrieved pages are scored 0-10 by a small
   LLM. Opt-in via `session.use_llm_rerank`. Implementation:
   `src/models/reranker.py`.

## Generation

`src/api/routers/chat.py` POST `/api/v1/chat/stream` — Server-Sent Events.

- Body: `{query, session_uuid, is_rag_mode, is_batch_mode, pasted_images}`
- Flow: decode pasted images → delegate to `src/services/response_generator/
  generator.generate_streaming_response()` → stream SSE chunks of types
  `response`, `reasoning`, `images`, `model_info`, `stage`, `error`,
  `doc_start`/`doc_complete` (batch).
- LLM: Ollama only today. Client is centralized in
  `src/utils/ollama_client.py` with env-driven connect/read timeouts
  (`OLLAMA_CONNECT_TIMEOUT`, `OLLAMA_READ_TIMEOUT`).
- All downstream calls to Ollama respect those timeouts so a hung backend
  can't wedge a worker thread forever.

## Caches

- `src/models/embedding_cache.py` — embedding cache on disk, `.npy` +
  JSON meta sidecar (no pickle).
- `src/models/search_results_cache.py` — query → results cache, JSON.
- `src/models/vector_stores/bm25_store.py` — BM25 index, bm25s native
  safe format (numpy + JSON, `allow_pickle=False`).
- `src/models/ocr/ocr_cache.py` — OCR text cache by image hash.
- All cache dirs are created via `src/utils/secure_dirs.secure_makedirs`
  with mode `0o700`.

## Memory (optional)

Cross-session conversational memory via Mem0 + ChromaDB — disabled by
default (`memory_enabled: False` in new sessions). When on, the chat
streaming endpoint appends finalized turns to Mem0 in non-RAG mode only
— RAG mode has its own document-grounded context and shouldn't pollute
cross-session memory.

## Observability

- **Logs**: `logs/app.log`, human-readable by default. Set `LOG_FORMAT=json`
  for structured one-line-per-record output suitable for Loki/Elastic/Datadog.
  A redaction filter scrubs `Authorization: Bearer …`, `hf_…`, `sk-…`, and
  registered secrets (JWT_SECRET, OLLAMA_API_KEY) from every record before
  it hits a handler.
- **Errors**: opt-in Sentry via `SENTRY_DSN`. Init happens before app
  construction so startup failures are captured.
- **Metrics**: `/metrics` (admin-only) exposes HTTP volume + latency,
  cache hits, LLM inference duration, retrieval duration — standard
  Prometheus text format. See `src/api/metrics.py`.

## Evaluation harness

`src/eval/` — measure retrieval quality against a golden set without
depending on the live retriever stack. CLI: `python -m src.eval.cli
--golden tests/fixtures/eval/sample_golden_set.json`. Metrics: precision@k,
recall@k, MRR, NDCG (binary + graded).

## Frontend

`frontend/` — SvelteKit 5 + Svelte 5 runes + TypeScript + Vite.

- Entry points: `src/routes/+layout.svelte` (sidebar, session management,
  theme, modals) and `src/routes/+page.svelte` (chat UI).
- Reactive state: `src/lib/stores/` — `session`, `chat`, `toast`.
- API client: `src/lib/api/client.ts` — thin fetch wrapper, all calls go
  through the SvelteKit `credentials: 'include'` fetch so the auth cookie
  rides along.
- Generated types: `src/lib/types/generated.ts` — regenerate with
  `python scripts/gen_types.py` when a pydantic request/response model
  changes. CI drift-checks the committed output against the generator.
- Theming: `data-theme="light"|"dark"` on `<html>`, CSS variables in
  `+layout.svelte` `:root` and `:root[data-theme="dark"]`.

## Tests + CI

- `tests/` — pytest + pytest-asyncio. `conftest.py` stubs the heavy
  lifespan paths (ColPali loading, template seeding) so a TestClient
  boots in milliseconds without a GPU.
- Run: `pytest` from the repo root. CI installs only `requirements-test.txt`
  (~30s) — no torch/colpali/lancedb needed because the models aren't
  imported at module load time.
- CI workflows in `.github/workflows/`:
  - `test.yml` — pytest on Python 3.10/3.11/3.12
  - `lint.yml` — ruff on tests/ + new utils, frontend svelte-check + build
  - `security.yml` — gitleaks (secret scan) + bandit

## Filesystem cheat sheet

```
fastapi_app.py                top-level ASGI app + middleware stack
pyproject.toml                pytest + ruff config
requirements.txt              full runtime deps (torch, colpali, lancedb)
requirements-test.txt         CI-minimal deps (no ML)

src/api/
  config.py                   env → config constants, JWT secret persistence
  users.py                    fastapi-users setup, password policy
  deps.py                     get_current_user, get_session_id, etc.
  rate_limit.py               slowapi Limiter + path-based guard
  csrf.py                     Origin-header CSRF middleware
  security_headers.py         X-Frame-Options, CSP, Permissions-Policy
  static_cache.py             per-path Cache-Control middleware
  metrics.py                  Prometheus counters + middleware
  observability.py            Sentry init
  routers/                    one file per resource
    auth.py  sessions.py  documents.py  chat.py
    settings.py  system.py  scores.py  templates.py  feedback.py

src/services/
  session_manager/manager.py  session JSON read/write + defaults merging
  document_processor/         upload → page extract → index
  response_generator/         LLM streaming + chat history persistence

src/models/
  rag_retriever.py            primary retriever (ColPali + slope + budget + rerank)
  reranker.py                 LLM reranker primitive
  query_expansion.py          HyDE + multi-query primitives
  retriever_manager.py        singleton retrievers warmed at startup
  embedding_cache.py          disk cache for embeddings (npy)
  search_results_cache.py     disk cache for query results (json)
  vector_stores/
    bm25_store.py             BM25 index per session
    lancedb_manager.py        ColPali embeddings in LanceDB
    score_analysis.py         score-slope + token-budget filters
  ocr/                        OCR wrappers (EasyOCR, SmolDocling, VLM)
  llm_handlers/               provider-specific chat adapters (Ollama)
  memory/                     Mem0 + ChromaDB bindings

src/utils/
  logger.py                   structured/text log formatters + filters
  log_redaction.py            scrubs Bearer/hf_/sk-/registered-secret tokens
  file_lock.py                fcntl + threading.Lock JSON helpers
  path_safety.py              safe_filename + safe_join
  secure_dirs.py              0o700 mkdir
  ollama_client.py            URL/auth/timeout/client factory
  memory_management/          cross-process IPC + cleanup

src/eval/                     metrics + golden-set harness + CLI

frontend/
  src/app.html                SPA shell
  src/hooks.client.ts         frontend Sentry init (opt-in)
  src/routes/
    +layout.svelte            sidebar, modals, theme, auth bootstrap
    +page.svelte              chat UI
    login/+page.svelte        login form
    register/+page.svelte     register form
  src/lib/
    api/client.ts             fetch wrapper for all /api/v1/* calls
    stores/                   session, chat, toast
    types/generated.ts        pydantic-derived interfaces (regen with scripts/gen_types.py)
    components/               SettingsModal, AILabModal, DocumentPanel,
                              RetrievedImages, FeedbackSection, ToastContainer,
                              Markdown, ModelBadge, TemplatePanel

tests/                        pytest suite + fixtures
  conftest.py                 autouse isolated_workdir + fastapi_app fixture
  fixtures/eval/              sample golden set

scripts/
  gen_types.py                pydantic → TypeScript generator

.github/workflows/
  test.yml  lint.yml  security.yml
```

## Environment variables

Grouped by concern. Unless marked **required**, each has a sensible default.

**App mode**
- `APP_ENV` — `development` or `production`. Production requires `JWT_SECRET`.

**Auth**
- `JWT_SECRET` — **required in production**. Persisted to `data/.jwt_secret` in dev.
- `RATE_LIMIT_LOGIN` / `RATE_LIMIT_REGISTER` / `RATE_LIMIT_UPLOAD` — slowapi-format strings (e.g. `5/minute`, `20/hour`).

**Network**
- `CORS_ORIGINS` — comma-separated allowlist of scheme://host[:port].
- `CSRF_ALLOWED_ORIGINS` — explicit CSRF allowlist; falls back to `CORS_ORIGINS`.
- `CSRF_DISABLE` — `true` to turn CSRF off (tests only).

**Security headers**
- `CSP_ENABLE` — `true` to emit Content-Security-Policy.
- `CSP_POLICY` — override the default CSP string.
- `FRAME_DENY` — default `true` (X-Frame-Options: DENY).

**Ollama**
- `OLLAMA_BASE_URL` — default `http://localhost:11434`.
- `OLLAMA_API_KEY` — for Ollama Cloud models (`*:cloud` suffix).
- `OLLAMA_CONNECT_TIMEOUT` / `OLLAMA_READ_TIMEOUT` — seconds; defaults 10 / 1800.

**Observability**
- `LOG_FORMAT` — `json` for structured logs.
- `SENTRY_DSN` — enable Sentry. Plus `SENTRY_ENVIRONMENT`, `SENTRY_RELEASE`,
  `SENTRY_TRACES_SAMPLE_RATE`, `SENTRY_SEND_PII`, `PUBLIC_SENTRY_DSN` (frontend).

**Mem0 / memory**
- `MEM0_LLM_MODEL`, `MEM0_EMBED_MODEL`, `MEM0_CHROMA_PATH`.

**HuggingFace**
- `HF_HUB_OFFLINE=1` to skip online checks.

**Session data**
- `SESSION_FOLDER` (default `sessions`), `UPLOAD_FOLDER` (`uploaded_documents`),
  `STATIC_FOLDER` (`static`), `LANCEDB_FOLDER` (`.lancedb`), `PROMPT_TEMPLATES_DIR`.

## Development commands

```bash
# Full dev stack
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install && npm run dev       # Vite on :5173
uvicorn fastapi_app:app --reload --port 8000    # in another terminal

# Run tests (no ML deps needed)
pip install -r requirements-test.txt
pytest                                           # ~5s full suite

# Lint
ruff check .
cd frontend && npm run check

# Regenerate TS types after changing pydantic models
python scripts/gen_types.py

# Retrieval eval
python -m src.eval.cli --golden tests/fixtures/eval/sample_golden_set.json
```

## Design principles (observed)

- **Local-first**: nothing leaves the box unless an env var explicitly opts
  in. Sentry, Ollama Cloud, external log shippers all start disabled.
- **No single-point-of-failure daemons**: session JSON on disk, BM25 index
  on disk, SQLite for users. One `uvicorn` process is the whole deployment.
- **Defense in depth for a self-hosted OSS app**: path-traversal guards,
  atomic writes, rate limits, CSRF check, CSP, log redaction, secure dir
  perms, pickle-free cache format — even though most deployments are
  single-user local, the cost of these is one middleware each.
- **No pickle**: caches use numpy/JSON, inter-process payloads use
  `torch.save(weights_only=True)` + JSON. Pickle deserialization is RCE.
- **Soft-fail observability**: if Sentry, Prometheus, or any extra logger
  fails, the app still serves requests. Observability must never block
  the hot path.
