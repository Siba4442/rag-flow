# RAG Flow

A LangGraph rewrite of [LibreChat's RAG API](https://github.com/danny-avila/rag_api), plus a document chat with cited answers and a small web UI.

- **Drop-in for LibreChat.** Same endpoints, form fields and response shapes as rag-api (`/embed`, `/query`, `/query_multiple`, `/documents`, …).
- **Built on LangGraph.** Ingestion, retrieval and chat are each a `StateGraph`.
- **Vector stores.** PGVector by default; Pinecone, Chroma or MongoDB Atlas with one setting.
- **Chat with your files.** `POST /chat` streams answers over SSE. Every claim is cited to a stored passage, and conversations are kept in Postgres.
- **Reading-room UI.** Vite + TypeScript, no framework. Sources sit in the margin and light up when you hover a citation.

```
rag-flow/
  backend/    Python 3.12, uv, FastAPI, LangGraph   (src/ragflow)
  frontend/   Vite + TypeScript, no framework
  rag-api/    original LibreChat service, kept for reference (git-ignored)
```

## How it works

| Graph | Nodes | Used by |
|---|---|---|
| `ingest` | `load → split → store_batch` (fan-out via `Send`, `PARALLEL_EXECUTION` wide) `→ finalize` | `/embed`, `/embed-upload`, `/local/embed` |
| `retrieve` | `embed_query` (LRU-cached) `→ search → apply_threshold` | `/query`, `/query_multiple`, chat tool |
| `chat` | `generate_query_or_respond → retrieve → grade → rewrite_question \| generate_answer` | `/chat` |

Things that hold everywhere:

- **Owner scope is enforced in the store query, before ranking.** Every store method takes the caller's owners; a file id is never treated as an authorization. In chat, owners and file ids come from the run context, and the model only chooses the search text.
- **Failed batches roll back.** If any embedding batch fails, every chunk from that ingestion attempt is deleted. Earlier uploads of the same file are kept.
- **Sub-graphs keep no state.** `ingest` and `retrieve` compile with `checkpointer=False`, so conversation checkpoints only contain the conversation.

### Vector stores (`VECTOR_DB_TYPE`)

| Value | Backend | Notes |
|---|---|---|
| `pgvector` (default) | `langchain_postgres.PGVectorStore` | New table `rag_chunks` with indexed `file_id` / `user_id` columns and an HNSW index. Chat memory uses the same Postgres. |
| `chroma` | `langchain-chroma` | `CHROMA_HOST` for a server, otherwise a local `CHROMA_PERSIST_DIR` |
| `pinecone` | `langchain-pinecone` | Vector ids are prefixed with owner and file, so listing and deleting work on serverless indexes |
| `atlas-mongo` | `langchain-mongodb` | Ported from rag-api; the search index must allow filtering on `file_id` and `user_id` |
| `memory` | in-process | Tests and quick trials only |

Chat memory is persistent only with `pgvector`. With the other stores it is in-process and is lost on restart.

### Is LlamaIndex needed?

Not as a framework. Its indices, retrievers and query engines duplicate what LangChain and LangGraph already do here. It is worth using in one place: **LlamaParse**, for scanned PDFs, tables and complex layouts. Set `PDF_PARSER=llamaparse` (extra `llamaparse`, needs `LLAMA_CLOUD_API_KEY`). `PDF_PARSER=docling` is a free, local alternative. The default stays `pypdf`, as in rag-api.

### Why no frontend framework?

The UI has three views, little state and one streaming connection. Plain TypeScript modules (a typed API client, a small store and an escape-first renderer) produce a ~16 kB bundle. Vite provides the dev server, the `/api` proxy and the build. If the UI grows to include routing, auth flows or many components, Vue 3 would be the next step.

## Quick start (development)

Requirements: [uv](https://docs.astral.sh/uv/), Node 22+, and Docker for Postgres. On WSL, turn on *Docker Desktop → Settings → Resources → WSL integration* for your distro.

```bash
cp .env.example .env              # set POSTGRES_PASSWORD, RAG_OPENAI_API_KEY, CHAT_MODEL …
docker compose up -d db           # pgvector on 127.0.0.1:5433

cd backend
uv sync --extra openai --extra anthropic   # add: chroma, pinecone, mongo, ollama, google, bedrock, huggingface, llamaparse, docling
uv run ragflow                    # http://127.0.0.1:8000  (API docs at /docs)

cd ../frontend
npm install
npm run dev                       # http://localhost:5173  (proxies /api → :8000)
```

Run everything in one container. The UI is served at `/ui`:

```bash
docker compose up -d --build      # http://127.0.0.1:8000
RAGFLOW_EXTRAS="openai chroma" docker compose up -d --build   # pick extras
```

## Configuration

`.env` uses the same names as rag-api (`EMBEDDINGS_PROVIDER`, `EMBEDDINGS_MODEL`, `RAG_OPENAI_API_KEY`, `POSTGRES_*`, `DB_HOST`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_BATCH_SIZE`, `PARALLEL_EXECUTION`, `RAG_DISTANCE_THRESHOLD`, `JWT_SECRET`, `COLLECTION_NAME`, `ATLAS_*` …). An existing rag-api `.env` mostly carries over. New settings:

| Variable | Default | Purpose |
|---|---|---|
| `CHAT_MODEL` | `openai:gpt-5-mini` | Any `init_chat_model` id, e.g. `anthropic:claude-sonnet-5-5`, `ollama:llama3.2` |
| `GRADER_MODEL` | = `CHAT_MODEL` | A cheaper model for relevance grading |
| `CHAT_RETRIEVAL_K` / `CHAT_MAX_REWRITES` | `6` / `2` | Passages per search / query rewrites per turn |
| `ENTITY_ID_POLICY` | `claim` | See *Security changes* |
| `LOCAL_EMBED_SCOPE` | `owner` | See *Security changes* |
| `PGVECTOR_TABLE` | `rag_chunks` | Chunk table name |
| `PINECONE_API_KEY` / `PINECONE_INDEX` / `PINECONE_NAMESPACE` | | Pinecone |
| `CHROMA_HOST` / `CHROMA_PORT` / `CHROMA_PERSIST_DIR` | | Chroma |
| `PDF_PARSER` / `LLAMA_CLOUD_API_KEY` | `pypdf` | `pypdf`, `llamaparse` or `docling` |
| `CORS_ORIGINS` | `*` | Comma-separated |
| `SERVE_FRONTEND` / `FRONTEND_DIST` | `false` | Serve the built UI from FastAPI |

If `CHAT_MODEL` can't be initialised (for example, a missing key or package), chat is disabled and `/chat` returns 503. The rest of the API still works.

## Differences from rag-api

**Security changes (breaking).** These only apply when `JWT_SECRET` is set.

- **`ENTITY_ID_POLICY=claim`.** A caller-supplied `entity_id` is honoured only if it appears in the token's `entities` claim; otherwise the request gets 403. rag-api trusted it blindly, so any user could read or delete another owner's files. For LibreChat, set `ENTITY_ID_POLICY=trust` (rag-api behaviour), or issue tokens that include the claim.
- **`LOCAL_EMBED_SCOPE=owner`.** `/local/embed` only reads files under `{RAG_UPLOAD_DIR}/{owner}/`. Set `any` to restore rag-api behaviour.

**Bug fixes.**

- `/health` returns a real 503 when unhealthy. rag-api sent 200 with a JSON array.
- `/embed-upload` and `/local/embed` return `status: false` when storing fails. rag-api reported success.
- Error responses carry a reference id instead of raw exception text, which could leak DSNs or paths. The details are in the server log.
- Upload filenames and owner ids can't affect where temp files are written.
- CORS preflight works when JWT auth is on.

**Storage.** PGVector data lives in a new table. To copy rag-api data across without re-embedding:

```bash
cd backend
uv run python scripts/migrate_legacy_pgvector.py --dry-run
uv run python scripts/migrate_legacy_pgvector.py --collection testcollection
```

**New endpoints.** `GET /files`, `POST /chat` (SSE: `token`, `step`, `sources`, `done`, `error`), `GET`/`DELETE /chat/threads/{id}`.

## Development

```bash
cd backend
uv run ruff check && uv run ruff format --check
uv run mypy                       # strict
uv run pytest                     # unit tests
uv run pytest -m integration      # pgvector + checkpointer + migration (needs Docker)

cd ../frontend
npm run typecheck && npm run build
```

- **Language servers.** Pyright reads `pyrightconfig.json` at the repo root, which points at `backend/.venv`. The TypeScript server uses `frontend/tsconfig.json`.
- **MCP servers.** `.mcp.json` registers the LangChain docs server and a Conventional Commit message generator for Claude Code.
- **WSL.** For faster tooling, clone the repo inside the Linux filesystem (e.g. `~/code`) rather than `/mnt/c`.
