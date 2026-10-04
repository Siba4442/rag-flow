<h1 align="center">RAG Flow</h1>

<p align="center"><strong>Every claim, cited.</strong><br />
Upload your documents, ask questions, and get answers where every sentence points to the passage it came from.</p>

---

RAG Flow is a self-hosted document question-answering app. It ingests your files, indexes them in a vector database, and answers questions with an agent that searches, checks its evidence, and cites it. You can verify each claim against the source with one hover.

It's built on **LangGraph** and **FastAPI**, and has a fast, framework-free web UI.

## Highlights

- **Cited answers.** Each answer links its claims to numbered passages. Hover a citation and the passage lights up beside it. A coverage bar shows how many claims are backed by a source.
- **An agent that checks its evidence.** The chat graph decides when to search, grades whether the retrieved passages are relevant, and rewrites the query if they aren't, before it answers.
- **Your vector store.** PGVector by default; Pinecone, Chroma or MongoDB Atlas with one setting.
- **Your models.** OpenAI, Anthropic, Google, AWS Bedrock, Ollama, Hugging Face, or any OpenAI-compatible endpoint such as OpenRouter, for both chat and embeddings.
- **Private by design.** Optional JWT auth. Every read, search and delete is filtered by owner inside the database query, so one user can never see another's files or conversations.
- **Conversation memory.** Threads are stored in Postgres and survive restarts.
- **Many file types.** PDF, Word, Excel, PowerPoint, Markdown, CSV, EPUB, XML, JSON and source code. Optional LlamaParse or Docling handle scanned PDFs and complex layouts.
- **Works with LibreChat.** The document API (`/embed`, `/query`, `/query_multiple`, `/documents`, …) speaks the same protocol, so RAG Flow can serve as LibreChat's RAG backend.

## How it works

Three LangGraph graphs do the work:

| Graph | Steps | Used by |
|---|---|---|
| **Ingest** | load → split → embed and store in parallel batches → finalize | uploads |
| **Retrieve** | embed query (cached) → owner-scoped search → distance threshold | search, chat |
| **Chat** | decide to search → retrieve → grade relevance → rewrite the query or answer | `/chat` |

- **Failed uploads roll back.** If any embedding batch fails, every chunk from that upload is removed. Earlier uploads of the same file are kept.
- **The model can't widen its own access.** It only chooses the search text. Which files and owners it can see come from the request.

```
rag-flow/
  backend/    Python 3.12, FastAPI, LangGraph  (src/ragflow)
  frontend/   Vite + TypeScript, no framework
```

## Quick start

Requirements: Docker. For local development you also need [uv](https://docs.astral.sh/uv/) and Node 22+.

```bash
cp .env.example .env        # set POSTGRES_PASSWORD, your model keys and CHAT_MODEL
docker compose up -d --build
```

Open **http://127.0.0.1:8000**. The design system is at **/design**.

```bash
docker compose down         # stop (data is kept in Docker volumes)
docker compose logs -f app  # follow logs
```

### Using OpenRouter (or any OpenAI-compatible API)

```bash
EMBEDDINGS_PROVIDER=openai
RAG_OPENAI_BASEURL=https://openrouter.ai/api/v1
RAG_OPENAI_API_KEY=sk-or-...
EMBEDDINGS_MODEL=qwen/qwen3-embedding-8b
RAG_CHECK_EMBEDDING_CTX_LENGTH=false

OPENAI_API_KEY=sk-or-...
OPENAI_API_BASE=https://openrouter.ai/api/v1
CHAT_MODEL=openai:<provider>/<model>
```

The `openai:` prefix in `CHAT_MODEL` selects the OpenAI-compatible client; the rest is the model id sent to the API. Choose your embeddings model before uploading anything, because switching later means re-uploading.

### Local development

```bash
docker compose up -d db                       # Postgres + pgvector on 127.0.0.1:5433

cd backend
uv sync --extra openai --extra anthropic      # more: chroma, pinecone, mongo, ollama, google,
uv run ragflow                                #   bedrock, huggingface, llamaparse, docling
                                              # API on :8000, docs at /docs
cd ../frontend
npm install
npm run dev                                   # UI on :5173, proxies /api to :8000
```

## The interface

The UI uses a symmetric three-column layout.

- **Left:** navigation (Home, Ask, Search passages), your library and your conversations. Tick a file to put it in scope. A status card shows the connected models and store.
- **Centre:**
  - **Home** shows an ask panel with suggested questions, library and activity stats, and recent answers.
  - **Ask** shows the conversation.
  - **Search passages** ranks stored passages with no generation.
- **Right:** the evidence. Numbered sources, the highlighted passage for the citation you're hovering, claim coverage, and an *Add files* drop zone.

Light, dark and auto themes are included. One accent, gold, is reserved for evidence and selection.

## Vector stores

| `VECTOR_DB_TYPE` | Backend | Notes |
|---|---|---|
| `pgvector` (default) | `langchain_postgres.PGVectorStore` | `rag_chunks` table with indexed owner and file columns; HNSW index up to 2000 dimensions (larger embeddings use exact search within the selected files). Chat memory uses the same database. |
| `chroma` | Chroma | `CHROMA_HOST` for a server, or a local `CHROMA_PERSIST_DIR` |
| `pinecone` | Pinecone | Works with serverless indexes |
| `atlas-mongo` | MongoDB Atlas Vector Search | The search index must allow filtering on `file_id` and `user_id` |

Chat memory persists with `pgvector`. With the other stores it is kept in process.

## Configuration

All settings come from environment variables (see `.env.example`).

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDINGS_PROVIDER` / `EMBEDDINGS_MODEL` | `openai` / provider default | `openai`, `azure`, `google_genai`, `vertexai`, `bedrock`, `ollama`, `huggingface`, `huggingfacetei` |
| `CHAT_MODEL` | `openai:gpt-5-mini` | Any `init_chat_model` id, e.g. `anthropic:claude-sonnet-5-5`, `ollama:llama3.2` |
| `GRADER_MODEL` | = `CHAT_MODEL` | A cheaper model for relevance grading |
| `CHAT_RETRIEVAL_K` / `CHAT_MAX_REWRITES` | `6` / `2` | Passages per search / query rewrites per turn |
| `VECTOR_DB_TYPE` | `pgvector` | See *Vector stores* |
| `POSTGRES_*`, `DB_HOST`, `DB_PORT` | | Database connection |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1500` / `100` | Text splitting |
| `EMBEDDING_BATCH_SIZE` / `PARALLEL_EXECUTION` | `500` / `2` | Ingestion batching and concurrency |
| `RAG_DISTANCE_THRESHOLD` | unset | Drop matches farther than this distance |
| `PDF_PARSER` | `pypdf` | `pypdf`, `llamaparse` (needs `LLAMA_CLOUD_API_KEY`) or `docling` |
| `JWT_SECRET` | unset | Turns on authentication |
| `ENTITY_ID_POLICY` | `claim` | How a caller-supplied `entity_id` is treated (see *Security*) |
| `LOCAL_EMBED_SCOPE` | `owner` | Which paths `/local/embed` may read |
| `CORS_ORIGINS` | `*` | Comma-separated |
| `SERVE_FRONTEND` | `false` | Serve the built UI from FastAPI (on in Docker) |

If `CHAT_MODEL` can't be initialised (for example, a missing key), chat is disabled and `/chat` returns 503. The rest of the app keeps working.

## API

| Endpoint | Purpose |
|---|---|
| `POST /embed`, `POST /embed-upload` | Upload and index a file |
| `POST /local/embed` | Index a file already on the server's upload volume |
| `POST /text` | Extract text without indexing |
| `GET /files`, `GET /ids` | List your files |
| `GET /documents`, `DELETE /documents` | Read or delete files' passages |
| `GET /documents/{id}/context` | A file's full text, in order |
| `POST /query`, `POST /query_multiple` | Ranked passages for a query |
| `POST /chat` | Streamed, cited answer over SSE (`token`, `step`, `sources`, `done`, `error`) |
| `GET` / `DELETE /chat/threads/{id}` | Conversation history |
| `GET /info`, `GET /health` | Connected models and store; health check |

Interactive docs are at `/docs`.

## Security

- **Authentication:** set `JWT_SECRET` to require HS256 bearer tokens. The token's `id` claim is the user.
- **Owner scoping** is enforced inside every store query, before ranking. A file id alone never grants access.
- **`ENTITY_ID_POLICY`** controls requests that name an `entity_id` (for example, an agent sharing files across users):
  - `claim` (default) honours it only if it appears in the token's `entities` claim.
  - `trust` accepts it as asserted. Use this behind a gateway that authorises entities itself, such as a LibreChat deployment.
  - `deny` ignores it.
- **Uploads** go to a private temp folder under random names, so filenames can't steer where they're written. With `LOCAL_EMBED_SCOPE=owner`, `/local/embed` only reads from the caller's own folder.
- **Errors** return a reference id instead of internal details. The details go to the server log.

## Importing existing LangChain PGVector data

If you already have embeddings in LangChain's classic `langchain_pg_embedding` tables, copy them into RAG Flow without re-embedding:

```bash
cd backend
uv run python scripts/migrate_legacy_pgvector.py --dry-run
uv run python scripts/migrate_legacy_pgvector.py --collection <name>
```

Keep the same embeddings model for imported data.

## Development

```bash
cd backend
uv run ruff check && uv run ruff format --check
uv run mypy                       # strict
uv run pytest                     # unit tests
uv run pytest -m integration      # pgvector, checkpointer, migration (needs Docker)

cd ../frontend
npm run typecheck && npm run build
```

- **Language servers:** Pyright reads `pyrightconfig.json` at the repo root, which points at `backend/.venv`. TypeScript uses `frontend/tsconfig.json`.
- **MCP:** `.mcp.json` registers the LangChain docs server and a Conventional Commit message generator for Claude Code.
- **WSL:** for faster tooling, keep the repo on the Linux filesystem (e.g. `~/code`) rather than `/mnt/c`.
