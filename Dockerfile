# syntax=docker/dockerfile:1
# One image: FastAPI backend + the built frontend served at /ui.

# ---- frontend ---------------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- backend deps -----------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
# Provider/store extras to bake in, e.g. "openai anthropic chroma pinecone".
ARG EXTRAS="openai anthropic ollama"
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY backend/pyproject.toml backend/uv.lock backend/README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project $(printf -- '--extra %s ' $EXTRAS)
COPY backend/src ./src
COPY backend/scripts ./scripts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev $(printf -- '--extra %s ' $EXTRAS)

# ---- runtime ----------------------------------------------------------------
FROM python:3.12-slim-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends pandoc libmagic1 libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 app \
 && mkdir -p /data/uploads && chown app /data/uploads
WORKDIR /app
COPY --from=build --chown=app /app /app
COPY --from=web --chown=app /web/dist /app/frontend-dist
ENV PATH="/app/.venv/bin:$PATH" \
    RAG_UPLOAD_DIR=/data/uploads \
    SERVE_FRONTEND=true \
    FRONTEND_DIST=/app/frontend-dist \
    SCARF_NO_ANALYTICS=true
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health').status!=200)"
CMD ["ragflow"]
