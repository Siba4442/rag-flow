// Typed client for the RAG Flow backend (LibreChat-compatible routes + /files + /chat).

import { getSettings } from "./store";
import { readSSE } from "./sse";

export interface FileSummary {
  file_id: string;
  filename: string | null;
  chunks: number;
  added_at?: number | null;
}

export interface SystemInfo {
  chat_model: string | null;
  embeddings_model: string;
  embeddings_provider: string;
  vector_store: string;
}

export interface ChunkDoc {
  id?: string | null;
  page_content: string;
  metadata: Record<string, unknown>;
}

export type ScoredChunk = [ChunkDoc, number];

export interface Source {
  n: number;
  file_id: string | null;
  filename: string | null;
  page: number | null;
  chunk_index: number | null;
  score: number;
  snippet: string;
}

export interface ThreadMessage {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

// Dev: Vite proxies /api to the backend. Built: served by FastAPI on the same origin.
const DEFAULT_BASE = import.meta.env.DEV ? "/api" : "";

function base(): string {
  return (getSettings().baseUrl || DEFAULT_BASE).replace(/\/$/, "");
}

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const { token } = getSettings();
  return token ? { Authorization: `Bearer ${token}`, ...extra } : extra;
}

function withEntity(params: URLSearchParams = new URLSearchParams()): URLSearchParams {
  const { entityId } = getSettings();
  if (entityId) params.set("entity_id", entityId);
  return params;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${base()}${path}`, { ...init, headers: headers(init.headers as Record<string, string>) });
  } catch {
    throw new ApiError(0, "Can't reach the RAG Flow server. Check the API address in connection settings.");
  }
  if (!res.ok) throw new ApiError(res.status, await errorText(res));
  return (await res.json()) as T;
}

async function errorText(res: Response): Promise<string> {
  try {
    const body: unknown = await res.json();
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
    }
  } catch {
    /* not JSON */
  }
  if (res.status === 401) return "The server needs an access token. Add one in connection settings.";
  if (res.status === 403) return "This token isn't allowed to use that entity id.";
  return `Request failed (${res.status}).`;
}

export const api = {
  health: () => request<{ status: string }>("/health"),

  files: () => request<FileSummary[]>(`/files?${withEntity()}`),

  info: () => request<SystemInfo>("/info"),

  upload(file: File, fileId: string): Promise<{ status: boolean; message: string }> {
    const form = new FormData();
    form.set("file_id", fileId);
    form.set("file", file);
    const { entityId } = getSettings();
    if (entityId) form.set("entity_id", entityId);
    return request("/embed", { method: "POST", body: form });
  },

  deleteFiles: (ids: string[]) =>
    request<{ message: string }>(`/documents?${withEntity()}`, {
      method: "DELETE",
      body: JSON.stringify(ids),
      headers: { "Content-Type": "application/json" },
    }),

  context: (fileId: string) =>
    request<string>(`/documents/${encodeURIComponent(fileId)}/context?${withEntity()}`),

  search: (query: string, fileIds: string[], k = 8) =>
    request<ScoredChunk[]>("/query_multiple", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, file_ids: fileIds, k, entity_id: getSettings().entityId || null }),
    }).catch((err: unknown) => {
      if (err instanceof ApiError && err.status === 404) return [] as ScoredChunk[];
      throw err;
    }),

  thread: (threadId: string) =>
    request<{ messages: ThreadMessage[] }>(`/chat/threads/${encodeURIComponent(threadId)}`),

  deleteThread: (threadId: string) =>
    request<{ message: string }>(`/chat/threads/${encodeURIComponent(threadId)}`, { method: "DELETE" }),
};

export interface ChatHandlers {
  onToken(text: string): void;
  onStep(node: string): void;
  onSources(sources: Source[]): void;
}

export async function streamChat(
  body: { message: string; thread_id: string; file_ids: string[] },
  handlers: ChatHandlers,
  signal: AbortSignal,
): Promise<void> {
  const { entityId } = getSettings();
  let res: Response;
  try {
    res = await fetch(`${base()}/chat`, {
      method: "POST",
      signal,
      headers: headers({ "Content-Type": "application/json", Accept: "text/event-stream" }),
      body: JSON.stringify({ ...body, entity_id: entityId || null }),
    });
  } catch (err) {
    if (signal.aborted) return;
    throw new ApiError(0, "Can't reach the RAG Flow server. Check the API address in connection settings.");
  }
  if (!res.ok || !res.body) throw new ApiError(res.status, await errorText(res));

  for await (const event of readSSE(res.body)) {
    const data: unknown = event.data ? JSON.parse(event.data) : null;
    switch (event.event) {
      case "token":
        handlers.onToken((data as { text: string }).text);
        break;
      case "step":
        handlers.onStep((data as { node: string }).node);
        break;
      case "sources":
        handlers.onSources(data as Source[]);
        break;
      case "error":
        throw new ApiError(500, (data as { detail: string }).detail);
      case "done":
        return;
    }
  }
}
