// Ask view: streamed answers with [n] citations linked to the evidence column.

import { api, ApiError, streamChat, type Source, type ThreadMessage } from "../api";
import { h } from "../dom";
import { renderAnswer } from "../render";
import { getSelected, getThread, getThreads, newId, recordActivity, upsertThread } from "../store";
import { coverageOf, linkCitations, renderMargin, sourceNotes } from "./margin";

const STEP_LABELS: Record<string, string> = {
  generate_query_or_respond: "Reading your question",
  retrieve: "Searching your files",
  rewrite_question: "Rephrasing the search",
  generate_answer: "Writing the answer",
};

export interface ChatDeps {
  page: HTMLElement;
  margin: HTMLElement;
  setStatus(text: string, tone?: "error" | "ok"): void;
  openReader(fileId: string, title: string): void;
}

let threadId = getThreads()[0]?.id ?? newId();
let controller: AbortController | null = null;

export const currentThread = (): string => threadId;

function intro(deps: ChatDeps): void {
  const selected = getSelected().length;
  deps.page.append(
    h(
      "div",
      { class: "intro" },
      h("h2", {}, "Ask your documents"),
      h(
        "p",
        {},
        selected
          ? `Answers come only from the ${selected === 1 ? "file" : `${selected} files`} in scope, and every claim points to the passage it came from.`
          : "Tick files in the library to put them in scope, then ask. Every claim will point to its passage.",
      ),
    ),
  );
}

function marginFor(deps: ChatDeps, sources: Source[], answer = ""): void {
  renderMargin(deps.margin, sourceNotes(sources), {
    heading: "Sources",
    empty: "Passages behind the latest answer appear here, numbered to match its citations.",
    coverage: sources.length ? coverageOf(answer) : null,
    onOpen: deps.openReader,
  });
}

function messageEl(msg: ThreadMessage): HTMLElement {
  if (msg.role === "user") return h("div", { class: "turn user" }, h("p", { class: "q" }, msg.content));
  const body = h("div", { class: "answer" });
  body.innerHTML = renderAnswer(msg.content); // renderAnswer escapes before formatting
  return h("div", { class: "turn assistant" }, body);
}

export async function showThread(id: string, deps: ChatDeps): Promise<void> {
  controller?.abort();
  threadId = id;
  deps.page.replaceChildren();
  marginFor(deps, []);
  try {
    const { messages } = await api.thread(id);
    if (!messages.length) return intro(deps);
    for (const m of messages) deps.page.append(messageEl(m));
    const last = messages.at(-1);
    marginFor(deps, last?.sources ?? [], last?.role === "assistant" ? last.content : "");
    linkCitations(deps.page, deps.margin);
    deps.page.scrollTop = deps.page.scrollHeight;
  } catch (err) {
    intro(deps);
    if (!(err instanceof ApiError && err.status === 503)) {
      deps.setStatus(err instanceof ApiError ? err.message : "Couldn't load this conversation.", "error");
    }
  }
}

export function newThread(deps: ChatDeps): void {
  controller?.abort();
  threadId = newId();
  deps.page.replaceChildren();
  marginFor(deps, []);
  intro(deps);
}

export async function ask(question: string, deps: ChatDeps): Promise<void> {
  deps.page.querySelector(".intro")?.remove();
  deps.page.append(messageEl({ role: "user", content: question }));
  const body = h("div", { class: "answer streaming", "aria-live": "polite" });
  const turn = h("div", { class: "turn assistant" }, body);
  deps.page.append(turn);
  turn.scrollIntoView({ block: "end" });

  const existing = getThread(threadId);
  upsertThread({ ...existing, id: threadId, title: existing?.title ?? question.slice(0, 80), updatedAt: Date.now() });
  controller?.abort();
  controller = new AbortController();
  let text = "";
  let sources: Source[] = [];

  try {
    await streamChat(
      { message: question, thread_id: threadId, file_ids: getSelected() },
      {
        onToken(t) {
          text += t;
          body.innerHTML = renderAnswer(text);
        },
        onStep(node) {
          const label = STEP_LABELS[node];
          if (label) deps.setStatus(`${label}…`);
        },
        onSources(s) {
          sources = s;
        },
      },
      controller.signal,
    );
    deps.setStatus("");
    const cites = new Set([...text.matchAll(/\[(\d{1,2})\]/g)].map((m) => m[1])).size;
    const files = new Set(sources.map((s) => s.file_id)).size;
    recordActivity({ ts: Date.now(), cites, multiSource: sources.length > 1 || files > 1 });
    const meta = getThread(threadId);
    upsertThread({
      id: threadId,
      title: meta?.title ?? question.slice(0, 80),
      updatedAt: Date.now(),
      snippet: text.replace(/\s*\[\d+\]|\*\*/g, "").split(/(?<=[.!?])\s/)[0]?.slice(0, 120) ?? "",
      cites,
    });
  } catch (err) {
    deps.setStatus(err instanceof ApiError ? err.message : "The answer stopped unexpectedly. Try again.", "error");
    if (!text) turn.remove();
  } finally {
    body.classList.remove("streaming");
  }
  marginFor(deps, sources, text);
  linkCitations(turn, deps.margin);
}
