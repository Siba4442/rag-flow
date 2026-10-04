// "Ask" mode: streamed answers with [n] citations linked to margin notes.

import { api, ApiError, streamChat, type Source, type ThreadMessage } from "../api";
import { h } from "../dom";
import { renderAnswer } from "../render";
import { getSelected, getThreads, newId, upsertThread } from "../store";
import { linkCitations, renderMargin, sourceNotes } from "./margin";

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
          ? "Questions are answered from the files ticked in your library. Every claim points to the passage it came from, shown in the margin."
          : "Tick one or more files in your library, then ask a question. Answers cite the exact passages they rely on.",
      ),
    ),
  );
}

function marginFor(deps: ChatDeps, sources: Source[]): void {
  renderMargin(deps.margin, sourceNotes(sources), {
    heading: "Sources",
    empty: "Passages used in the latest answer appear here.",
    onOpen: deps.openReader,
  });
}

function messageEl(msg: ThreadMessage): HTMLElement {
  if (msg.role === "user") return h("div", { class: "turn user" }, h("p", {}, msg.content));
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
    marginFor(deps, last?.sources ?? []);
    linkCitations(deps.page, deps.margin);
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

  upsertThread({ id: threadId, title: question.slice(0, 80), updatedAt: Date.now() });
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
  } catch (err) {
    deps.setStatus(err instanceof ApiError ? err.message : "The answer stopped unexpectedly. Try again.", "error");
    if (!text) turn.remove();
  } finally {
    body.classList.remove("streaming");
  }
  marginFor(deps, sources);
  linkCitations(turn, deps.margin);
}
