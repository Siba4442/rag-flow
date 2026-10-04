// The margin: numbered source notes beside the answer. Hovering or focusing a [n]
// citation highlights its note (and vice versa).

import type { Source } from "../api";
import { h, plural } from "../dom";

export interface MarginNote {
  n: number;
  title: string;
  where: string;
  text: string;
  fileId: string | null;
}

export function sourceNotes(sources: Source[]): MarginNote[] {
  return sources.map((s) => ({
    n: s.n,
    title: s.filename ?? s.file_id ?? "Untitled file",
    where: typeof s.page === "number" ? `page ${s.page + 1}` : `passage ${(s.chunk_index ?? 0) + 1}`,
    text: s.snippet,
    fileId: s.file_id,
  }));
}

export function renderMargin(
  container: HTMLElement,
  notes: MarginNote[],
  opts: { heading: string; empty: string; onOpen?: (fileId: string, title: string) => void },
): void {
  container.replaceChildren();
  container.append(
    h("h2", { class: "margin-head" }, notes.length ? `${opts.heading} (${plural(notes.length, "passage")})` : opts.heading),
  );
  if (!notes.length) {
    container.append(h("p", { class: "margin-empty" }, opts.empty));
    return;
  }
  const list = h("ol", { class: "notes" });
  for (const note of notes) {
    const open =
      note.fileId && opts.onOpen
        ? h("button", { class: "text-btn", type: "button" }, "Read whole file")
        : null;
    if (open && note.fileId) {
      const id = note.fileId;
      open.addEventListener("click", () => opts.onOpen?.(id, note.title));
    }
    list.append(
      h(
        "li",
        { class: "note", id: `source-${note.n}`, "data-note": String(note.n), tabindex: "-1" },
        h("span", { class: "note-n", "aria-hidden": "true" }, String(note.n)),
        h(
          "div",
          { class: "note-body" },
          h("p", { class: "note-title" }, note.title, h("span", { class: "note-where" }, `, ${note.where}`)),
          h("blockquote", {}, h("mark", {}, note.text.trim())),
          open,
        ),
      ),
    );
  }
  container.append(list);
}

/** Link citations inside `scope` with notes inside `margin` (hover, focus, click). */
export function linkCitations(scope: HTMLElement, margin: HTMLElement): void {
  const mark = (n: string | undefined, on: boolean): void => {
    if (!n) return;
    margin.querySelector(`[data-note="${CSS.escape(n)}"]`)?.classList.toggle("lit", on);
    scope.querySelectorAll(`[data-cite="${CSS.escape(n)}"]`).forEach((a) => a.classList.toggle("lit", on));
  };
  scope.querySelectorAll<HTMLAnchorElement>("a.cite").forEach((a) => {
    const n = a.dataset.cite;
    if (!margin.querySelector(`[data-note="${CSS.escape(n ?? "")}"]`)) {
      // The model cited a number with no matching passage: show it, but inert.
      const span = document.createElement("span");
      span.className = "cite missing";
      span.title = "No matching source for this citation";
      span.textContent = a.textContent;
      a.replaceWith(span);
      return;
    }
    a.addEventListener("mouseenter", () => mark(n, true));
    a.addEventListener("mouseleave", () => mark(n, false));
    a.addEventListener("focus", () => mark(n, true));
    a.addEventListener("blur", () => mark(n, false));
    a.addEventListener("click", (ev) => {
      ev.preventDefault();
      const note = margin.querySelector<HTMLElement>(`[data-note="${CSS.escape(n ?? "")}"]`);
      note?.scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
      note?.focus({ preventScroll: true });
    });
  });
  margin.querySelectorAll<HTMLElement>(".note").forEach((note) => {
    const n = note.dataset.note;
    note.addEventListener("mouseenter", () => mark(n, true));
    note.addEventListener("mouseleave", () => mark(n, false));
  });
}
