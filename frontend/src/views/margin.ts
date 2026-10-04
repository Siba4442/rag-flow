// The evidence column: numbered passages beside the answer. Hovering or focusing a
// [n] citation lights its passage with the highlighter (and vice versa).

import type { Source } from "../api";
import { h, plural } from "../dom";

export interface MarginNote {
  n: number;
  title: string;
  where: string;
  text: string;
  fileId: string | null;
}

export interface Coverage {
  cited: number;
  total: number;
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

/** Sentences that carry at least one [n] citation, out of all sentences. */
export function coverageOf(answer: string): Coverage | null {
  const sentences = answer
    .replace(/\n+/g, " ")
    .split(/(?<=[.!?])\s+(?=[A-Z0-9*])/)
    .map((x) => x.trim())
    .filter((x) => x.replace(/\[\d+\]/g, "").trim().length > 12);
  if (!sentences.length) return null;
  return { cited: sentences.filter((x) => /\[\d+\]/.test(x)).length, total: sentences.length };
}

export function renderMargin(
  container: HTMLElement,
  notes: MarginNote[],
  opts: {
    heading: string;
    sub?: string;
    empty: string;
    coverage?: Coverage | null;
    onOpen?: (fileId: string, title: string) => void;
  },
): void {
  container.replaceChildren(
    h(
      "div",
      { class: "ev-head" },
      h("h2", {}, opts.heading),
      h("span", { class: "hint" }, opts.sub ?? (notes.length ? plural(notes.length, "passage") : "")),
    ),
  );
  if (!notes.length) {
    container.append(h("p", { class: "ev-empty" }, opts.empty));
    return;
  }
  const list = h("ol", { class: "ev-list" });
  for (const note of notes) {
    const open =
      note.fileId && opts.onOpen ? h("button", { class: "text-btn", type: "button" }, "Read whole file") : null;
    if (open && note.fileId) {
      const id = note.fileId;
      open.addEventListener("click", () => opts.onOpen?.(id, note.title));
    }
    list.append(
      h(
        "li",
        { class: "ev", id: `source-${note.n}`, "data-note": String(note.n), tabindex: "-1" },
        h("span", { class: "ev-n", "aria-hidden": "true" }, String(note.n).padStart(2, "0")),
        h("p", { class: "ev-src" }, note.title, h("span", {}, note.where)),
        h("p", { class: "ev-text" }, h("mark", {}, note.text.trim())),
        open,
      ),
    );
  }
  container.append(list);

  if (opts.coverage) {
    const { cited, total } = opts.coverage;
    const bar = h("div", { class: "cov-bar" }, h("span", {}));
    (bar.firstElementChild as HTMLElement).style.width = `${Math.round((cited / total) * 100)}%`;
    container.append(
      h(
        "div",
        { class: "coverage" },
        h("div", { class: "cov-head" }, h("span", {}, "Claims cited"), h("b", {}, `${cited} of ${total}`)),
        bar,
        h(
          "p",
          { class: "hint" },
          cited === total ? "Every sentence points to a passage." : "Uncited sentences are the model's own wording.",
        ),
      ),
    );
  }
}

/** Link citations inside `scope` with passages inside `margin` (hover, focus, click). */
export function linkCitations(scope: HTMLElement, margin: HTMLElement): void {
  const find = (n: string | undefined): HTMLElement | null =>
    n ? margin.querySelector<HTMLElement>(`[data-note="${CSS.escape(n)}"]`) : null;
  const mark = (n: string | undefined, on: boolean): void => {
    if (!n) return;
    find(n)?.classList.toggle("lit", on);
    scope.querySelectorAll(`[data-cite="${CSS.escape(n)}"]`).forEach((a) => a.classList.toggle("lit", on));
  };
  scope.querySelectorAll<HTMLAnchorElement>("a.cite").forEach((a) => {
    const n = a.dataset.cite;
    if (!find(n)) {
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
      const note = find(n);
      note?.scrollIntoView({
        block: "nearest",
        behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
      });
      note?.focus({ preventScroll: true });
    });
  });
  margin.querySelectorAll<HTMLElement>(".ev").forEach((note) => {
    const n = note.dataset.note;
    note.addEventListener("mouseenter", () => mark(n, true));
    note.addEventListener("mouseleave", () => mark(n, false));
  });
}
