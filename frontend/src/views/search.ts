// "Search passages" mode: ranked chunks from the selected files, no LLM involved.

import { api, ApiError } from "../api";
import { h } from "../dom";
import { getSelected } from "../store";
import { fileTitle } from "./library";
import { renderMargin } from "./margin";

export interface SearchDeps {
  page: HTMLElement;
  margin: HTMLElement;
  setStatus(text: string, tone?: "error" | "ok"): void;
  openReader(fileId: string, title: string): void;
}

export function searchIntro(deps: SearchDeps): void {
  deps.page.replaceChildren(
    h(
      "div",
      { class: "intro" },
      h("h2", {}, "Search passages"),
      h("p", {}, "Find the passages closest in meaning to your words, ranked by similarity. Nothing is generated: you see exactly what's stored."),
    ),
  );
  renderMargin(deps.margin, [], { heading: "Matches", empty: "Matching passages appear here." });
}

export async function search(query: string, deps: SearchDeps): Promise<void> {
  const ids = getSelected();
  if (!ids.length) {
    deps.setStatus("Tick at least one file in your library to search it.", "error");
    return;
  }
  deps.setStatus("Searching…");
  try {
    const hits = await api.search(query, ids);
    deps.page.replaceChildren(
      h("div", { class: "turn user" }, h("p", {}, query)),
      h(
        "p",
        { class: "result-summary" },
        hits.length
          ? `The ${hits.length} closest passages, best match first.`
          : "No passages matched. Try different words or select more files.",
      ),
    );
    renderMargin(
      deps.margin,
      hits.map(([doc, score], i) => {
        const meta = doc.metadata;
        const fileId = typeof meta.file_id === "string" ? meta.file_id : null;
        const page = typeof meta.page === "number" ? `page ${meta.page + 1}` : `passage ${Number(meta.chunk_index ?? 0) + 1}`;
        return {
          n: i + 1,
          title: fileId ? fileTitle(fileId) : "Unknown file",
          where: `${page}, score ${score.toFixed(3)}`,
          text: doc.page_content,
          fileId,
        };
      }),
      { heading: "Matches", empty: "No matches.", onOpen: deps.openReader },
    );
    deps.setStatus("");
  } catch (err) {
    deps.setStatus(err instanceof ApiError ? err.message : "Search failed. Try again.", "error");
  }
}
