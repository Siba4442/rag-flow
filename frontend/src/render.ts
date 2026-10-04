// Answer rendering: a small, escape-first Markdown subset plus [n] citation links.

import { escape } from "./dom";

function inline(text: string): string {
  return escape(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
    .replace(
      /\[(\d{1,2})\]/g,
      '<a class="cite" href="#source-$1" data-cite="$1" aria-label="Source $1">$1</a>',
    );
}

export function renderAnswer(markdown: string): string {
  const blocks = markdown.trim().split(/\n{2,}/);
  return blocks
    .map((block) => {
      const lines = block.split("\n");
      if (lines.every((l) => /^\s*[-*]\s+/.test(l))) {
        return `<ul>${lines.map((l) => `<li>${inline(l.replace(/^\s*[-*]\s+/, ""))}</li>`).join("")}</ul>`;
      }
      if (lines.every((l) => /^\s*\d+[.)]\s+/.test(l))) {
        return `<ol>${lines.map((l) => `<li>${inline(l.replace(/^\s*\d+[.)]\s+/, ""))}</li>`).join("")}</ol>`;
      }
      const heading = /^(#{1,3})\s+(.*)$/.exec(block);
      if (heading && lines.length === 1) return `<h3>${inline(heading[2] ?? "")}</h3>`;
      return `<p>${lines.map(inline).join("<br>")}</p>`;
    })
    .join("");
}
