// Home: a bento of real context instead of empty space. Ask hero, library and weekly
// stats, recent answers, and the latest evidence in the side column.

import { api } from "../api";
import { h, plural } from "../dom";
import { getActivity, getSelected, getThreads, type ThreadMeta } from "../store";
import { extOf, knownFiles } from "./library";
import { coverageOf, renderMargin, sourceNotes } from "./margin";

export interface HomeDeps {
  page: HTMLElement;
  margin: HTMLElement;
  ask(question: string): void;
  openThread(id: string): void;
  openReader(fileId: string, title: string): void;
}

const DAY = 86_400_000;

function greeting(): string {
  const hour = new Date().getHours();
  if (hour < 5) return "Working late";
  if (hour < 12) return "Good morning";
  if (hour < 18) return "Good afternoon";
  return "Good evening";
}

function relTime(ts: number): string {
  const d = new Date(ts);
  const today = new Date();
  if (d.toDateString() === today.toDateString()) {
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  if (Date.now() - ts < 2 * DAY && today.getDate() - d.getDate() === 1) return "Yesterday";
  if (Date.now() - ts < 7 * DAY) return d.toLocaleDateString([], { weekday: "short" });
  return d.toLocaleDateString([], { day: "numeric", month: "short" });
}

function suggestions(): string[] {
  const inScope = knownFiles().filter((f) => getSelected().includes(f.file_id));
  const first = inScope[0]?.filename?.replace(/\.[^.]+$/, "");
  if (!first) return [];
  const out = [`Summarise ${first} in five points`, `What dates and deadlines does ${first} set?`];
  const second = inScope[1]?.filename?.replace(/\.[^.]+$/, "");
  out.push(second ? `Where do ${first} and ${second} disagree?` : `Who is responsible for what in ${first}?`);
  return out;
}

function heroCard(deps: HomeDeps): HTMLElement {
  const n = getSelected().length;
  const input = h("input", {
    id: "home-q",
    placeholder: n ? `Ask about ${plural(n, "selected file")}` : "Tick files in the library, then ask",
    autocomplete: "off",
  });
  const form = h(
    "form",
    { class: "ask" },
    h("label", { class: "sr-only", for: "home-q" }, "Your question"),
    input,
    h("button", { class: "btn-primary btn-lg btn-gold", type: "submit" }, "Ask"),
  );
  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const q = input.value.trim();
    if (q) deps.ask(q);
  });
  const chips = h("div", { class: "suggest", "aria-label": "Suggested questions" });
  for (const s of suggestions()) {
    const b = h("button", { type: "button" }, s);
    b.addEventListener("click", () => deps.ask(s));
    chips.append(b);
  }
  return h(
    "article",
    { class: "tile-card hero-card" },
    h("p", { class: "greeting" }, greeting()),
    h("h2", { class: "hero-q" }, "What do you want to verify today?"),
    form,
    chips.childElementCount ? chips : null,
  );
}

function libraryCard(): HTMLElement {
  const files = knownFiles();
  const passages = files.reduce((sum, f) => sum + f.chunks, 0);
  const groups = new Map<string, number>();
  for (const f of files) {
    const ext = extOf(f.filename);
    const key = ["PDF", "DOC", "XLS", "CSV"].includes(ext) ? ext : "Other";
    groups.set(key, (groups.get(key) ?? 0) + 1);
  }
  const top = [...groups.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4);
  const bar = h("div", { class: "type-bar", role: "img", "aria-label": top.map(([k, v]) => `${k} ${v}`).join(", ") });
  const legend = h("ul", { class: "legend" });
  top.forEach(([key, count], i) => {
    const seg = h("span", { class: `seg-${i}` });
    seg.style.flex = String(count);
    bar.append(seg);
    legend.append(h("li", {}, h("i", { class: `seg-${i}` }), `${key} ${Math.round((count / files.length) * 100)}%`));
  });
  return h(
    "article",
    { class: "tile-card stat-card" },
    h("p", { class: "card-label" }, "Library"),
    h("p", { class: "big-num" }, String(files.length), h("span", {}, files.length === 1 ? " file" : " files")),
    h("p", { class: "sub-num" }, `${plural(passages, "passage").replace(String(passages), passages.toLocaleString())} searchable`),
    files.length ? bar : h("div", { class: "type-bar empty" }),
    files.length ? legend : h("p", { class: "hint" }, "Add files to see the mix."),
  );
}

function weekCard(): HTMLElement {
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  const days = [...Array(7)].map((_, i) => start.getTime() - (6 - i) * DAY);
  const recent = getActivity().filter((a) => a.ts >= days[0]!);
  const perDay = days.map((d) => recent.filter((a) => a.ts >= d && a.ts < d + DAY).length);
  const max = Math.max(1, ...perDay);
  const cites = recent.reduce((s, a) => s + a.cites, 0);
  const multi = recent.length ? Math.round((recent.filter((a) => a.multiSource).length / recent.length) * 100) : 0;
  const spark = h("div", { class: "spark", role: "img", "aria-label": `Answers per day: ${perDay.join(", ")}` });
  const labels = h("ul", { class: "legend days" });
  perDay.forEach((count, i) => {
    const bar = h("span", { class: i === 6 ? "today" : "" });
    bar.style.setProperty("--h", `${Math.max(6, (count / max) * 100)}%`);
    spark.append(bar);
    labels.append(h("li", {}, new Date(days[i]!).toLocaleDateString([], { weekday: "narrow" })));
  });
  return h(
    "article",
    { class: "tile-card stat-card" },
    h("p", { class: "card-label" }, "This week", h("span", { class: "hint" }, " in this browser")),
    h("p", { class: "big-num" }, String(recent.length), h("span", {}, recent.length === 1 ? " answer" : " answers")),
    h("p", { class: "sub-num" }, recent.length ? `${cites} citations, ${multi}% from 2+ passages` : "Ask a question to start the week."),
    spark,
    labels,
  );
}

function recentCard(deps: HomeDeps, threads: ThreadMeta[]): HTMLElement {
  const list = h("ol", { class: "feed" });
  for (const t of threads.slice(0, 4)) {
    const row = h(
      "li",
      {},
      h(
        "button",
        { class: "row", type: "button" },
        h(
          "span",
          { class: "row-body" },
          h("span", { class: "row-title" }, t.title),
          h(
            "span",
            { class: "row-meta" },
            t.snippet ?? "Open to see the answer",
            t.cites ? h("span", { class: "cite-chip" }, String(t.cites)) : null,
          ),
        ),
        h("time", {}, relTime(t.updatedAt)),
      ),
    );
    row.querySelector("button")?.addEventListener("click", () => deps.openThread(t.id));
    list.append(row);
  }
  return h(
    "article",
    { class: "tile-card list-card" },
    h("div", { class: "card-head" }, h("p", { class: "card-label" }, "Recent answers"), h("span", { class: "hint" }, plural(threads.length, "conversation"))),
    threads.length ? list : h("p", { class: "hint" }, "Your answers will collect here, newest first."),
  );
}

export async function showHome(deps: HomeDeps): Promise<void> {
  const threads = getThreads();
  deps.page.replaceChildren(
    h("div", { class: "bento" }, heroCard(deps), libraryCard(), weekCard(), recentCard(deps, threads)),
  );
  renderMargin(deps.margin, [], {
    heading: "Latest evidence",
    empty: "The passages behind your most recent answer will show here.",
  });
  const latest = threads[0];
  if (!latest) return;
  try {
    const { messages } = await api.thread(latest.id);
    const last = messages.at(-1);
    if (last?.role === "assistant" && last.sources?.length && deps.page.querySelector(".bento")) {
      renderMargin(deps.margin, sourceNotes(last.sources), {
        heading: "Latest evidence",
        sub: `for "${latest.title.slice(0, 24)}${latest.title.length > 24 ? "…" : ""}"`,
        empty: "",
        coverage: coverageOf(last.content),
        onOpen: deps.openReader,
      });
    }
  } catch {
    /* evidence column stays in its empty state */
  }
}
