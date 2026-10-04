import "./styles/main.css";

import { api, ApiError } from "./api";
import { $, h } from "./dom";
import { getSettings, getThreads, removeThread, saveSettings, subscribe } from "./store";
import { ask, currentThread, newThread, showThread, type ChatDeps } from "./views/chat";
import { showHome } from "./views/home";
import { mountLibrary, refreshLibrary, updateScopeHint } from "./views/library";
import { search, searchIntro } from "./views/search";

type Mode = "home" | "chat" | "search";
let mode: Mode = "home";

const page = $("#page");
const margin = $("#margin");
const statusEl = $("#status");
const prompt = $("#prompt") as HTMLTextAreaElement;
const send = $("#send") as HTMLButtonElement;
const composer = $("#composer") as HTMLFormElement;

function setStatus(text: string, tone?: "error" | "ok"): void {
  statusEl.textContent = text;
  statusEl.dataset.tone = tone ?? "";
}

async function openReader(fileId: string, title: string): Promise<void> {
  const dialog = $("#reader") as HTMLDialogElement;
  $("#reader-title").textContent = title;
  const body = $("#reader-body");
  body.textContent = "Loading…";
  dialog.showModal();
  try {
    body.textContent = await api.context(fileId);
  } catch (err) {
    body.textContent = err instanceof ApiError ? err.message : "Couldn't open this file.";
  }
}

const deps: ChatDeps = { page, margin, setStatus, openReader: (id, t) => void openReader(id, t) };

// ------------------------------------------------------------------ views
const TITLES: Record<Mode, string> = { home: "Home", chat: "Ask", search: "Search passages" };

function renderHome(): void {
  void showHome({
    page,
    margin,
    openReader: deps.openReader,
    ask: (q) => {
      setMode("chat", { keepThread: false });
      void runAsk(q);
    },
    openThread: (id) => {
      setMode("chat", { thread: id });
    },
  });
}

function setMode(next: Mode, opts: { thread?: string; keepThread?: boolean } = {}): void {
  mode = next;
  for (const tab of document.querySelectorAll<HTMLButtonElement>(".mode")) {
    tab.setAttribute("aria-selected", String(tab.dataset.mode === next));
  }
  $("#view-title").textContent = TITLES[next];
  composer.hidden = next === "home";
  document.body.dataset.mode = next;
  prompt.placeholder = next === "search" ? "Describe the passage you're looking for" : "Ask a follow-up or a new question";
  send.textContent = next === "search" ? "Search" : "Ask";
  setStatus("");
  if (next === "home") renderHome();
  else if (next === "search") searchIntro(deps);
  else if (opts.keepThread === false) newThread(deps);
  else void showThread(opts.thread ?? currentThread(), deps).then(renderThreads);
  closeRail();
}

for (const tab of document.querySelectorAll<HTMLButtonElement>(".mode")) {
  tab.addEventListener("click", () => {
    const m = tab.dataset.mode;
    setMode(m === "search" || m === "chat" ? m : "home");
  });
}

async function runAsk(text: string): Promise<void> {
  send.disabled = true;
  try {
    await ask(text, deps);
  } finally {
    send.disabled = false;
    renderThreads();
  }
}

// ------------------------------------------------------------------ threads
function renderThreads(): void {
  const list = $("#thread-list");
  list.replaceChildren();
  const threads = getThreads();
  if (!threads.length) {
    list.append(h("li", { class: "empty" }, "Your conversations will be listed here."));
    return;
  }
  for (const t of threads.slice(0, 8)) {
    const open = h(
      "button",
      {
        class: "thread",
        type: "button",
        title: t.title,
        "aria-current": mode === "chat" && t.id === currentThread() ? "true" : undefined,
      },
      t.title,
    );
    open.addEventListener("click", () => setMode("chat", { thread: t.id }));
    const del = h("button", { class: "icon-btn", type: "button", "aria-label": `Delete conversation: ${t.title}` }, "✕");
    del.addEventListener("click", async () => {
      try {
        await api.deleteThread(t.id);
      } catch {
        /* still remove locally */
      }
      removeThread(t.id);
      if (t.id === currentThread() && mode === "chat") newThread(deps);
    });
    list.append(h("li", {}, open, del));
  }
}

// ------------------------------------------------------------------ composer
function autosize(): void {
  prompt.style.height = "auto";
  prompt.style.height = `${Math.min(prompt.scrollHeight, 200)}px`;
}
prompt.addEventListener("input", autosize);
prompt.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    composer.dispatchEvent(new Event("submit", { cancelable: true }));
  }
});

composer.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = prompt.value.trim();
  if (!text || send.disabled) return;
  prompt.value = "";
  autosize();
  if (mode === "search") {
    send.disabled = true;
    try {
      await search(text, deps);
    } finally {
      send.disabled = false;
    }
  } else {
    await runAsk(text);
  }
  prompt.focus();
});

$("#new-thread").addEventListener("click", () => {
  setMode("chat", { keepThread: false });
  renderThreads();
  prompt.focus();
});

// ------------------------------------------------------------------ rail (mobile drawer)
function closeRail(): void {
  document.body.classList.remove("rail-open");
}
$("#open-rail").addEventListener("click", () => document.body.classList.add("rail-open"));
$("#close-rail").addEventListener("click", closeRail);

// ------------------------------------------------------------------ settings
const settingsDialog = $("#settings") as HTMLDialogElement;
const settingsForm = $("#settings-form") as HTMLFormElement;
$("#open-settings").addEventListener("click", () => {
  const s = getSettings();
  (settingsForm.elements.namedItem("baseUrl") as HTMLInputElement).value = s.baseUrl;
  (settingsForm.elements.namedItem("token") as HTMLInputElement).value = s.token;
  (settingsForm.elements.namedItem("entityId") as HTMLInputElement).value = s.entityId;
  settingsDialog.showModal();
});
settingsDialog.addEventListener("close", () => {
  if (settingsDialog.returnValue !== "save") return;
  const get = (n: string): string => (settingsForm.elements.namedItem(n) as HTMLInputElement).value.trim();
  saveSettings({ baseUrl: get("baseUrl"), token: get("token"), entityId: get("entityId") });
  setStatus("Connection settings saved.", "ok");
  void boot();
});

// ------------------------------------------------------------------ theme
type ThemeChoice = "light" | "dark" | "system";
const THEME_KEY = "ragflow.theme";

function applyTheme(choice: ThemeChoice): void {
  if (choice === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = choice;
  for (const b of document.querySelectorAll<HTMLButtonElement>("[data-theme-choice]")) {
    b.setAttribute("aria-checked", String(b.dataset.themeChoice === choice));
  }
  try {
    if (choice === "system") localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, choice);
  } catch {
    /* theme still applies for this visit */
  }
}
for (const b of document.querySelectorAll<HTMLButtonElement>("[data-theme-choice]")) {
  b.addEventListener("click", () => {
    const c = b.dataset.themeChoice;
    applyTheme(c === "light" || c === "dark" ? c : "system");
  });
}
{
  const saved = document.documentElement.dataset.theme;
  applyTheme(saved === "light" || saved === "dark" ? saved : "system");
}

// ------------------------------------------------------------------ system card
async function renderSystem(): Promise<void> {
  const dl = $("#system-info");
  const health = $("#health");
  const set = (ok: boolean, text: string): void => {
    health.dataset.ok = String(ok);
    health.lastElementChild!.textContent = text;
  };
  try {
    const info = await api.info();
    const row = (k: string, v: string): HTMLElement =>
      h("div", {}, h("dt", {}, k), h("dd", { title: v }, v));
    dl.replaceChildren(
      row("Chat", info.chat_model ?? "Not configured"),
      row("Embeddings", info.embeddings_model.split("/").pop() ?? info.embeddings_model),
      row("Store", info.vector_store),
    );
  } catch {
    dl.replaceChildren();
  }
  try {
    await api.health();
    set(true, "All systems up");
  } catch {
    set(false, "Server unreachable");
  }
}

// ------------------------------------------------------------------ boot
subscribe(() => {
  renderThreads();
  updateScopeHint();
});

async function boot(): Promise<void> {
  await Promise.all([
    renderSystem(),
    refreshLibrary({ ...deps, onChange: () => mode === "home" && renderHome() }),
  ]);
  renderThreads();
  setMode(mode);
}

mountLibrary({ ...deps, onChange: () => mode === "home" && renderHome() });
void boot();
