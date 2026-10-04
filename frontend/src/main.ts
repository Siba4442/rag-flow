import "./styles/main.css";

import { api, ApiError } from "./api";
import { $, h } from "./dom";
import { getSettings, getThreads, removeThread, saveSettings, subscribe } from "./store";
import { ask, currentThread, newThread, showThread, type ChatDeps } from "./views/chat";
import { mountLibrary, refreshLibrary, updateScopeHint } from "./views/library";
import { search, searchIntro } from "./views/search";

type Mode = "chat" | "search";
let mode: Mode = "chat";

const page = $("#page");
const margin = $("#margin");
const statusEl = $("#status");
const prompt = $("#prompt") as HTMLTextAreaElement;
const send = $("#send") as HTMLButtonElement;

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

// ------------------------------------------------------------------ threads
function renderThreads(): void {
  const list = $("#thread-list");
  list.replaceChildren();
  const threads = getThreads();
  if (!threads.length) {
    list.append(h("li", { class: "empty" }, "Your conversations will be listed here."));
    return;
  }
  for (const t of threads) {
    const open = h(
      "button",
      { class: "thread", type: "button", "aria-current": t.id === currentThread() ? "true" : undefined },
      t.title,
    );
    open.addEventListener("click", () => {
      setMode("chat");
      void showThread(t.id, deps).then(renderThreads);
      closeRail();
    });
    const del = h("button", { class: "icon-btn", type: "button", "aria-label": `Delete conversation: ${t.title}` }, "✕");
    del.addEventListener("click", async () => {
      try {
        await api.deleteThread(t.id);
      } catch {
        /* still remove locally */
      }
      removeThread(t.id);
      if (t.id === currentThread()) newThread(deps);
    });
    list.append(h("li", {}, open, del));
  }
}

// ------------------------------------------------------------------ modes
function setMode(next: Mode): void {
  mode = next;
  for (const tab of document.querySelectorAll<HTMLButtonElement>(".mode")) {
    tab.setAttribute("aria-selected", String(tab.dataset.mode === next));
  }
  prompt.placeholder = next === "chat" ? "Ask about the selected files" : "Describe the passage you're looking for";
  send.textContent = next === "chat" ? "Ask" : "Search";
  setStatus("");
  if (next === "search") searchIntro(deps);
  else void showThread(currentThread(), deps);
}

for (const tab of document.querySelectorAll<HTMLButtonElement>(".mode")) {
  tab.addEventListener("click", () => setMode(tab.dataset.mode === "search" ? "search" : "chat"));
}

// ------------------------------------------------------------------ composer
function autosize(): void {
  prompt.style.height = "auto";
  prompt.style.height = `${Math.min(prompt.scrollHeight, 220)}px`;
}
prompt.addEventListener("input", autosize);
prompt.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    $("#composer").dispatchEvent(new Event("submit", { cancelable: true }));
  }
});

$("#composer").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = prompt.value.trim();
  if (!text || send.disabled) return;
  prompt.value = "";
  autosize();
  send.disabled = true;
  try {
    if (mode === "chat") await ask(text, deps);
    else await search(text, deps);
  } finally {
    send.disabled = false;
    prompt.focus();
  }
});

$("#new-thread").addEventListener("click", () => {
  setMode("chat");
  newThread(deps);
  renderThreads();
  closeRail();
  prompt.focus();
});

// ------------------------------------------------------------------ rail (mobile)
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

// ------------------------------------------------------------------ boot
subscribe(() => {
  renderThreads();
  updateScopeHint();
});

async function boot(): Promise<void> {
  try {
    await api.health();
  } catch {
    setStatus("Can't reach the RAG Flow server. Check the API address in connection settings.", "error");
  }
  await refreshLibrary(deps);
  renderThreads();
  setMode(mode);
}

mountLibrary(deps);
void boot();
