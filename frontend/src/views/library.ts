// Library: upload (evidence-column drop zone), list in the rail, select (= scope for
// Ask and Search), read and delete.

import { api, ApiError, type FileSummary } from "../api";
import { $, h, plural } from "../dom";
import { getSelected, isSelected, newId, setSelected, toggleSelected } from "../store";

export interface LibraryDeps {
  setStatus(text: string, tone?: "error" | "ok"): void;
  openReader(fileId: string, title: string): void;
  onChange?(): void;
}

let files: FileSummary[] = [];
const pending = new Map<string, string>(); // temp key -> filename while uploading

export const knownFiles = (): FileSummary[] => files;

export function fileTitle(fileId: string): string {
  return files.find((x) => x.file_id === fileId)?.filename ?? fileId;
}

export function extOf(name: string | null): string {
  const ext = (name ?? "").split(".").pop()?.toLowerCase() ?? "";
  const map: Record<string, string> = {
    pdf: "PDF", doc: "DOC", docx: "DOC", xls: "XLS", xlsx: "XLS", csv: "CSV",
    ppt: "PPT", pptx: "PPT", md: "MD", txt: "TXT", json: "JSON",
  };
  return map[ext] ?? (ext ? ext.slice(0, 4).toUpperCase() : "FILE");
}

export async function refreshLibrary(deps: LibraryDeps): Promise<void> {
  try {
    files = await api.files();
    files.sort((a, b) => (b.added_at ?? 0) - (a.added_at ?? 0));
    const ids = new Set(files.map((f) => f.file_id));
    const kept = getSelected().filter((id) => ids.has(id));
    if (kept.length !== getSelected().length) setSelected(kept);
  } catch (err) {
    deps.setStatus(err instanceof ApiError ? err.message : "Couldn't load your files.", "error");
  }
  renderLibrary(deps);
  deps.onChange?.();
}

function renderLibrary(deps: LibraryDeps): void {
  const list = $("#file-list");
  list.replaceChildren();
  for (const [key, name] of pending) {
    list.append(
      h(
        "li",
        { class: "file pending", "data-key": key },
        h("span", { class: "spinner", "aria-hidden": "true" }),
        h("span", { class: "file-name" }, name),
        h("span", { class: "count" }, "Reading"),
      ),
    );
  }
  if (!files.length && !pending.size) {
    list.append(h("li", { class: "empty" }, "No files yet. Use Add files to upload your first document."));
  }
  for (const f of files) {
    const id = `sel-${f.file_id}`;
    const box = h("input", { type: "checkbox", id, checked: isSelected(f.file_id) });
    box.addEventListener("change", () => toggleSelected(f.file_id, box.checked));
    const read = h("button", { class: "text-btn", type: "button" }, "Read");
    read.addEventListener("click", () => deps.openReader(f.file_id, f.filename ?? f.file_id));
    const del = h("button", { class: "text-btn danger", type: "button" }, "Delete");
    del.addEventListener("click", () => void removeFile(f, deps));
    list.append(
      h(
        "li",
        { class: `file${isSelected(f.file_id) ? " is-on" : ""}` },
        box,
        h("span", { class: "ext", "aria-hidden": "true" }, extOf(f.filename)),
        h("label", { for: id, class: "file-name", title: f.filename ?? f.file_id }, f.filename ?? f.file_id),
        h("span", { class: "count", title: plural(f.chunks, "passage") }, String(f.chunks)),
        h("span", { class: "file-actions" }, read, del),
      ),
    );
  }
  updateScopeHint();
}

export function updateScopeHint(): void {
  const n = getSelected().length;
  $("#scope-hint").textContent = files.length ? `${n} of ${files.length} in scope` : "";
  document.querySelectorAll<HTMLLIElement>("#file-list .file").forEach((li) => {
    const box = li.querySelector<HTMLInputElement>("input[type=checkbox]");
    li.classList.toggle("is-on", Boolean(box?.checked));
  });
}

async function removeFile(f: FileSummary, deps: LibraryDeps): Promise<void> {
  const name = f.filename ?? f.file_id;
  if (!confirm(`Delete "${name}" from the library? Its passages are removed for good.`)) return;
  try {
    await api.deleteFiles([f.file_id]);
    toggleSelected(f.file_id, false);
    deps.setStatus(`Deleted ${name}.`, "ok");
  } catch (err) {
    deps.setStatus(err instanceof ApiError ? err.message : `Couldn't delete ${name}.`, "error");
  }
  await refreshLibrary(deps);
}

async function uploadAll(list: FileList | File[], deps: LibraryDeps): Promise<void> {
  const items = [...list].map((file) => [newId(), file] as const);
  for (const [key, file] of items) pending.set(key, file.name);
  renderLibrary(deps);
  let added = 0;
  for (const [key, file] of items) {
    const fileId = newId();
    try {
      const res = await api.upload(file, fileId);
      if (res.status) {
        added += 1;
        toggleSelected(fileId, true); // new files join the current scope
      } else {
        deps.setStatus(`${file.name}: ${res.message}`, "error");
      }
    } catch (err) {
      deps.setStatus(err instanceof ApiError ? `${file.name}: ${err.message}` : `Couldn't add ${file.name}.`, "error");
    } finally {
      pending.delete(key);
    }
  }
  if (added) deps.setStatus(`Added ${plural(added, "file")}.`, "ok");
  await refreshLibrary(deps);
}

export function mountLibrary(deps: LibraryDeps): void {
  const zone = $("#dropzone");
  const input = $("#file-input") as HTMLInputElement;
  input.addEventListener("change", () => {
    if (input.files?.length) void uploadAll(input.files, deps);
    input.value = "";
  });
  zone.addEventListener("dragover", (ev) => {
    ev.preventDefault();
    zone.classList.add("over");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (ev) => {
    ev.preventDefault();
    zone.classList.remove("over");
    if (ev.dataTransfer?.files.length) void uploadAll(ev.dataTransfer.files, deps);
  });
}
