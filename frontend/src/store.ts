// Small app state with per-browser persistence. Storage can be unavailable (private
// mode, blocked site data), so every access is guarded and the app works without it.

export interface Settings {
  baseUrl: string;
  token: string;
  entityId: string;
}

export interface ThreadMeta {
  id: string;
  title: string;
  updatedAt: number;
}

const KEYS = { settings: "ragflow.settings", threads: "ragflow.threads", selected: "ragflow.selected" };

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function write(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable: keep in memory only */
  }
}

let settings: Settings = { baseUrl: "", token: "", entityId: "", ...read<Partial<Settings>>(KEYS.settings, {}) };
let threads: ThreadMeta[] = read<ThreadMeta[]>(KEYS.threads, []);
let selected = new Set<string>(read<string[]>(KEYS.selected, []));

type Listener = () => void;
const listeners = new Set<Listener>();
export function subscribe(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
function emit(): void {
  for (const fn of listeners) fn();
}

export const getSettings = (): Settings => settings;
export function saveSettings(next: Settings): void {
  settings = next;
  write(KEYS.settings, next);
  emit();
}

export const getThreads = (): ThreadMeta[] => [...threads].sort((a, b) => b.updatedAt - a.updatedAt);
export function upsertThread(meta: ThreadMeta): void {
  threads = [meta, ...threads.filter((t) => t.id !== meta.id)].slice(0, 50);
  write(KEYS.threads, threads);
  emit();
}
export function removeThread(id: string): void {
  threads = threads.filter((t) => t.id !== id);
  write(KEYS.threads, threads);
  emit();
}

export const getSelected = (): string[] => [...selected];
export const isSelected = (id: string): boolean => selected.has(id);
export function setSelected(ids: Iterable<string>): void {
  selected = new Set(ids);
  write(KEYS.selected, [...selected]);
  emit();
}
export function toggleSelected(id: string, on: boolean): void {
  const next = new Set(selected);
  if (on) next.add(id);
  else next.delete(id);
  setSelected(next);
}

export function newId(): string {
  return crypto.randomUUID().replace(/-/g, "").slice(0, 20);
}
