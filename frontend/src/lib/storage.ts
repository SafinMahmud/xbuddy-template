/**
 * What the browser remembers: which conversations this visitor started.
 * The messages themselves live on the server (the API's checkpointer), so a
 * conversation picked from this list is always loaded fresh from /api/history.
 */
import type { SavedThread } from "./types";

const THREADS_KEY = "jobbuddy.threads";
const ACTIVE_KEY = "jobbuddy.active";
const USER_KEY = "jobbuddy.user";
const MAX_THREADS = 20;

function read<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback; // private mode, blocked storage, or a damaged value
  }
}

function write(key: string, value: unknown): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage is a convenience. The app works without it.
  }
}

/** A random id per browser. The demo has no accounts, so this is not a login. */
export function getUserId(): number {
  let id = read<number>(USER_KEY, 0);
  if (!Number.isInteger(id) || id <= 0) {
    id = 1000 + Math.floor(Math.random() * 2_000_000_000);
    write(USER_KEY, id);
  }
  return id;
}

export function listThreads(): SavedThread[] {
  const threads = read<SavedThread[]>(THREADS_KEY, []);
  return Array.isArray(threads) ? threads : [];
}

export function saveThread(threadId: string, title?: string): SavedThread[] {
  const threads = listThreads();
  const existing = threads.find((t) => t.threadId === threadId);
  const entry: SavedThread = {
    threadId,
    title: existing?.title || title || "New conversation",
    updatedAt: new Date().toISOString(),
  };
  const next = [entry, ...threads.filter((t) => t.threadId !== threadId)].slice(0, MAX_THREADS);
  write(THREADS_KEY, next);
  return next;
}

export function forgetThread(threadId: string): SavedThread[] {
  const next = listThreads().filter((t) => t.threadId !== threadId);
  write(THREADS_KEY, next);
  return next;
}

export function getActiveThread(): string | null {
  return read<string | null>(ACTIVE_KEY, null);
}

export function setActiveThread(threadId: string | null): void {
  write(ACTIVE_KEY, threadId);
}

export function titleFrom(message: string): string {
  const text = message.replace(/\s+/g, " ").trim();
  return text.length > 48 ? `${text.slice(0, 48)}…` : text;
}
