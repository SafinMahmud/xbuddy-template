/**
 * Server-side access to the JobBuddy API.
 *
 * The browser only ever talks to this app's /api routes. They forward to the API
 * from the server, so the API URL and token stay out of the browser and there is
 * no CORS setup to maintain.
 */

export const AGENT_ID = "xbuddy";

const THREAD_ID = /^[A-Za-z0-9_-]{1,80}$/;
export const MAX_MESSAGE_CHARS = 30_000; // room for a few pasted job postings

export function apiUrl(path: string): string {
  const base = (process.env.JOBBUDDY_API_URL || "http://localhost:8080").replace(/\/+$/, "");
  return `${base}${path}`;
}

export function apiHeaders(): Record<string, string> {
  const token = process.env.JOBBUDDY_API_TOKEN;
  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

export function validThreadId(value: unknown): value is string {
  return typeof value === "string" && THREAD_ID.test(value);
}

export function validUserId(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value > 0 && value < 2 ** 31;
}

export function problem(status: number, error: string): Response {
  return Response.json({ error }, { status });
}
