import {
  AGENT_ID,
  MAX_MESSAGE_CHARS,
  apiHeaders,
  apiUrl,
  problem,
  validThreadId,
  validUserId,
} from "@/lib/backend";

export const dynamic = "force-dynamic";
// A reply streams for a few seconds, but the free API host can take most of a
// minute to wake up. 60 seconds is the most every Vercel plan allows.
export const maxDuration = 60;

/**
 * One chat turn. Forwards the message to the API's /stream endpoint and passes
 * its Server-Sent Events straight through, so tokens reach the browser as the
 * model writes them.
 */
export async function POST(req: Request) {
  let body: { message?: unknown; threadId?: unknown; userId?: unknown };
  try {
    body = await req.json();
  } catch {
    return problem(400, "The request body is not valid JSON.");
  }

  const message = typeof body.message === "string" ? body.message.trim() : "";
  if (!message) return problem(400, "The message is empty.");
  if (message.length > MAX_MESSAGE_CHARS) {
    return problem(413, `The message is longer than ${MAX_MESSAGE_CHARS} characters.`);
  }
  if (!validUserId(body.userId)) return problem(400, "userId must be a positive whole number.");
  if (body.threadId != null && !validThreadId(body.threadId)) {
    return problem(400, "threadId is not valid.");
  }

  let upstream: Response;
  try {
    upstream = await fetch(apiUrl(`/${AGENT_ID}/stream`), {
      method: "POST",
      headers: apiHeaders(),
      body: JSON.stringify({
        message,
        user_id: body.userId,
        stream_tokens: true,
        ...(body.threadId ? { thread_id: body.threadId } : {}),
      }),
      signal: req.signal, // stop the upstream call if the visitor leaves
      cache: "no-store",
    });
  } catch {
    return problem(502, "The JobBuddy API could not be reached.");
  }
  if (!upstream.ok || !upstream.body) {
    return problem(502, `The JobBuddy API answered with HTTP ${upstream.status}.`);
  }

  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
    },
  });
}
