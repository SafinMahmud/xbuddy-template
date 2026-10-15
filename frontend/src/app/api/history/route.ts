import { apiHeaders, apiUrl, problem, validThreadId } from "@/lib/backend";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

interface ApiMessage {
  type: string;
  content: string;
}

/** Messages and section progress of a saved conversation, for a returning visitor. */
export async function POST(req: Request) {
  const body = await req.json().catch(() => null);
  if (!validThreadId(body?.threadId)) return problem(400, "threadId is not valid.");

  let upstream: Response;
  try {
    upstream = await fetch(apiUrl("/history"), {
      method: "POST",
      headers: apiHeaders(),
      body: JSON.stringify({ thread_id: body.threadId }),
      cache: "no-store",
    });
  } catch {
    return problem(502, "The JobBuddy API could not be reached.");
  }
  if (!upstream.ok) return problem(502, `The JobBuddy API answered with HTTP ${upstream.status}.`);

  const data = (await upstream.json()) as { messages?: ApiMessage[]; section?: unknown };
  const messages = (data.messages ?? [])
    .filter((m) => (m.type === "human" || m.type === "ai") && m.content)
    .map((m, index) => ({
      id: `h${index}`,
      role: m.type === "human" ? "user" : "assistant",
      content: m.content,
    }));
  return Response.json({ messages, section: data.section ?? null });
}
