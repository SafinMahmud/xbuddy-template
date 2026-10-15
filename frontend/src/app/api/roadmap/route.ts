import { AGENT_ID, apiHeaders, apiUrl, problem, validThreadId } from "@/lib/backend";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

/** The finished roadmap of a conversation, or ready: false while it is not written yet. */
export async function GET(req: Request) {
  const params = new URL(req.url).searchParams;
  const threadId = params.get("threadId");
  const userId = Number(params.get("userId") ?? "1");
  if (!validThreadId(threadId)) return problem(400, "threadId is not valid.");

  const query = new URLSearchParams({
    thread_id: threadId,
    user_id: String(Number.isInteger(userId) && userId > 0 ? userId : 1),
  });
  let upstream: Response;
  try {
    upstream = await fetch(apiUrl(`/roadmap/${AGENT_ID}?${query}`), {
      headers: apiHeaders(),
      cache: "no-store",
    });
  } catch {
    return problem(502, "The JobBuddy API could not be reached.");
  }
  if (!upstream.ok) return problem(502, `The JobBuddy API answered with HTTP ${upstream.status}.`);

  const data = (await upstream.json()) as { roadmap?: string | null; section?: unknown };
  return Response.json({
    ready: Boolean(data.roadmap),
    roadmap: data.roadmap ?? null,
    section: data.section ?? null,
  });
}
