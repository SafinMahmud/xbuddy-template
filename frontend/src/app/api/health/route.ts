import { apiUrl } from "@/lib/backend";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

/**
 * Is the API awake? The page calls this as soon as it opens: on a free host the
 * request itself wakes a sleeping API, so it is warming up while the visitor
 * reads the first screen.
 */
export async function GET() {
  try {
    const upstream = await fetch(apiUrl("/health"), {
      cache: "no-store",
      signal: AbortSignal.timeout(55_000),
    });
    return Response.json({ ok: upstream.ok }, { status: upstream.ok ? 200 : 503 });
  } catch {
    return Response.json({ ok: false }, { status: 503 });
  }
}
