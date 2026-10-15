#!/usr/bin/env node
/**
 * Signed-out smoke test for a deployed JobBuddy frontend.
 *
 * "Signed out" means plain requests with no cookies and no Authorization header:
 * what a stranger with only the link can do. It goes through the frontend's own
 * /api routes, so it checks the whole path: Vercel -> API -> model -> database.
 *
 * Usage (Node 18+, no install needed):
 *   node scripts/smoke.mjs https://your-app.vercel.app          one turn
 *   node scripts/smoke.mjs https://your-app.vercel.app --full   all five sections, to the roadmap
 *
 * Checks:
 *   1. the page loads and shows the demo-only warning
 *   2. /api/health answers (waits up to 2 minutes for a sleeping API)
 *   3. one turn through /api/chat streams tokens, ends with a real reply (not the
 *      fallback) and reports the Background section
 *   4. /api/history returns that turn, /api/roadmap says "not ready"
 *   5. --full: a scripted job seeker answers every section and confirms each
 *      summary until the roadmap is ready, then the roadmap is read back.
 *      It uses the real model, so it takes a few minutes and a little free quota.
 *
 * Exit code 0 when every check passes, 1 otherwise.
 */

const base = (process.argv[2] || "").replace(/\/+$/, "");
const full = process.argv.includes("--full");
if (!/^https?:\/\//.test(base)) {
  console.error("Usage: node scripts/smoke.mjs https://your-app.vercel.app [--full]");
  process.exit(1);
}

const USER_ID = 990002;
const FALLBACK_REPLY = "Sorry, I had trouble responding just now. Could you say that again?";
const CONFIRM = "Yes, that is all correct.";
const MAX_TURNS_PER_SECTION = 6;

// What the scripted job seeker says in each section. Each answer covers the
// section's whole checklist, so one answer plus a confirmation should finish it.
const ANSWERS = {
  background:
    "I'm a backend developer with 3 years of experience. Before that I was a junior web " +
    "developer for 2 years. My key skills are Python, Django, PostgreSQL and Docker. " +
    "My highest education is an MSc in Computer Science.",
  target_role:
    "I'm aiming for Backend Developer or Python Developer roles in Toronto, hybrid. " +
    "My top priorities are growth and a modern tech stack.",
  skill_gap:
    "Here are two postings.\n\n" +
    "Maple Systems - Backend Developer\nRequirements: 3+ years of Python and Django. " +
    "Experience with AWS is required. Kubernetes is a plus.\n\n" +
    "Harbour Labs - Python Developer\nRequirements: strong Python and PostgreSQL. " +
    "AWS experience is required. CI/CD experience is nice to have.\n\n" +
    "My main gap is AWS. My learning action: take an AWS developer course and deploy one " +
    "of my Django projects on AWS.",
  application_strategy:
    "Resume and LinkedIn: I'll highlight Python, Django and my AWS project. Channels: " +
    "LinkedIn, company career pages and referrals. Networking: I'll message 3 backend " +
    "engineers a week. Weekly target: 8 applications and 3 outreach messages.",
  interview_prep:
    "I expect behavioral, technical coding and system design interviews. My weak area is " +
    "system design. Practice plan: coding problems 3 times a week and one mock system design " +
    "interview a week. Stories to prepare: migrating a legacy API, fixing a slow database " +
    "query, and mentoring a junior developer.",
};

const results = [];
function check(name, ok, detail = "") {
  results.push(ok);
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
  return ok;
}

async function getJson(path, init) {
  const res = await fetch(base + path, { redirect: "follow", ...init });
  let data = null;
  try {
    data = await res.json();
  } catch {
    // not JSON: leave data as null
  }
  return { status: res.status, data };
}

/** One chat turn. Returns what a browser would end up showing. */
async function turn(message, threadId) {
  const res = await fetch(`${base}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, threadId, userId: USER_ID }),
  });
  const out = { status: res.status, threadId, tokens: 0, replies: [], section: null, done: false };
  if (!res.ok || !res.body) return out;

  let buffer = "";
  const decoder = new TextDecoder();
  for await (const chunk of res.body) {
    buffer += decoder.decode(chunk, { stream: true });
    let end;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const event = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      for (const line of event.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6);
        if (data === "[DONE]") {
          out.done = true;
          continue;
        }
        let parsed;
        try {
          parsed = JSON.parse(data);
        } catch {
          continue;
        }
        if (parsed.type === "metadata") out.threadId = parsed.content.thread_id;
        else if (parsed.type === "token") out.tokens += 1;
        else if (parsed.type === "message" && parsed.content?.type === "ai") {
          out.replies.push(parsed.content.content);
        } else if (parsed.type === "section") out.section = parsed.content;
      }
    }
  }
  return out;
}

async function main() {
  console.log(`Signed-out smoke test: ${base}${full ? "  (full conversation)" : ""}\n`);

  // 1. the page itself
  const page = await fetch(base + "/");
  const html = await page.text();
  check("page loads", page.status === 200, `HTTP ${page.status}`);
  check("page shows the demo-only warning", html.includes("Demo only"));

  // 2. the API behind it (a sleeping free instance needs time to wake)
  let awake = false;
  for (let attempt = 1; attempt <= 6 && !awake; attempt++) {
    const health = await getJson("/api/health").catch(() => ({ status: 0 }));
    awake = health.status === 200;
    if (!awake) {
      console.log(`      API not awake yet (attempt ${attempt} of 6)`);
      await new Promise((resolve) => setTimeout(resolve, 10_000));
    }
  }
  if (!check("API is reachable through /api/health", awake)) return;

  // 3. one real turn
  const first = await turn(ANSWERS.background, null);
  const reply = first.replies[0] || "";
  check("chat turn is accepted", first.status === 200, `HTTP ${first.status}`);
  check("reply streams token by token", first.tokens > 1, `${first.tokens} tokens`);
  check("stream ends with [DONE]", first.done);
  check("reply came from the model, not the fallback", Boolean(reply) && reply !== FALLBACK_REPLY);
  check("progress reports the Background section", first.section?.id === "background");
  if (!first.threadId) {
    check("turn returns a thread id", false);
    return;
  }
  console.log(`      Thread: ${first.threadId}`);

  // 4. read it back
  const history = await getJson("/api/history", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ threadId: first.threadId }),
  });
  check("history returns the turn", history.data?.messages?.length === 2);
  const early = await getJson(`/api/roadmap?threadId=${first.threadId}&userId=${USER_ID}`);
  check("roadmap is not ready after one turn", early.data?.ready === false);
  if (!full) return;

  // 5. the whole conversation
  let section = first.section;
  let turns = 1;
  let stuck = 0;
  let lastWasAnswer = true; // the first turn above already gave the Background answer
  while (section && !section.roadmap_ready) {
    if (stuck >= MAX_TURNS_PER_SECTION) {
      check(`section "${section.id}" finishes`, false, `no progress after ${stuck} turns`);
      return;
    }
    // Alternate: confirm the summary, and if that did not finish the section, answer again.
    const message = lastWasAnswer ? CONFIRM : ANSWERS[section.id] || CONFIRM;
    lastWasAnswer = !lastWasAnswer;
    const before = `${section.id}:${section.completed_sections}`;
    const next = await turn(message, first.threadId);
    turns += 1;
    if (next.status !== 200 || !next.section) {
      check("conversation continues", false, `HTTP ${next.status} on turn ${turns}`);
      return;
    }
    section = next.section;
    const after = `${section.id}:${section.completed_sections}`;
    if (after === before) {
      stuck += 1;
    } else {
      console.log(`      ${section.completed_sections} of ${section.total_sections} sections done (turn ${turns})`);
      stuck = 0;
      lastWasAnswer = false; // new section: give its answer next
    }
  }
  check("all five sections completed", section?.completed_sections === 5, `${turns} turns`);
  const final = await getJson(`/api/roadmap?threadId=${first.threadId}&userId=${USER_ID}`);
  const roadmap = final.data?.roadmap || "";
  check("roadmap is ready", final.data?.ready === true);
  check("roadmap has a title and a weekly plan table", roadmap.startsWith("#") && roadmap.includes("|"));
  console.log(`\n--- first lines of the roadmap ---\n${roadmap.split("\n").slice(0, 12).join("\n")}\n`);
}

try {
  await main();
} catch (error) {
  check("smoke test ran without crashing", false, String(error?.message || error));
}
const failed = results.filter((ok) => !ok).length;
console.log(`\n${results.length - failed} of ${results.length} checks passed.`);
process.exit(failed ? 1 : 0);
