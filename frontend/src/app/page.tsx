"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import Chat, { type ServerState } from "@/components/Chat";
import Conversations from "@/components/Conversations";
import Roadmap from "@/components/Roadmap";
import RouteProgress from "@/components/RouteProgress";
import { readSse } from "@/lib/sse";
import {
  forgetThread,
  getActiveThread,
  getUserId,
  listThreads,
  saveThread,
  setActiveThread,
  titleFrom,
} from "@/lib/storage";
import type { ChatMessage, Progress, SavedThread } from "@/lib/types";

const PLACEHOLDERS: Record<string, string> = {
  background: "Your current role, years of experience, main skills…",
  target_role: "The titles you want, where, and what matters most to you…",
  skill_gap: "Paste a job posting, or ask JobBuddy to find some openings…",
  application_strategy: "How you plan to apply, or ask to see current openings…",
  interview_prep: "The interviews you expect and what worries you…",
};
const DEFAULT_PLACEHOLDER = "Type your answer. Shift+Enter adds a new line.";
const FAILED_TURN = "That message did not go through. Check your connection and send it again.";
const CUT_OFF = "The reply stopped before it finished. Send your message again.";

let nextId = 0;
const newId = (prefix: string) => `${prefix}${Date.now()}-${nextId++}`;

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [threads, setThreads] = useState<SavedThread[]>([]);
  const [busy, setBusy] = useState(false); // a reply is being written
  const [loading, setLoading] = useState(false); // a saved conversation is being loaded
  const [view, setView] = useState<"chat" | "roadmap">("chat");
  const [server, setServer] = useState<ServerState>("checking");
  const userId = useRef(0);
  const threadRef = useRef<string | null>(null); // current thread, readable inside the stream loop

  const activate = useCallback((id: string | null) => {
    threadRef.current = id;
    setThreadId(id);
    setActiveThread(id);
  }, []);

  const openThread = useCallback(
    async (id: string) => {
      setLoading(true);
      setView("chat");
      activate(id);
      try {
        const res = await fetch("/api/history", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ threadId: id }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        if (threadRef.current !== id) return; // the visitor picked something else meanwhile
        if (data.messages.length === 0) {
          // The server no longer has it (for example the demo database was reset).
          setThreads(forgetThread(id));
          activate(null);
          setMessages([]);
          setProgress(null);
        } else {
          setMessages(data.messages);
          setProgress(data.section);
        }
      } catch {
        if (threadRef.current !== id) return;
        setMessages([
          {
            id: newId("e"),
            role: "assistant",
            failed: true,
            content: "This conversation could not be loaded. The server may still be waking up. Pick it again in a moment.",
          },
        ]);
        setProgress(null);
      } finally {
        if (threadRef.current === id || threadRef.current === null) setLoading(false);
      }
    },
    [activate],
  );

  // On first load: identify this browser, wake the API, and reopen the last conversation.
  useEffect(() => {
    userId.current = getUserId();
    setThreads(listThreads());

    let cancelled = false;
    const slow = setTimeout(() => !cancelled && setServer((s) => (s === "checking" ? "waking" : s)), 2500);
    (async () => {
      for (let attempt = 0; attempt < 3 && !cancelled; attempt++) {
        try {
          const res = await fetch("/api/health", { cache: "no-store" });
          if (res.ok) {
            if (!cancelled) setServer("ready");
            return;
          }
        } catch {
          // fall through to the next attempt
        }
      }
      if (!cancelled) setServer("down");
    })();

    const last = getActiveThread();
    if (last) void openThread(last);

    return () => {
      cancelled = true;
      clearTimeout(slow);
    };
  }, [openThread]);

  const startNew = () => {
    activate(null);
    setMessages([]);
    setProgress(null);
    setView("chat");
  };

  const forget = (id: string) => {
    setThreads(forgetThread(id));
    if (id === threadRef.current) startNew();
  };

  const send = async (text: string) => {
    const content = text.trim();
    if (!content || busy) return;
    const replyId = newId("a");
    let openId: string | null = replyId; // the reply bubble currently receiving tokens
    let received = false;
    const startedIn = threadRef.current;

    setBusy(true);
    setMessages((prev) => [
      ...prev,
      { id: newId("u"), role: "user", content },
      { id: replyId, role: "assistant", content: "", open: true },
    ]);

    const patch = (id: string, change: (m: ChatMessage) => ChatMessage) =>
      setMessages((prev) => prev.map((m) => (m.id === id ? change(m) : m)));

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: content, threadId: startedIn, userId: userId.current }),
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      setServer("ready");

      for await (const data of readSse(res.body)) {
        if (data === "[DONE]") break;
        let event: { type: string; content: any }; // eslint-disable-line @typescript-eslint/no-explicit-any
        try {
          event = JSON.parse(data);
        } catch {
          continue; // one unreadable event should not end the reply
        }

        if (event.type === "metadata" && event.content?.thread_id && !threadRef.current) {
          activate(event.content.thread_id);
          setThreads(saveThread(event.content.thread_id, titleFrom(content)));
        } else if (event.type === "token" && typeof event.content === "string") {
          received = true;
          if (!openId) {
            // A second message in the same turn (the roadmap after the last reply).
            const id = (openId = newId("a"));
            setMessages((prev) => [...prev, { id, role: "assistant", content: "", open: true }]);
          }
          const id = openId;
          patch(id, (m) => ({ ...m, content: m.content + event.content }));
        } else if (event.type === "message" && event.content?.type === "ai" && event.content.content) {
          // The complete message. It replaces the streamed text, so the bubble is
          // right even if a token was lost on the way.
          received = true;
          const full: string = event.content.content;
          if (openId) {
            patch(openId, (m) => ({ ...m, content: full, open: false }));
            openId = null;
          } else {
            setMessages((prev) => [...prev, { id: newId("a"), role: "assistant", content: full }]);
          }
        } else if (event.type === "section" && event.content) {
          setProgress(event.content as Progress);
        }
      }

      if (openId) {
        const id = openId;
        patch(id, (m) =>
          m.content ? { ...m, open: false } : { ...m, open: false, failed: true, content: CUT_OFF },
        );
      }
    } catch {
      const id = openId ?? replyId;
      // Keep whatever part of the reply arrived. Only an empty bubble becomes a notice.
      patch(id, (m) =>
        received && m.content
          ? { ...m, open: false }
          : { ...m, open: false, failed: true, content: FAILED_TURN },
      );
    } finally {
      setBusy(false);
      if (threadRef.current) setThreads(saveThread(threadRef.current));
    }
  };

  const showRoadmap = view === "roadmap" && threadId && progress?.roadmap_ready;

  return (
    <div className="app">
      <aside className="sidebar">
        <p className="wordmark">JobBuddy</p>
        <RouteProgress
          progress={progress}
          showingRoadmap={Boolean(showRoadmap)}
          onOpenRoadmap={() => setView("roadmap")}
        />
        <Conversations
          threads={threads}
          activeId={threadId}
          disabled={busy}
          onSelect={openThread}
          onNew={startNew}
          onForget={forget}
        />
      </aside>

      <main className="main">
        {showRoadmap ? (
          <Roadmap threadId={threadId} userId={userId.current} onBack={() => setView("chat")} />
        ) : (
          <Chat
            messages={messages}
            busy={busy}
            loading={loading}
            server={server}
            placeholder={(progress && PLACEHOLDERS[progress.id]) || DEFAULT_PLACEHOLDER}
            onSend={send}
          />
        )}
      </main>
    </div>
  );
}
