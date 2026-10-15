"use client";

import { useEffect, useRef, useState } from "react";

import type { ChatMessage } from "@/lib/types";

import Markdown from "./Markdown";

export type ServerState = "checking" | "waking" | "ready" | "down";

interface Props {
  messages: ChatMessage[];
  busy: boolean;
  loading: boolean;
  server: ServerState;
  placeholder: string;
  onSend: (text: string) => void;
}

const SERVER_NOTE: Partial<Record<ServerState, string>> = {
  waking:
    "Waking the server. It runs on a free host that sleeps when idle, so the first reply can take up to a minute.",
  down: "The server is not answering yet. Wait a moment, then send your message.",
};

export default function Chat({ messages, busy, loading, server, placeholder, onSend }: Props) {
  const [draft, setDraft] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Keep the newest text in view while a reply streams in.
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  // Grow the input with its content, up to the CSS max-height.
  useEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    input.style.height = "auto";
    input.style.height = `${input.scrollHeight}px`;
  }, [draft]);

  useEffect(() => {
    if (!busy) inputRef.current?.focus();
  }, [busy]);

  const canSend = draft.trim().length > 0 && !busy && !loading;

  const submit = () => {
    if (!canSend) return;
    onSend(draft);
    setDraft("");
  };

  return (
    <div className="chat">
      <div className="messages" aria-live="polite">
        <div className="column">
          {loading ? (
            <p className="quiet">Loading your conversation…</p>
          ) : messages.length === 0 ? (
            <div className="welcome">
              <h1>Plan your job search in five steps</h1>
              <p>
                JobBuddy asks about your background, your target role, what real job postings
                require, how you will apply, and how you will prepare for interviews. Then it
                writes you an eight-week roadmap built only from what you told it.
              </p>
              <p className="first-question">
                To start, what is your current or most recent job title?
              </p>
            </div>
          ) : (
            messages.map((message) => (
              <article
                key={message.id}
                className="message"
                data-role={message.role}
                data-failed={message.failed || undefined}
              >
                <h2 className="visually-hidden">{message.role === "user" ? "You" : "JobBuddy"}</h2>
                {message.role === "user" ? (
                  <p className="user-text">{message.content}</p>
                ) : message.content ? (
                  <Markdown>{message.content}</Markdown>
                ) : (
                  <p className="quiet thinking">Thinking…</p>
                )}
              </article>
            ))
          )}
          <div ref={endRef} />
        </div>
      </div>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
      >
        <div className="column">
          {SERVER_NOTE[server] && (
            <p className="server-note" role="status">
              {SERVER_NOTE[server]}
            </p>
          )}
          <div className="composer-row">
            <label className="visually-hidden" htmlFor="message">
              Your message
            </label>
            <textarea
              id="message"
              ref={inputRef}
              rows={1}
              value={draft}
              placeholder={placeholder}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                // Enter sends. Shift+Enter adds a line, for pasted postings and lists.
                if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                  event.preventDefault();
                  submit();
                }
              }}
              disabled={loading}
            />
            <button type="submit" className="send-button" disabled={!canSend}>
              {busy ? "Replying…" : "Send"}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
