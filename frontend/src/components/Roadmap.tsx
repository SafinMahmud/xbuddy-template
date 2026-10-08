"use client";

import { useEffect, useState } from "react";

import Markdown from "./Markdown";

interface Props {
  threadId: string;
  userId: number;
  onBack: () => void;
}

type State =
  | { kind: "loading" }
  | { kind: "error"; text: string }
  | { kind: "ready"; roadmap: string };

/** The finished roadmap as a document the visitor can copy or download. */
export default function Roadmap({ threadId, userId, onBack }: Props) {
  const [state, setState] = useState<State>({ kind: "loading" });
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setState({ kind: "loading" });
    fetch(`/api/roadmap?threadId=${encodeURIComponent(threadId)}&userId=${userId}`)
      .then(async (res) => {
        const data = await res.json();
        if (cancelled) return;
        if (!res.ok) setState({ kind: "error", text: data.error ?? "The roadmap could not be loaded." });
        else if (!data.ready) setState({ kind: "error", text: "The roadmap is not written yet. Finish all five sections first." });
        else setState({ kind: "ready", roadmap: data.roadmap });
      })
      .catch(() => {
        if (!cancelled) setState({ kind: "error", text: "The roadmap could not be loaded. Check your connection and try again." });
      });
    return () => {
      cancelled = true;
    };
  }, [threadId, userId]);

  const copy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard access can be blocked. Download still works.
    }
  };

  const download = (text: string) => {
    const url = URL.createObjectURL(new Blob([text], { type: "text/markdown;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "job-search-roadmap.md";
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="roadmap">
      <div className="roadmap-bar">
        <button type="button" className="text-button" onClick={onBack}>
          Back to the conversation
        </button>
        {state.kind === "ready" && (
          <div className="roadmap-actions">
            <button type="button" className="text-button" onClick={() => copy(state.roadmap)}>
              {copied ? "Copied" : "Copy as Markdown"}
            </button>
            <button type="button" className="send-button" onClick={() => download(state.roadmap)}>
              Download .md
            </button>
          </div>
        )}
      </div>
      <div className="roadmap-page">
        {state.kind === "loading" && <p className="quiet">Loading your roadmap…</p>}
        {state.kind === "error" && <p className="quiet">{state.text}</p>}
        {state.kind === "ready" && <Markdown>{state.roadmap}</Markdown>}
      </div>
    </div>
  );
}
