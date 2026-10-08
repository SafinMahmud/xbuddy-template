import type { SavedThread } from "@/lib/types";

interface Props {
  threads: SavedThread[];
  activeId: string | null;
  disabled: boolean;
  onSelect: (threadId: string) => void;
  onNew: () => void;
  onForget: (threadId: string) => void;
}

function when(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/** Conversations this browser started. Picking one reloads it from the server. */
export default function Conversations({
  threads,
  activeId,
  disabled,
  onSelect,
  onNew,
  onForget,
}: Props) {
  return (
    <section className="conversations" aria-label="Conversations">
      <button type="button" className="new-button" onClick={onNew} disabled={disabled}>
        Start a new conversation
      </button>
      {threads.length > 0 && (
        <ul>
          {threads.map((thread) => (
            <li key={thread.threadId} data-active={thread.threadId === activeId}>
              <button
                type="button"
                className="thread-button"
                onClick={() => onSelect(thread.threadId)}
                disabled={disabled}
                aria-current={thread.threadId === activeId ? "true" : undefined}
              >
                <span className="thread-title">{thread.title}</span>
                <span className="thread-date">{when(thread.updatedAt)}</span>
              </button>
              <button
                type="button"
                className="forget-button"
                onClick={() => onForget(thread.threadId)}
                disabled={disabled}
                aria-label={`Remove "${thread.title}" from this list`}
                title="Remove from this list"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
