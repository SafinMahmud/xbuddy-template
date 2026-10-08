import type { Progress, SectionStatus } from "@/lib/types";

/** The five sections, in the order the conversation visits them. */
export const STEPS = [
  { id: "background", name: "Background", about: "Where you are now" },
  { id: "target_role", name: "Target role", about: "Where you want to go" },
  { id: "skill_gap", name: "Skill gap", about: "What real postings ask for" },
  { id: "application_strategy", name: "Application strategy", about: "How you get interviews" },
  { id: "interview_prep", name: "Interview preparation", about: "How you pass them" },
] as const;

const STATUS_TEXT: Record<SectionStatus, string> = {
  pending: "Not started",
  in_progress: "In progress",
  done: "Done",
};

function statusOf(stepId: string, index: number, progress: Progress | null): SectionStatus {
  if (!progress) return "pending";
  const reported = progress.sections?.find((s) => s.id === stepId);
  if (reported) return reported.status;
  // Older API without the per-section list: infer from the current section.
  const current = STEPS.findIndex((s) => s.id === progress.id);
  if (index < current) return "done";
  return index === current ? progress.status : "pending";
}

interface Props {
  progress: Progress | null;
  showingRoadmap: boolean;
  onOpenRoadmap: () => void;
}

/**
 * Progress drawn as a route: five stops and a destination. It is a real sequence,
 * the conversation visits the stops in this order, and the roadmap is where it ends.
 */
export default function RouteProgress({ progress, showingRoadmap, onOpenRoadmap }: Props) {
  const ready = Boolean(progress?.roadmap_ready);
  return (
    <nav aria-label="Progress">
      <ol className="route">
        {STEPS.map((step, index) => {
          const status = statusOf(step.id, index, progress);
          const current = progress?.id === step.id && !ready;
          return (
            <li
              key={step.id}
              className="stop"
              data-status={status}
              aria-current={current ? "step" : undefined}
            >
              <span className="marker" aria-hidden="true" />
              <span className="stop-name">{step.name}</span>
              <span className="stop-about">
                {status === "pending" ? step.about : STATUS_TEXT[status]}
              </span>
            </li>
          );
        })}
        <li className="stop destination" data-status={ready ? "done" : "pending"}>
          <span className="marker" aria-hidden="true" />
          {ready ? (
            <button
              type="button"
              className="destination-button"
              onClick={onOpenRoadmap}
              aria-pressed={showingRoadmap}
            >
              Open your roadmap
            </button>
          ) : (
            <>
              <span className="stop-name">Your roadmap</span>
              <span className="stop-about">Written when all five are done</span>
            </>
          )}
        </li>
      </ol>
    </nav>
  );
}
