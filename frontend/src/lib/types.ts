export type SectionStatus = "pending" | "in_progress" | "done";

export interface SectionStep {
  id: string;
  name: string;
  status: SectionStatus;
}

/** Progress object the API attaches to every reply (service.section_progress). */
export interface Progress {
  id: string;
  name: string;
  status: SectionStatus;
  completed_sections: number;
  total_sections: number;
  roadmap_ready: boolean;
  sections?: SectionStep[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  /** Still receiving tokens. */
  open?: boolean;
  /** A failed turn, shown as a notice instead of a reply. */
  failed?: boolean;
}

export interface SavedThread {
  threadId: string;
  title: string;
  updatedAt: string;
}
