-- JobBuddy: keep the user's confirmed answer separate from the unconfirmed draft.
-- content / plain_text hold the draft; confirmed_summary holds what the user confirmed.
ALTER TABLE section_states ADD COLUMN IF NOT EXISTS confirmed_summary TEXT;

-- section_id values for JobBuddy:
-- 'background', 'target_role', 'skill_gap', 'application_strategy', 'interview_prep'
