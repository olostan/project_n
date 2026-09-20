/**
 * Project N: Personal Profile Facts and Clinical Knowledge Types.
 * Strict parity with storage/db_schema.py and storage/facts_repo.py.
 */

export type FactCategory =
  | "comfort_object"
  | "calming_cue"
  | "sensory_trigger"
  | "therapist_technique"
  | "communication_routine";

export type FactSourceType =
  | "home_observation"
  | "ot_session"
  | "slp_session"
  | "school";

export interface ChildProfileFact {
  id: string;
  category: FactCategory;
  fact_title: string;
  description: string;
  source_type: FactSourceType;
  clinician_role?: string | null;
  clinician_id?: string | null;
  provenance_episode_id?: string | null;
  confirmed_by_caregiver: number; // 0 = unconfirmed, 1 = confirmed
  times_tried: number;
  times_helpful: number;
  created_at?: string;
  updated_at?: string;
}

export interface FactCreateRequest {
  category: FactCategory;
  fact_title: string;
  description: string;
  source_type: FactSourceType;
  clinician_role?: string;
  clinician_id?: string;
  provenance_episode_id?: string;
}

export interface FactConfirmRequest {
  confirmed: boolean;
  custom_notes?: string;
}
