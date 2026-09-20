/**
 * Project N: Episode & Contract Data Models.
 * Strict 1-to-1 parity with models/contracts.py and storage/db_schema.py.
 */

export type ControlledAction =
  | "quiet_refuge"
  | "dimmed_lighting"
  | "hydration_water"
  | "deep_pressure_proprioceptive"
  | "vestibular_rocking"
  | "preferred_comfort_object"
  | "sensory_break"
  | "motor_movement_break"
  | "warm_compress"
  | "aac_choice_board"
  | "open_observation"
  | "other_custom";

export type ControlledAntecedent =
  | "post_school_transition"
  | "mealtime"
  | "bedtime_routine"
  | "loud_environment"
  | "unfamiliar_setting"
  | "physical_transition"
  | "preferred_activity_ended"
  | "unknown";

export type OutcomeState =
  | "settled_immediately"
  | "settled_delayed"
  | "no_change"
  | "escalated";

export type ChildResponse =
  | "reach"
  | "gesture"
  | "vocal_signal"
  | "aac_selection"
  | "none";

export type ResponseChannel = "motor" | "vocal" | "aac" | "none";

export type ResponseIndependence =
  | "independent"
  | "prompted"
  | "passive"
  | "refusal"
  | "none";

export type PerformanceStatus =
  | "completed"
  | "attempted_refused"
  | "aborted"
  | "not_attempted";

export type CaregiverDecision =
  | "accepted"
  | "modified"
  | "rejected"
  | "open_observation";

export type PromptLevel =
  | "none"
  | "visual_cue"
  | "verbal_model"
  | "gestural"
  | "physical_prompt";

export interface Episode {
  id: string;
  vault_uri: string;
  encoder_version_id: string;
  captured_at: string;
  duration_ms: number;
  windows_count: number;
  observed_f0_mean: number | null;
  observed_motion_rhythm_hz: number | null;
  eda_tonic_level: number | null;
  antecedent_id: ControlledAntecedent;
  antecedent_notes: string | null;
  caregiver_hypothesis: string | null;
  action_offered: ControlledAction;
  action_custom_label: string | null;
  action_performed: ControlledAction | null;
  performance_status: PerformanceStatus | null;
  action_notes: string | null;
  caregiver_decision: CaregiverDecision | null;
  outcome_state: OutcomeState | null;
  settled_within_sec: number | null;
  child_response: ChildResponse | null;
  response_channel: ResponseChannel | null;
  response_independence: ResponseIndependence | null;
  prompt_level: PromptLevel | null;
  observer: string | null;
  nccpc_instrument: "nccpc_pv" | "nccpc_r" | "none" | null;
  nccpc_score: number | null;
  pain_cutoff_breached: number | null; // -1 = unassessed, 0 = no, 1 = breached
  metric_embedding: string | null;
  created_at?: string;
  umap_x?: number;
  umap_y?: number;
}

export interface EpisodeListResponse {
  episodes: Episode[];
  total: number;
}

export interface FourLayerCardData {
  task_id: string;
  L1_measured: string;
  L2_historical: string;
  L3_context: string;
  L4_evidence: string;
  view_mode: "parent" | "therapist";
  parent_view_text?: string;
  abstained: boolean;
}

export interface EpisodeOutcomeRequest {
  action_performed: ControlledAction;
  outcome_state: OutcomeState;
  caregiver_decision: CaregiverDecision;
  settled_within_sec?: number | null;
  child_response: ChildResponse;
  response_channel: ResponseChannel;
  response_independence: ResponseIndependence;
  performance_status: PerformanceStatus;
}
