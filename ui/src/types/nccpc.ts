/**
 * Project N: NCCPC Pain Observation Checklist Types & Subscales.
 * Exact canonical parity with models/nccpc.py.
 */

export type NCCPCResponseValue = 0 | 1 | 2 | 3 | "NA";

export interface NCCPCItem {
  id: string;
  description: string;
}

export interface NCCPCSubscale {
  id: string;
  title: string;
  items: NCCPCItem[];
}

export interface NCCPCScoreResponse {
  episode_id: string;
  instrument: "nccpc_pv" | "nccpc_r";
  total_score: number;
  cutoff_threshold: number;
  pain_cutoff_breached: boolean;
  na_count: number;
  subscale_scores: Record<string, number>;
  escalation_required: boolean;
  guidance: string;
}

export const NCCPC_PV_SUBSCALES: NCCPCSubscale[] = [
  {
    id: "vocal",
    title: "I. Vocal Items",
    items: [
      { id: "moaning_whining_whimpering", description: "Moaning, whining, whimpering (fairly soft)" },
      { id: "crying", description: "Crying (moderately loud)" },
      { id: "screaming_yelling", description: "Screaming/yelling (very loud)" },
      { id: "pain_sound_word", description: "A specific sound or word for pain (e.g., a cry, 'ouch')" },
    ],
  },
  {
    id: "social",
    title: "II. Social Items",
    items: [
      { id: "not_cooperating_cranky", description: "Not cooperating, cranky" },
      { id: "less_interaction_withdrawn", description: "Less interaction with others, withdrawn" },
      { id: "seeking_comfort_closeness", description: "Seeking comfort or physical closeness" },
      { id: "difficult_to_comfort_please", description: "Being difficult to comfort or please" },
    ],
  },
  {
    id: "facial",
    title: "III. Facial Items",
    items: [
      { id: "furrowed_brow", description: "A furrowed brow" },
      { id: "change_in_eyes_squinching", description: "Change in eyes (squinching, open wide, frowning)" },
      { id: "turning_mouth_down", description: "Turning down of mouth, not smiling" },
      { id: "lips_puckering_tight_pouting_quivering", description: "Lips puckering up, tight, pouting, or quivering" },
      { id: "clenching_teeth_chewing_thrusting_tongue", description: "Clenching or grinding teeth, chewing, thrusting tongue" },
    ],
  },
  {
    id: "activity",
    title: "IV. Activity Items",
    items: [
      { id: "not_moving_less_active_quiet", description: "Not moving, less active, quiet" },
      { id: "jumping_agitated_fidgety", description: "Jumping around, agitated, fidgety" },
    ],
  },
  {
    id: "body_and_limbs",
    title: "V. Body and Limbs Items",
    items: [
      { id: "floppy", description: "Floppy" },
      { id: "stiff_spastic_rigid_tense", description: "Stiff, spastic, rigid, tense" },
      { id: "gesturing_touching_hurt_part", description: "Gesturing to or touching a part of the body that seems to hurt" },
      { id: "protecting_favoring_guarding", description: "Protecting, favoring, or guarding part of the body" },
      { id: "flinching_moving_away", description: "Flinching or moving body part away" },
      { id: "moving_body_specifically", description: "Moving body in a specific way (head back, arms/legs out)" },
    ],
  },
  {
    id: "physiological",
    title: "VI. Physiological Signs",
    items: [
      { id: "shivering", description: "Shivering" },
      { id: "change_in_color_pallor", description: "Change in color, pallor" },
      { id: "sweating_perspiring", description: "Sweating, perspiring" },
      { id: "tears", description: "Tears" },
      { id: "sharp_breath_gasping", description: "Sharp intake of breath, gasping" },
      { id: "breath_holding", description: "Breath holding" },
    ],
  },
];

export const NCCPC_R_EXTRA_SUBSCALE: NCCPCSubscale = {
  id: "eating_and_sleeping",
  title: "VII. Eating and Sleeping Items (NCCPC-R Only)",
  items: [
    { id: "eating_less_not_interested", description: "Eating less, not interested in food" },
    { id: "increased_sleep", description: "Increase in sleep" },
    { id: "decreased_sleep", description: "Decrease in sleep" },
  ],
};
