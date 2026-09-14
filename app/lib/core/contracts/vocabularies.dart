/// Project N: Canonical Controlled Vocabularies.
/// Mirrored from models/contracts.py for type safety across mobile companion client.
library;

const List<String> controlledActions = [
  'quiet_refuge',
  'dimmed_lighting',
  'hydration_water',
  'deep_pressure_proprioceptive',
  'vestibular_rocking',
  'preferred_comfort_object',
  'sensory_break',
  'motor_movement_break',
  'warm_compress',
  'aac_choice_board',
  'open_observation',
  'other_custom',
];

const List<String> controlledAntecedents = [
  'post_school_transition',
  'mealtime',
  'bedtime_routine',
  'loud_environment',
  'unfamiliar_setting',
  'physical_transition',
  'preferred_activity_ended',
  'unknown',
];

const List<String> outcomeStates = [
  'settled_immediately',
  'settled_delayed',
  'no_change',
  'escalated',
];

const int minClipDurationSec = 30;
const int maxClipDurationSec = 120;
const int nccpcPvCutoffThreshold = 11;
