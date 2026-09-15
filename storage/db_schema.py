"""
Project N: Relational Database Schema & DDL.
Maintains typed SQLite schema with CHECK constraints across all controlled vocabularies.
"""

import contextlib
import sqlite3
from pathlib import Path

CREATE_EPISODES_TABLE = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    vault_uri TEXT NOT NULL,
    encoder_version_id TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    windows_count INTEGER NOT NULL,
    observed_f0_mean REAL,
    observed_motion_rhythm_hz REAL,
    eda_tonic_level REAL,
    antecedent_id TEXT NOT NULL CHECK(antecedent_id IN (
        'post_school_transition',
        'mealtime',
        'bedtime_routine',
        'loud_environment',
        'unfamiliar_setting',
        'physical_transition',
        'preferred_activity_ended',
        'unknown'
    )),
    antecedent_notes TEXT,
    caregiver_hypothesis TEXT,
    action_offered TEXT NOT NULL CHECK(action_offered IN (
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
        'other_custom'
    )),
    action_custom_label TEXT,
    action_performed TEXT CHECK(action_performed IN (
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
        'other_custom'
    )),
    performance_status TEXT CHECK(performance_status IN (
        'completed',
        'attempted_refused',
        'aborted',
        'not_attempted'
    )),
    action_notes TEXT,
    caregiver_decision TEXT CHECK(caregiver_decision IN (
        'accepted',
        'modified',
        'rejected',
        'open_observation'
    )),
    outcome_state TEXT CHECK(outcome_state IN (
        'settled_immediately',
        'settled_delayed',
        'no_change',
        'escalated'
    )),
    settled_within_sec INTEGER,
    child_response TEXT CHECK(child_response IN ('reach', 'gesture', 'vocal_signal', 'aac_selection', 'none')),
    response_channel TEXT CHECK(response_channel IN ('motor', 'vocal', 'aac', 'none')),
    response_independence TEXT CHECK(response_independence IN ('independent', 'prompted', 'passive', 'refusal', 'none')),
    prompt_level TEXT CHECK(prompt_level IN ('none', 'visual_cue', 'verbal_model', 'gestural', 'physical_prompt')),
    observer TEXT,
    nccpc_instrument TEXT CHECK(nccpc_instrument IN ('nccpc_pv', 'nccpc_r', 'none')),
    nccpc_score INTEGER,
    pain_cutoff_breached INTEGER DEFAULT -1 CHECK(pain_cutoff_breached IN (-1, 0, 1)),
    metric_embedding TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_MODEL_CHECKPOINTS_TABLE = """
CREATE TABLE IF NOT EXISTS model_checkpoints (
    id TEXT PRIMARY KEY,
    checkpoint_path TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    validation_manifest_hash TEXT,
    retrieval_mrr REAL NOT NULL,
    holdout_coverage REAL NOT NULL,
    ece_score REAL,
    evaluation_status TEXT NOT NULL DEFAULT 'pending' CHECK(evaluation_status IN ('pending', 'passed', 'failed')),
    distress_cases_evaluated INTEGER DEFAULT 0,
    zero_distress_misses INTEGER,
    validation_timestamp TEXT,
    validation_report_hash TEXT,
    caregiver_utility_score REAL,
    is_production INTEGER DEFAULT 0,
    promoted_at TEXT,
    notes TEXT
);
"""

CREATE_FACTS_TABLE = """
CREATE TABLE IF NOT EXISTS child_profile_facts (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL CHECK(category IN (
        'comfort_object',
        'calming_cue',
        'sensory_trigger',
        'therapist_technique',
        'communication_routine'
    )),
    fact_title TEXT NOT NULL,
    description TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK(source_type IN (
        'home_observation',
        'ot_session',
        'slp_session',
        'school'
    )),
    clinician_role TEXT,
    clinician_id TEXT,
    provenance_episode_id TEXT,
    confirmed_by_caregiver INTEGER DEFAULT 0,
    times_tried INTEGER DEFAULT 0,
    times_helpful INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db(db_path: str | Path = ":memory:") -> sqlite3.Connection:
    """
    Initializes SQLite connection and creates all required tables.

    Args:
        db_path: File path or ':memory:' for tests.

    Returns:
        sqlite3.Connection object with row_factory set to sqlite3.Row.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(CREATE_EPISODES_TABLE)
        conn.execute(CREATE_MODEL_CHECKPOINTS_TABLE)
        conn.execute(CREATE_FACTS_TABLE)
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("ALTER TABLE episodes ADD COLUMN metric_embedding TEXT;")
    return conn
