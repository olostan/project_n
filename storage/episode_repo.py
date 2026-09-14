"""
Project N: Episode Repository.
Data access layer for saving, retrieving, and updating verified episodes.
"""

import sqlite3
from pathlib import Path
from typing import Any

from storage.db_schema import init_db


class EpisodeRepository:
    """Typed SQLite repository for Child N's episodic records."""

    def __init__(self, db: sqlite3.Connection | str | Path = ":memory:") -> None:
        if isinstance(db, sqlite3.Connection):
            self.conn = db
        else:
            self.conn = init_db(db)

    def insert_episode(self, ep: dict[str, Any]) -> str:
        """Inserts an episode record into SQLite."""
        sql = """
        INSERT INTO episodes (
            id, vault_uri, encoder_version_id, captured_at, duration_ms, windows_count,
            observed_f0_mean, observed_motion_rhythm_hz, eda_tonic_level,
            antecedent_id, antecedent_notes, caregiver_hypothesis,
            action_offered, action_custom_label, action_performed, performance_status,
            action_notes, caregiver_decision, outcome_state, settled_within_sec,
            child_response, response_channel, response_independence, prompt_level,
            observer, nccpc_instrument, nccpc_score, pain_cutoff_breached
        ) VALUES (
            :id, :vault_uri, :encoder_version_id, :captured_at, :duration_ms, :windows_count,
            :observed_f0_mean, :observed_motion_rhythm_hz, :eda_tonic_level,
            :antecedent_id, :antecedent_notes, :caregiver_hypothesis,
            :action_offered, :action_custom_label, :action_performed, :performance_status,
            :action_notes, :caregiver_decision, :outcome_state, :settled_within_sec,
            :child_response, :response_channel, :response_independence, :prompt_level,
            :observer, :nccpc_instrument, :nccpc_score, :pain_cutoff_breached
        );
        """
        defaults = {
            "observed_f0_mean": None,
            "observed_motion_rhythm_hz": None,
            "eda_tonic_level": None,
            "antecedent_notes": None,
            "caregiver_hypothesis": None,
            "action_custom_label": None,
            "action_performed": None,
            "performance_status": None,
            "action_notes": None,
            "caregiver_decision": None,
            "outcome_state": None,
            "settled_within_sec": None,
            "child_response": "none",
            "response_channel": "none",
            "response_independence": "none",
            "prompt_level": "none",
            "observer": "primary_caregiver",
            "nccpc_instrument": "none",
            "nccpc_score": None,
            "pain_cutoff_breached": -1,
        }
        params = {**defaults, **ep}
        with self.conn:
            self.conn.execute(sql, params)
        return str(params["id"])

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        """Retrieves a single episode by ID."""
        cursor = self.conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_episodes(
        self,
        limit: int = 50,
        offset: int = 0,
        antecedent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Lists episodes with optional antecedent filter."""
        if antecedent_id:
            query = "SELECT * FROM episodes WHERE antecedent_id = ? ORDER BY captured_at DESC LIMIT ? OFFSET ?"
            cursor = self.conn.execute(query, (antecedent_id, limit, offset))
        else:
            query = "SELECT * FROM episodes ORDER BY captured_at DESC LIMIT ? OFFSET ?"
            cursor = self.conn.execute(query, (limit, offset))
        return [dict(row) for row in cursor.fetchall()]

    def update_outcome(
        self,
        episode_id: str,
        action_performed: str,
        outcome_state: str,
        settled_within_sec: int | None = None,
        child_response: str = "none",
    ) -> bool:
        """Updates caregiver observed action and resolution state."""
        sql = """
        UPDATE episodes SET
            action_performed = ?,
            outcome_state = ?,
            settled_within_sec = ?,
            child_response = ?
        WHERE id = ?
        """
        with self.conn:
            cursor = self.conn.execute(
                sql,
                (action_performed, outcome_state, settled_within_sec, child_response, episode_id),
            )
            return cursor.rowcount > 0

    def record_nccpc_score(
        self,
        episode_id: str,
        instrument: str,
        score: int,
        cutoff_breached: bool,
    ) -> bool:
        """Records clinical pain observation result for triage tracking."""
        sql = """
        UPDATE episodes SET
            nccpc_instrument = ?,
            nccpc_score = ?,
            pain_cutoff_breached = ?
        WHERE id = ?
        """
        with self.conn:
            cursor = self.conn.execute(
                sql, (instrument, score, 1 if cutoff_breached else 0, episode_id)
            )
            return cursor.rowcount > 0

    def count_episodes(self) -> int:
        """Returns total episode count."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM episodes")
        return int(cursor.fetchone()[0])
