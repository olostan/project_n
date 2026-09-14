"""
Project N: Child Profile Facts and Clinical Techniques Repository.
Implements the Human-in-the-Loop Confirmation Gate for therapist-suggested techniques.
"""

import sqlite3
from pathlib import Path
from typing import Any

from storage.db_schema import init_db


class FactsRepository:
    """Repository for personal dyadic knowledge and staged clinical techniques."""

    def __init__(self, db: sqlite3.Connection | str | Path = ":memory:") -> None:
        if isinstance(db, sqlite3.Connection):
            self.conn = db
        else:
            self.conn = init_db(db)

    def create_fact(self, fact: dict[str, Any]) -> str:
        """
        Stages a new technique or observation.
        Defaults to confirmed_by_caregiver = 0 (Fail-closed confirmation gate).
        """
        sql = """
        INSERT INTO child_profile_facts (
            id, category, fact_title, description, source_type,
            clinician_role, clinician_id, provenance_episode_id,
            confirmed_by_caregiver, times_tried, times_helpful
        ) VALUES (
            :id, :category, :fact_title, :description, :source_type,
            :clinician_role, :clinician_id, :provenance_episode_id,
            :confirmed_by_caregiver, :times_tried, :times_helpful
        );
        """
        defaults = {
            "clinician_role": None,
            "clinician_id": None,
            "provenance_episode_id": None,
            "confirmed_by_caregiver": 0,
            "times_tried": 0,
            "times_helpful": 0,
        }
        params = {**defaults, **fact}
        with self.conn:
            self.conn.execute(sql, params)
        return str(params["id"])

    def confirm_fact(self, fact_id: str, confirmed: bool = True) -> bool:
        """Caregiver confirms or revokes a staged clinician technique."""
        sql = "UPDATE child_profile_facts SET confirmed_by_caregiver = ? WHERE id = ?"
        with self.conn:
            cursor = self.conn.execute(sql, (1 if confirmed else 0, fact_id))
            return cursor.rowcount > 0

    def list_facts(
        self,
        category: str | None = None,
        source_type: str | None = None,
        confirmed_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Lists facts with optional category and confirmation filters."""
        query = "SELECT * FROM child_profile_facts WHERE 1=1"
        args: list[Any] = []

        if confirmed_only:
            query += " AND confirmed_by_caregiver = 1"
        if category:
            query += " AND category = ?"
            args.append(category)
        if source_type:
            query += " AND source_type = ?"
            args.append(source_type)

        query += " ORDER BY times_helpful DESC, created_at DESC"
        cursor = self.conn.execute(query, tuple(args))
        return [dict(row) for row in cursor.fetchall()]

    def increment_usage(self, fact_id: str, helpful: bool) -> bool:
        """Increments times_tried and conditionally times_helpful."""
        sql = """
        UPDATE child_profile_facts SET
            times_tried = times_tried + 1,
            times_helpful = times_helpful + ?
        WHERE id = ?
        """
        with self.conn:
            cursor = self.conn.execute(sql, (1 if helpful else 0, fact_id))
            return cursor.rowcount > 0
