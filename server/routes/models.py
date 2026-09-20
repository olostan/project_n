"""
Project N: Model Checkpoints & Promotion Gate Routes.
"""

import contextlib
import sqlite3
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from server.deps import get_analysis_service, get_db
from server.services.analysis_service import AnalysisService

router = APIRouter(prefix="/api/v1/models", tags=["models"])


class PromoteRequest(BaseModel):
    candidate_id: str


@router.get("/checkpoints")
def list_checkpoints(
    db: sqlite3.Connection = Depends(get_db),
) -> dict[str, Any]:
    """Lists registered model checkpoints with evaluation metrics."""
    cursor = db.execute("SELECT * FROM model_checkpoints ORDER BY is_production DESC, ROWID DESC")
    rows = cursor.fetchall()
    checkpoints = [dict(r) for r in rows]
    return {"checkpoints": checkpoints, "total": len(checkpoints)}


@router.post("/promote")
def promote_checkpoint(
    req: PromoteRequest,
    db: sqlite3.Connection = Depends(get_db),
    analysis: AnalysisService = Depends(get_analysis_service),
) -> dict[str, Any]:
    """
    Caregiver promotes candidate model checkpoint to production.
    Enforces Invariant 5: zero distress misses required.
    """
    cursor = db.execute("SELECT * FROM model_checkpoints WHERE id = ?", (req.candidate_id,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Checkpoint {req.candidate_id} not found.",
        )

    ckpt = dict(row)
    if ckpt.get("zero_distress_misses", 1) != 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Safety invariant violation: cannot promote model with acute distress misses.",
        )

    now_iso = datetime.now(UTC).isoformat()
    with db:
        db.execute("UPDATE model_checkpoints SET is_production = 0")
        db.execute(
            "UPDATE model_checkpoints SET is_production = 1, promoted_at = ? WHERE id = ?",
            (now_iso, req.candidate_id),
        )

    # Reload checkpoint in active analysis service if file exists
    ckpt_path = ckpt.get("checkpoint_path")
    if ckpt_path:
        with contextlib.suppress(Exception):
            analysis.projection_head.load_checkpoint(ckpt_path)

    return {
        "status": "promoted",
        "checkpoint_id": req.candidate_id,
        "timestamp": now_iso,
    }


@router.post("/rollback")
def rollback_checkpoint(
    db: sqlite3.Connection = Depends(get_db),
    analysis: AnalysisService = Depends(get_analysis_service),
) -> dict[str, Any]:
    """Reverts active production pointer to prior stable checkpoint."""
    cursor = db.execute(
        "SELECT * FROM model_checkpoints WHERE is_production = 0 AND evaluation_status = 'passed' ORDER BY ROWID DESC LIMIT 1"
    )
    row = cursor.fetchone()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No prior stable checkpoint available for rollback.",
        )

    prior = dict(row)
    now_iso = datetime.now(UTC).isoformat()
    with db:
        db.execute("UPDATE model_checkpoints SET is_production = 0")
        db.execute(
            "UPDATE model_checkpoints SET is_production = 1, promoted_at = ? WHERE id = ?",
            (now_iso, prior["id"]),
        )

    ckpt_path = prior.get("checkpoint_path")
    if ckpt_path:
        with contextlib.suppress(Exception):
            analysis.projection_head.load_checkpoint(ckpt_path)

    return {
        "status": "rolled_back",
        "checkpoint_id": prior["id"],
        "timestamp": now_iso,
    }
