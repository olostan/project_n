"""
Project N: Episode Management & Clinical Pain Scoring Routes.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from models.nccpc import score_nccpc_pv, score_nccpc_r
from server.deps import get_episode_repo, get_vault, verify_auth_token
from server.schemas.episodes import (
    EpisodeListResponse,
    EpisodeResponse,
    NCCPCScoringResponse,
    NCCPCSubmissionRequest,
)
from storage.episode_repo import EpisodeRepository
from storage.vault import VaultManager

router = APIRouter(prefix="/api/v1/episodes", tags=["episodes"])


@router.get("", response_model=EpisodeListResponse)
def list_episodes(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    antecedent_id: str | None = None,
    repo: EpisodeRepository = Depends(get_episode_repo),
    _token: str = Depends(verify_auth_token),
) -> dict[str, Any]:
    """Lists verified historical episodes with optional antecedent filter."""
    episodes = repo.list_episodes(limit=limit, offset=offset, antecedent_id=antecedent_id)
    total = repo.count_episodes()
    return {"episodes": episodes, "total": total}


@router.get("/{episode_id}", response_model=EpisodeResponse)
def get_episode(
    episode_id: str,
    repo: EpisodeRepository = Depends(get_episode_repo),
    _token: str = Depends(verify_auth_token),
) -> dict[str, Any]:
    """Retrieves an episode by ID."""
    ep = repo.get_episode(episode_id)
    if not ep:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Episode not found.")
    return ep


@router.get("/{episode_id}/media")
def stream_episode_media(
    episode_id: str,
    repo: EpisodeRepository = Depends(get_episode_repo),
    _vault: VaultManager = Depends(get_vault),
    _token: str = Depends(verify_auth_token),
) -> Response:
    """Streams decrypted video media for playback."""
    ep = repo.get_episode(episode_id)
    if not ep:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Episode not found.")

    # In production, retrieves ciphertext and unwrap DEK. For testing/mock:
    media_bytes = b"mock_decrypted_mp4_video_bytes"
    return Response(content=media_bytes, media_type="video/mp4")


@router.post("/{episode_id}/nccpc", response_model=NCCPCScoringResponse)
def submit_nccpc_pain_checklist(
    episode_id: str,
    req: NCCPCSubmissionRequest,
    repo: EpisodeRepository = Depends(get_episode_repo),
    _token: str = Depends(verify_auth_token),
) -> dict[str, Any]:
    """
    Submits caregiver-completed NCCPC pain observation checklist.
    Delegates to canonical models.nccpc scoring engine and updates episode triage status.
    """
    ep = repo.get_episode(episode_id)
    if not ep:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Episode not found.")

    try:
        if req.instrument == "nccpc_pv":
            score_res = score_nccpc_pv(req.scores, strict_keys=False)
        else:
            score_res = score_nccpc_r(req.scores, strict_keys=False)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    # Record score into SQLite repository
    repo.record_nccpc_score(
        episode_id=episode_id,
        instrument=req.instrument,
        score=score_res.score,
        cutoff_breached=score_res.cutoff_breached,
    )

    guidance = (
        "MEDICAL ESCALATION REQUIRED: Observed pain threshold exceeded. "
        "Prompt caregiver to follow family pediatrician-approved physical comfort protocol. "
        "All behavioral interpretations suppressed."
        if score_res.cutoff_breached
        else "Checklist completed below pain cut-off threshold. Continue observing Child N."
    )

    return {
        "episode_id": episode_id,
        "instrument": score_res.instrument,
        "total_score": score_res.score,
        "cutoff_threshold": score_res.cutoff_threshold,
        "pain_cutoff_breached": score_res.cutoff_breached,
        "na_count": score_res.na_count,
        "subscale_scores": score_res.subscale_scores,
        "escalation_required": score_res.cutoff_breached,
        "guidance": guidance,
    }
