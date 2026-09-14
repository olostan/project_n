"""
Project N: Episode Management & Clinical Pain Scoring Routes.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from models.nccpc import score_nccpc_pv, score_nccpc_r
from rag.vector_store import VectorStore
from server.deps import (
    get_analysis_service,
    get_episode_repo,
    get_vault,
    get_vector_store,
    verify_auth_token,
)
from server.schemas.episodes import (
    EpisodeListResponse,
    EpisodeOutcomeRequest,
    EpisodeOutcomeResponse,
    EpisodeResponse,
    NCCPCScoringResponse,
    NCCPCSubmissionRequest,
)
from server.services.analysis_service import AnalysisService
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
    vault: VaultManager = Depends(get_vault),
    _token: str = Depends(verify_auth_token),
) -> Response:
    """Streams decrypted video media for playback from the secure encrypted vault."""
    ep = repo.get_episode(episode_id)
    if not ep:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Episode not found.")

    try:
        media_bytes = vault.retrieve_clip(episode_id)
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Media for episode {episode_id} not found in encrypted vault.",
        ) from err

    return Response(
        content=media_bytes,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(len(media_bytes)),
        },
    )


@router.post("/{episode_id}/outcome", response_model=EpisodeOutcomeResponse)
@router.post("/{episode_id}/confirm", response_model=EpisodeOutcomeResponse)
def confirm_episode_outcome(
    episode_id: str,
    req: EpisodeOutcomeRequest,
    repo: EpisodeRepository = Depends(get_episode_repo),
    analysis: AnalysisService = Depends(get_analysis_service),
    vector_store: VectorStore = Depends(get_vector_store),
    _token: str = Depends(verify_auth_token),
) -> dict[str, Any]:
    """
    Caregiver confirms or reports intervention outcome and indexes confirmed episode in ChromaDB.
    """
    ep = repo.get_episode(episode_id)
    if not ep:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Episode not found.")

    repo.update_outcome(
        episode_id=episode_id,
        action_performed=req.action_performed,
        outcome_state=req.outcome_state,
        caregiver_decision=req.caregiver_decision,
        settled_within_sec=req.settled_within_sec,
        child_response=req.child_response,
        response_channel=req.response_channel,
        response_independence=req.response_independence,
        performance_status=req.performance_status,
    )

    indexed = False
    clip_embedding = analysis.get_clip_embedding(episode_id)
    if clip_embedding is not None:
        vector_store.add_episode_embedding(
            episode_id=episode_id,
            embedding=clip_embedding,
            metadata={
                "episode_id": episode_id,
                "encoder_version_id": ep.get("encoder_version_id", "v1.0.0"),
                "windows_count": ep.get("windows_count", 1),
                "antecedent_id": ep.get("antecedent_id", "unknown"),
                "action_offered": ep.get("action_offered", "open_observation"),
                "action_performed": req.action_performed,
                "action_custom_label": ep.get("action_custom_label", ""),
                "caregiver_decision": req.caregiver_decision,
                "performance_status": req.performance_status,
                "outcome_state": req.outcome_state,
                "settled_within_sec": req.settled_within_sec or -1,
                "child_response": req.child_response,
                "response_channel": req.response_channel,
                "response_independence": req.response_independence,
                "nccpc_instrument": ep.get("nccpc_instrument", "none"),
                "nccpc_score": ep.get("nccpc_score") if ep.get("nccpc_score") is not None else -1,
                "pain_cutoff_breached": ep.get("pain_cutoff_breached", -1),
            },
        )
        indexed = True

    return {
        "episode_id": episode_id,
        "status": "confirmed",
        "indexed_in_vector_store": indexed,
    }


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
