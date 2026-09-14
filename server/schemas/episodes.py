"""
Project N: Pydantic v2 Schemas for Episodes and Pain Observation Triage.
"""

from pydantic import BaseModel, Field


class EpisodeResponse(BaseModel):
    id: str
    vault_uri: str
    encoder_version_id: str
    captured_at: str
    duration_ms: int
    windows_count: int
    antecedent_id: str
    action_offered: str
    action_performed: str | None = None
    outcome_state: str | None = None
    settled_within_sec: int | None = None
    child_response: str | None = "none"
    nccpc_score: int | None = None
    pain_cutoff_breached: int = -1
    created_at: str | None = None


class EpisodeListResponse(BaseModel):
    episodes: list[EpisodeResponse]
    total: int


class NCCPCSubmissionRequest(BaseModel):
    instrument: str = Field(..., pattern="^(nccpc_pv|nccpc_r)$")
    scores: dict[str, int | str] = Field(..., description="Canonical item scores in [0, 3] or 'NA'")
    duration_min: int = Field(default=10, ge=1, le=120)


class NCCPCScoringResponse(BaseModel):
    episode_id: str
    instrument: str
    total_score: int
    cutoff_threshold: int
    pain_cutoff_breached: bool
    na_count: int
    subscale_scores: dict[str, int]
    escalation_required: bool
    guidance: str
