"""
Project N: Pydantic v2 Schemas for Child Profile Facts and Therapist Techniques.
"""

from pydantic import BaseModel, Field


class FactCreateRequest(BaseModel):
    category: str = Field(
        ...,
        pattern="^(comfort_object|calming_cue|sensory_trigger|therapist_technique|communication_routine)$",
    )
    fact_title: str = Field(..., min_length=2, max_length=120)
    description: str = Field(..., min_length=5, max_length=1000)
    source_type: str = Field(..., pattern="^(home_observation|ot_session|slp_session|school)$")
    clinician_role: str | None = None
    clinician_id: str | None = None
    provenance_episode_id: str | None = None


class FactConfirmRequest(BaseModel):
    confirmed: bool = True
    custom_notes: str | None = None


class FactResponse(BaseModel):
    id: str
    category: str
    fact_title: str
    description: str
    source_type: str
    clinician_role: str | None = None
    clinician_id: str | None = None
    provenance_episode_id: str | None = None
    confirmed_by_caregiver: int
    times_tried: int
    times_helpful: int
    created_at: str | None = None


class FactListResponse(BaseModel):
    facts: list[FactResponse]
    total: int
