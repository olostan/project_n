"""
Project N: Facts and Therapist Techniques Routes.
"""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from server.deps import get_facts_repo, verify_auth_token
from server.schemas.facts import (
    FactConfirmRequest,
    FactCreateRequest,
    FactListResponse,
    FactResponse,
)
from storage.facts_repo import FactsRepository

router = APIRouter(prefix="/api/v1/facts", tags=["facts"])


@router.get("", response_model=FactListResponse)
def list_facts(
    category: str | None = None,
    source_type: str | None = None,
    confirmed_only: bool = Query(default=True),
    repo: FactsRepository = Depends(get_facts_repo),
    _token: str = Depends(verify_auth_token),
) -> dict[str, Any]:
    """Lists personal profile facts and clinician-suggested techniques."""
    facts = repo.list_facts(
        category=category, source_type=source_type, confirmed_only=confirmed_only
    )
    return {"facts": facts, "total": len(facts)}


@router.post("", response_model=FactResponse, status_code=status.HTTP_201_CREATED)
def create_fact(
    req: FactCreateRequest,
    repo: FactsRepository = Depends(get_facts_repo),
    _token: str = Depends(verify_auth_token),
) -> dict[str, Any]:
    """
    Stages an extracted clinician technique or personal fact.
    Initializes confirmed_by_caregiver = 0 (Fail-closed confirmation gate).
    """
    fact_id = str(uuid.uuid4())
    data = req.model_dump()
    data["id"] = fact_id

    repo.create_fact(data)
    # Fetch staged fact
    staged = repo.list_facts(confirmed_only=False)
    created = next((f for f in staged if f["id"] == fact_id), None)
    if not created:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to stage fact."
        )
    return created


@router.post("/{fact_id}/confirm", response_model=dict[str, Any])
def confirm_fact(
    fact_id: str,
    req: FactConfirmRequest,
    repo: FactsRepository = Depends(get_facts_repo),
    _token: str = Depends(verify_auth_token),
) -> dict[str, Any]:
    """Caregiver confirms or rejects a staged clinician technique."""
    success = repo.confirm_fact(fact_id=fact_id, confirmed=req.confirmed)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Fact not found.")
    return {"fact_id": fact_id, "confirmed_by_caregiver": req.confirmed}
