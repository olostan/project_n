"""
Project N: Local Authentication and Companion Pairing Routes.
"""

import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from server.ca import LocalCertificateAuthority
from server.deps import get_ca, get_or_create_pairing_pin, register_token

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class PairingRequest(BaseModel):
    device_id: str = Field(..., min_length=3, max_length=64)
    csr_pem: str = Field(
        ..., description="Certificate Signing Request from client hardware Keystore"
    )
    pairing_pin: str = Field(..., min_length=4, max_length=8)


class PairingResponse(BaseModel):
    client_cert_pem: str
    token: str
    expires_at: str


class PairingPinResponse(BaseModel):
    pairing_pin: str
    expires_in_sec: int


@router.get("/pin", response_model=PairingPinResponse)
def get_current_pairing_pin() -> dict[str, Any]:
    """Retrieves current ephemeral pairing PIN for local dashboard/display."""
    pin = get_or_create_pairing_pin()
    return {"pairing_pin": pin, "expires_in_sec": 300}


@router.post("/pair", response_model=PairingResponse)
def pair_companion_device(
    req: PairingRequest,
    ca: LocalCertificateAuthority = Depends(get_ca),
) -> dict[str, Any]:
    """
    PIN pairing endpoint for mobile companions.
    Authenticates ephemeral PIN and issues a local client token and signed X.509 certificate.
    """
    active_pin = get_or_create_pairing_pin()
    if req.pairing_pin != active_pin:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired pairing PIN.",
        )

    try:
        client_cert = ca.sign_csr(req.csr_pem, validity_days=365)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Certificate Signing Request validation failed: {err}",
        ) from err

    token = f"paired_{secrets.token_hex(16)}"
    expires_dt = datetime.now(UTC) + timedelta(days=365)
    register_token(token, expires_dt.timestamp())

    return {
        "client_cert_pem": client_cert,
        "token": token,
        "expires_at": expires_dt.isoformat(),
    }
