"""
Project N: Local Authentication and Companion Pairing Routes.
"""

import secrets
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from server.deps import _valid_tokens

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


@router.post("/pair", response_model=PairingResponse)
def pair_companion_device(req: PairingRequest) -> dict[str, Any]:
    """
    PIN pairing endpoint for mobile companions.
    Authenticates PIN and issues a local client token and mock signed certificate.
    """
    # Default pairing PIN for setup
    if req.pairing_pin != "123456":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid pairing PIN.",
        )

    # Generate paired bearer token
    token = f"paired_{secrets.token_hex(16)}"
    _valid_tokens.add(token)

    return {
        "client_cert_pem": "-----BEGIN CERTIFICATE-----\nMIIC...MOCK_CERT...\n-----END CERTIFICATE-----",
        "token": token,
        "expires_at": "2027-09-14T00:00:00Z",
    }
