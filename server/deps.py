"""
Project N: FastAPI Dependency Injection Providers.
"""

import os
import secrets
import sqlite3
import time

from fastapi import Header, HTTPException, status

from rag.vector_store import VectorStore
from server.ca import LocalCertificateAuthority
from server.services.analysis_service import AnalysisService
from server.sse_bus import SSEBus
from storage.db_schema import init_db
from storage.episode_repo import EpisodeRepository
from storage.facts_repo import FactsRepository
from storage.vault import MockKeychainProvider, VaultManager

# Global singletons for server lifecycle
_db_conn: sqlite3.Connection | None = None
_vault_manager: VaultManager | None = None
_vector_store: VectorStore | None = None
_analysis_service: AnalysisService | None = None
_sse_bus: SSEBus = SSEBus()

# Token registry: token -> expiration epoch timestamp
_issued_tokens: dict[str, float] = {}

# Ephemeral pairing PIN
_active_pairing_pin: str | None = None
_pairing_pin_expires_at: float = 0.0


def get_or_create_pairing_pin(ttl_seconds: int = 300) -> str:
    """Returns the current valid pairing PIN or generates an ephemeral 6-digit PIN."""
    global _active_pairing_pin, _pairing_pin_expires_at
    env_pin = os.getenv("PROJECT_N_PAIRING_PIN")
    if env_pin:
        return env_pin
    now = time.time()
    if _active_pairing_pin is None or now >= _pairing_pin_expires_at:
        _active_pairing_pin = f"{secrets.randbelow(1_000_000):06d}"
        _pairing_pin_expires_at = now + ttl_seconds
    return _active_pairing_pin


def register_token(token: str, expires_at_epoch: float) -> None:
    """Registers an authenticated paired token with an expiration timestamp."""
    _issued_tokens[token] = expires_at_epoch


def revoke_token(token: str) -> None:
    """Revokes a paired token."""
    _issued_tokens.pop(token, None)


def is_token_valid(token: str) -> bool:
    """Validates token presence and expiration."""
    exp = _issued_tokens.get(token)
    if exp is None:
        return False
    if time.time() >= exp:
        _issued_tokens.pop(token, None)
        return False
    return True


def get_db() -> sqlite3.Connection:
    global _db_conn
    if _db_conn is None:
        _db_conn = init_db(":memory:")
    return _db_conn


def get_episode_repo() -> EpisodeRepository:
    return EpisodeRepository(db=get_db())


def get_facts_repo() -> FactsRepository:
    return FactsRepository(db=get_db())


def get_vault() -> VaultManager:
    global _vault_manager
    if _vault_manager is None:
        _vault_manager = VaultManager(keychain=MockKeychainProvider())
    return _vault_manager


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(persist_path=None)
    return _vector_store


def get_sse_bus() -> SSEBus:
    return _sse_bus


def get_ca() -> LocalCertificateAuthority:
    return LocalCertificateAuthority.get_instance()


def get_analysis_service() -> AnalysisService:
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = AnalysisService(
            vault=get_vault(),
            episode_repo=get_episode_repo(),
            vector_store=get_vector_store(),
            sse_bus=get_sse_bus(),
        )
    return _analysis_service


def verify_auth_token(authorization: str | None = Header(default=None)) -> str:
    """Verifies local paired Bearer token and enforces expiration."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Bearer authentication token.",
        )
    token = authorization.split("Bearer ")[1].strip()
    if not is_token_valid(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unrecognized or expired local paired client token.",
        )
    return token
