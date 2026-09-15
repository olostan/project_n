"""
Project N: FastAPI Dependency Injection Providers.
"""

import contextlib
import os
import secrets
import sqlite3
import sys
import time
from pathlib import Path

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

# Ephemeral pairing PIN & Rate Limiting
_active_pairing_pin: str | None = None
_pairing_pin_expires_at: float = 0.0
_failed_pairing_attempts: int = 0
_MAX_PAIRING_ATTEMPTS: int = 5


def get_or_create_pairing_pin(ttl_seconds: int = 300) -> str:
    """Returns the current valid pairing PIN or generates an ephemeral 6-digit PIN."""
    global _active_pairing_pin, _pairing_pin_expires_at
    if os.getenv("PROJECT_N_TEST_MODE") == "1":
        env_pin = os.getenv("PROJECT_N_PAIRING_PIN")
        if env_pin:
            return env_pin

    now = time.time()
    if _active_pairing_pin is None or now >= _pairing_pin_expires_at:
        _active_pairing_pin = f"{secrets.randbelow(1_000_000):06d}"
        _pairing_pin_expires_at = now + ttl_seconds

        # Write to ~/.project_n/pairing.pin with 0600 permissions
        try:
            pn_dir = Path.home() / ".project_n"
            pn_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(pn_dir, 0o700)
            pin_file = pn_dir / "pairing.pin"
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            fd = os.open(pin_file, flags, 0o600)
            with open(fd, "w", encoding="utf-8") as f:
                f.write(_active_pairing_pin)
            with contextlib.suppress(OSError):
                os.chmod(pin_file, 0o600)
        except OSError as err:
            sys.stderr.write(f"[WARN] Failed to write pairing PIN file: {err}\n")

        # Emit to stderr for local host operator
        sys.stderr.write(f"[SECURITY] Ephemeral pairing PIN: {_active_pairing_pin}\n")
        sys.stderr.flush()

    return _active_pairing_pin


def check_and_record_pairing_attempt(candidate_pin: str) -> bool:
    """
    Validates candidate PIN with lockout after 5 failed attempts.
    Raises HTTPException 429 if locked out.
    Returns True if match, False if mismatch.
    """
    global _failed_pairing_attempts
    if _failed_pairing_attempts >= _MAX_PAIRING_ATTEMPTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed pairing attempts. Pairing locked out.",
        )
    active_pin = get_or_create_pairing_pin()
    if candidate_pin != active_pin:
        _failed_pairing_attempts += 1
        return False
    _failed_pairing_attempts = 0
    return True


def reset_pairing_state() -> None:
    """Resets pairing PIN and failure counter for testing purposes."""
    global _failed_pairing_attempts, _active_pairing_pin, _pairing_pin_expires_at
    _failed_pairing_attempts = 0
    _active_pairing_pin = None
    _pairing_pin_expires_at = 0.0


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
