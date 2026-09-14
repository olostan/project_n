"""
Project N: FastAPI Dependency Injection Providers.
"""

import sqlite3

from fastapi import Header, HTTPException, status

from rag.vector_store import VectorStore
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
_valid_tokens: set[str] = {"mock_local_paired_token", "dev_token"}


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
    """Verifies local paired Bearer token."""
    if not authorization or not authorization.startswith("Bearer "):
        # For development / initial pairing, allow dev tokens or raise 401
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Bearer authentication token.",
        )
    token = authorization.split("Bearer ")[1].strip()
    if token not in _valid_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unrecognized local paired client token.",
        )
    return token
