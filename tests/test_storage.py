"""
Project N: Unit Tests for Tier 3 Storage & Vault Layer.
Verifies AES-256-GCM envelope encryption, SQLite DDL with CHECK constraints,
Episode and Facts repositories, and ChromaDB vector retrieval.
"""

import sqlite3
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidTag

from rag.vector_store import VectorStore
from storage.db_schema import init_db
from storage.episode_repo import EpisodeRepository
from storage.facts_repo import FactsRepository
from storage.vault import MockKeychainProvider, VaultManager


def test_vault_encryption_decryption(tmp_path: Path) -> None:
    keychain = MockKeychainProvider()
    vault = VaultManager(keychain=keychain)

    clip_id = "test_clip_001"
    raw_video = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00synthetic_payload"

    # Encrypt
    ciphertext, wrapped_dek, nonce = vault.encrypt_clip(raw_video, clip_id=clip_id)
    assert ciphertext != raw_video
    assert len(nonce) == 12

    # Decrypt
    decrypted = vault.decrypt_clip(ciphertext, wrapped_dek, nonce, clip_id=clip_id)
    assert decrypted == raw_video

    # Tampered clip_id should raise error (AAD verification failure)
    with pytest.raises(InvalidTag):
        vault.decrypt_clip(ciphertext, wrapped_dek, nonce, clip_id="wrong_clip_id")

    # Cryptographic erasure test
    test_file = tmp_path / "clip.enc"
    test_file.write_bytes(ciphertext)
    assert test_file.exists()
    VaultManager.erase_clip(test_file)
    assert not test_file.exists()


def test_sqlite_episodes_crud_and_check_constraints() -> None:
    conn = init_db(":memory:")
    repo = EpisodeRepository(db=conn)

    # 1. Valid insertion
    ep_data = {
        "id": "ep_test_100",
        "vault_uri": "vault://clips/ep_test_100.enc",
        "encoder_version_id": "v1.0.0",
        "captured_at": "2026-09-14T12:00:00Z",
        "duration_ms": 60000,
        "windows_count": 12,
        "antecedent_id": "post_school_transition",
        "action_offered": "deep_pressure_proprioceptive",
    }
    ep_id = repo.insert_episode(ep_data)
    assert ep_id == "ep_test_100"

    retrieved = repo.get_episode("ep_test_100")
    assert retrieved is not None
    assert retrieved["action_offered"] == "deep_pressure_proprioceptive"

    # 2. Update outcome
    updated = repo.update_outcome(
        "ep_test_100",
        action_performed="deep_pressure_proprioceptive",
        outcome_state="settled_immediately",
        settled_within_sec=120,
        child_response="reach",
    )
    assert updated
    retrieved_after = repo.get_episode("ep_test_100")
    assert retrieved_after is not None
    assert retrieved_after["outcome_state"] == "settled_immediately"
    assert retrieved_after["settled_within_sec"] == 120

    # 3. Violation of controlled vocabulary CHECK constraint
    invalid_ep = {
        "id": "ep_invalid",
        "vault_uri": "vault://clips/invalid.enc",
        "encoder_version_id": "v1.0.0",
        "captured_at": "2026-09-14T12:00:00Z",
        "duration_ms": 60000,
        "windows_count": 12,
        "antecedent_id": "INVALID_UNDEFINED_ANTECEDENT",  # CHECK constraint should fail
        "action_offered": "deep_pressure_proprioceptive",
    }
    with pytest.raises(sqlite3.IntegrityError):
        repo.insert_episode(invalid_ep)


def test_facts_confirmation_gate() -> None:
    conn = init_db(":memory:")
    repo = FactsRepository(db=conn)

    # 1. Stage a clinical technique (defaults to confirmed_by_caregiver = 0)
    fact_id = repo.create_fact(
        {
            "id": "fact_01",
            "category": "therapist_technique",
            "fact_title": "Forearm joint compression",
            "description": "Gentle firm joint compression on forearms for 2 minutes",
            "source_type": "ot_session",
            "clinician_role": "OT",
            "clinician_id": "clinician_01",
        }
    )

    # 2. Confirmed only filter should exclude unconfirmed staged technique
    confirmed_facts = repo.list_facts(confirmed_only=True)
    assert len(confirmed_facts) == 0

    all_staged = repo.list_facts(confirmed_only=False)
    assert len(all_staged) == 1
    assert all_staged[0]["confirmed_by_caregiver"] == 0

    # 3. Human-in-the-Loop Confirmation Gate
    repo.confirm_fact(fact_id, confirmed=True)
    confirmed_after = repo.list_facts(confirmed_only=True)
    assert len(confirmed_after) == 1
    assert confirmed_after[0]["confirmed_by_caregiver"] == 1

    # 4. Usage tracking (times_tried, times_helpful)
    repo.increment_usage(fact_id, helpful=True)
    fact_updated = repo.list_facts(confirmed_only=True)[0]
    assert fact_updated["times_tried"] == 1
    assert fact_updated["times_helpful"] == 1


def test_chroma_vector_store() -> None:
    store = VectorStore(persist_path=None)  # Ephemeral in-memory

    # Add 128-dim vector
    test_vec = [0.1] * 128
    meta = {
        "episode_id": "ep_chroma_1",
        "encoder_version_id": "v1.0.0",
        "action_offered": "quiet_refuge",
        "outcome_state": "settled_immediately",
    }
    store.add_episode_embedding("ep_chroma_1", test_vec, meta)
    assert store.count_episodes() == 1

    # Query with matching version
    res = store.query_episodes(test_vec, top_k=1, encoder_version_id="v1.0.0")
    assert len(res["ids"]) == 1
    assert res["ids"][0] == "ep_chroma_1"

    # Query with non-matching version should yield 0 results (lineage invariant)
    res_stale = store.query_episodes(test_vec, top_k=1, encoder_version_id="v2.0.0")
    assert len(res_stale["ids"]) == 0


def test_lineage_verification_dispatch() -> None:
    from tests.verify_lineage import verify_lineage_invariants

    verify_lineage_invariants()
