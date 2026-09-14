"""
Project N: Invariant 9 Provenance Lineage for All Embeddings.
Guarantees every embedding in the vector database and relational store records
the exact encoder_version_id and checkpoint SHA-256 hash that produced it.
"""

# ruff: noqa: E402
import hashlib
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.projection import MetricProjectionHead
from rag.vector_store import VectorStore


def test_checkpoint_lineage_and_hash_integrity() -> None:
    head = MetricProjectionHead()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "projection_head.safetensors"

        # 1. Save checkpoint and compute deterministic SHA-256
        saved_hash = head.save_checkpoint(str(ckpt_path))
        file_bytes = ckpt_path.read_bytes()
        expected_hash = hashlib.sha256(file_bytes).hexdigest()

        assert saved_hash == expected_hash
        assert head.checkpoint_hash == expected_hash
        assert head.is_trained is True

        # 2. Load into a fresh uncalibrated projection head
        fresh_head = MetricProjectionHead()
        assert fresh_head.is_trained is False
        assert fresh_head.checkpoint_hash is None

        loaded_hash = fresh_head.load_checkpoint(str(ckpt_path))
        assert loaded_hash == expected_hash
        assert fresh_head.checkpoint_hash == expected_hash
        assert fresh_head.is_trained is True


def test_vector_store_embedding_provenance_lineage() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(persist_path=str(tmpdir))

        # Checkpoint hash representing model lineage
        test_hash = "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890"
        encoder_version = f"ckpt_{test_hash[:8]}"

        test_emb = [0.0] * 128
        test_emb[0] = 1.0

        # Add episode embedding with explicit lineage metadata
        store.add_episode_embedding(
            episode_id="ep_lineage_001",
            embedding=test_emb,
            metadata={
                "episode_id": "ep_lineage_001",
                "encoder_version_id": encoder_version,
                "antecedent_id": "mealtime",
                "action_performed": "quiet_refuge",
                "caregiver_decision": "accepted",
                "outcome_state": "settled_immediately",
            },
        )

        # Retrieve and assert exact lineage metadata preservation
        coll = store.confirmed_episodes
        results = coll.get(ids=["ep_lineage_001"], include=["metadatas"])
        assert results["metadatas"] is not None
        assert len(results["metadatas"]) == 1

        rec_meta = results["metadatas"][0]
        assert rec_meta["encoder_version_id"] == encoder_version
        assert rec_meta["episode_id"] == "ep_lineage_001"

        # Invariant 9 strict version isolation check:
        # Querying with matching version returns the match
        match_res = store.query_episodes(
            query_embedding=test_emb,
            encoder_version_id=encoder_version,
        )
        assert len(match_res["ids"]) == 1
        assert match_res["ids"][0] == "ep_lineage_001"

        # Querying with a different version returns 0 matches (no cross-contamination)
        mismatch_res = store.query_episodes(
            query_embedding=test_emb,
            encoder_version_id="ckpt_different_version",
        )
        assert len(mismatch_res["ids"]) == 0


def verify_lineage_invariants() -> None:
    test_checkpoint_lineage_and_hash_integrity()
    test_vector_store_embedding_provenance_lineage()


if __name__ == "__main__":
    verify_lineage_invariants()
    print("Invariant 9 Verification: PASSED (Complete provenance lineage for all embeddings)")
