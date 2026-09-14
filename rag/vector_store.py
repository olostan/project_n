"""
Project N: ChromaDB Vector Store & Episodic Retrieval Wrapper.
Manages 128-dim metric episodic memory, 768-dim clinical evidence, and personal dyadic knowledge.
"""

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection


class VectorStore:
    """Encapsulates persistent ChromaDB collections with cosine metric space."""

    def __init__(self, persist_path: str | None = None) -> None:
        if persist_path:
            self.client = chromadb.PersistentClient(path=persist_path)
        else:
            self.client = chromadb.EphemeralClient()

        # 1. 128-dim Episodic memory collection
        self.episodes_coll: Collection = self.client.get_or_create_collection(
            name="nd_confirmed_episodes",
            metadata={"hnsw:space": "cosine"},
        )

        # 2. 768-dim Clinical RAG evidence collection
        self.evidence_coll: Collection = self.client.get_or_create_collection(
            name="clinical_evidence",
            metadata={"hnsw:space": "cosine"},
        )

        # 3. 768-dim Personal dyadic knowledge collection
        self.knowledge_coll: Collection = self.client.get_or_create_collection(
            name="personal_dyadic_knowledge",
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def confirmed_episodes(self) -> Collection:
        """Returns the confirmed episodes collection."""
        return self.episodes_coll

    def add_episode_embedding(
        self,
        episode_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Stores a verified 128-dim episode vector."""
        self.episodes_coll.upsert(
            ids=[episode_id],
            embeddings=[embedding],  # type: ignore[arg-type]
            metadatas=[metadata],
        )

    def query_episodes(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        encoder_version_id: str = "v1.0.0",
    ) -> dict[str, Any]:
        """Queries matching historical episodes, strictly filtering by encoder version."""
        results = self.episodes_coll.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=top_k,
            where={"encoder_version_id": encoder_version_id},
        )
        ids_raw = results.get("ids")
        dist_raw = results.get("distances")
        meta_raw = results.get("metadatas")

        ids = ids_raw[0] if ids_raw is not None and len(ids_raw) > 0 else []
        distances = dist_raw[0] if dist_raw is not None and len(dist_raw) > 0 else []
        metadatas = meta_raw[0] if meta_raw is not None and len(meta_raw) > 0 else []

        return {
            "ids": ids,
            "distances": distances,
            "metadatas": metadatas,
        }

    def count_episodes(self) -> int:
        """Returns total records in the confirmed episodes collection."""
        return self.episodes_coll.count()
