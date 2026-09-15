"""
Project N: ChromaDB Vector Store & Episodic Retrieval Wrapper.
Manages 128-dim metric episodic memory, 768-dim clinical evidence, and personal dyadic knowledge.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection


def compute_offline_text_embedding(text: str, dim: int = 768) -> list[float]:
    """Generates deterministic 768-dim pseudo-lexical embedding without network downloads."""
    vec = np.zeros(dim, dtype=np.float32)
    words = re.findall(r"\w+", text.lower())
    if not words:
        return [0.0] * dim
    for w in words:
        h = int(hashlib.sha256(w.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 16) & 1 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return [float(x) for x in vec]


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

        # Automatically seed evidence library if empty
        if self.evidence_coll.count() == 0:
            self.seed_evidence_library()

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

    def seed_evidence_library(self, seed_path: str | Path | None = None) -> int:
        """Loads and indexes verified clinical literature citations from seed_corpus.json."""
        if seed_path is None:
            seed_path = Path(__file__).parent / "evidence" / "seed_corpus.json"
        p = Path(seed_path)
        if not p.exists():
            return 0

        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return 0

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        documents: list[str] = []

        for item in data:
            doc_id = item["id"]
            title = item.get("title", "")
            excerpt = item.get("excerpt", "")
            keywords = " ".join(item.get("keywords", []))
            full_text = f"{title} {excerpt} {keywords}"

            emb = compute_offline_text_embedding(full_text)
            ids.append(doc_id)
            embeddings.append(emb)
            documents.append(excerpt)
            metadatas.append(
                {
                    "doi": item.get("doi", ""),
                    "authors": item.get("authors", ""),
                    "year": int(item.get("year", 0)),
                    "title": title,
                    "journal": item.get("journal", ""),
                    "evidence_level": item.get("evidence_level", ""),
                }
            )

        self.evidence_coll.upsert(
            ids=ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=metadatas,  # type: ignore[arg-type]
            documents=documents,
        )
        return len(ids)

    def retrieve_evidence(
        self,
        query: str,
        top_k: int = 2,
        max_distance: float = 0.85,
    ) -> list[dict[str, Any]]:
        """Retrieves relevant clinical literature evidence matching query."""
        if self.evidence_coll.count() == 0:
            return []

        q_emb = compute_offline_text_embedding(query)
        res = self.evidence_coll.query(
            query_embeddings=[q_emb],  # type: ignore[arg-type]
            n_results=min(top_k, self.evidence_coll.count()),
        )

        ids_raw = res.get("ids")
        dist_raw = res.get("distances")
        meta_raw = res.get("metadatas")
        doc_raw = res.get("documents")

        if not ids_raw or not ids_raw[0]:
            return []

        hits: list[dict[str, Any]] = []
        for i in range(len(ids_raw[0])):
            dist = dist_raw[0][i] if dist_raw and len(dist_raw[0]) > i else 1.0
            if dist > max_distance:
                continue
            meta = meta_raw[0][i] if meta_raw and len(meta_raw[0]) > i else {}
            doc = doc_raw[0][i] if doc_raw and len(doc_raw[0]) > i else ""
            hits.append(
                {
                    "id": ids_raw[0][i],
                    "doi": meta.get("doi", ""),
                    "authors": meta.get("authors", ""),
                    "year": meta.get("year", 0),
                    "title": meta.get("title", ""),
                    "journal": meta.get("journal", ""),
                    "evidence_level": meta.get("evidence_level", ""),
                    "excerpt": doc,
                    "distance": round(float(dist), 3),
                }
            )
        return hits

    def count_episodes(self) -> int:
        """Returns total records in the confirmed episodes collection."""
        return self.episodes_coll.count()
