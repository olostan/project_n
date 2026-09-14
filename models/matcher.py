"""
Project N: Episodic Prototype and k-NN Matcher.
Retrieval engine with calibrated abstention and 4-state onboarding lifecycle.
"""

from typing import Any

import mlx.core as mx


class EpisodicPrototypeMatcher:
    """
    Episodic prototype and k-NN retrieval engine with calibrated abstention
    and a 4-state onboarding lifecycle:
    - 'collecting': Insufficient records (<20 verified episodes logged).
    - 'awaiting_calibration': Sufficient records (>=20) but distance threshold tau_abstain uncalibrated.
    - 'active': Distance threshold calibrated on temporal holdout split; normal inference enabled.
    - 'stale': Extractor, encoder, or schema version mismatch with active checkpoint.
    """

    MIN_EPISODES_FOR_CALIBRATION: int = 20

    def __init__(
        self,
        tau_abstain: float = 0.35,
        is_calibrated: bool = False,
        active_encoder_version: str = "v1.0.0",
    ) -> None:
        self.tau_abstain = tau_abstain
        self.is_calibrated = is_calibrated
        self.active_encoder_version = active_encoder_version

    def match(
        self, query_vector: mx.array, confirmed_collection: Any, top_k: int = 3
    ) -> dict[str, Any]:
        """
        Retrieves nearest verified historical episodes with calibrated abstention.

        Args:
            query_vector: L2-normalized query vector of shape (1, 128) or (128,).
            confirmed_collection: ChromaDB collection or compatible mock object.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            Dictionary with match status, candidates, and distances.
        """
        total_records = (
            confirmed_collection.count()
            if hasattr(confirmed_collection, "count")
            else len(confirmed_collection)
        )

        # State 1: Collecting (<20 confirmed episodes)
        if total_records < self.MIN_EPISODES_FOR_CALIBRATION:
            return {
                "abstained": True,
                "matcher_state": "collecting",
                "reason": (
                    f"Cold-start onboarding in progress ({total_records}/{self.MIN_EPISODES_FOR_CALIBRATION} "
                    f"verified episodes logged). Episodic retrieval is suspended."
                ),
                "nearest_distance": 1.0,
                "candidates": [],
            }

        # State 2: Awaiting Calibration (>=20 records, but calibration split not yet computed)
        if not self.is_calibrated:
            return {
                "abstained": True,
                "matcher_state": "awaiting_calibration",
                "reason": (
                    f"Sufficient records collected ({total_records} >= {self.MIN_EPISODES_FOR_CALIBRATION}), "
                    f"but distance threshold tau_abstain is not yet calibrated on temporal split."
                ),
                "nearest_distance": 1.0,
                "candidates": [],
            }

        if query_vector.ndim == 1:
            query_vector = query_vector[None, :]

        q_list = query_vector[0].tolist()
        results = confirmed_collection.query(
            query_embeddings=[q_list],
            n_results=top_k,
            where={"encoder_version_id": self.active_encoder_version},  # Version consistency
        )

        distances = (
            results["distances"][0] if results.get("distances") and results["distances"] else [1.0]
        )
        nearest_distance = float(distances[0])

        # Calibrated abstention cutoff
        if nearest_distance > self.tau_abstain:
            return {
                "abstained": True,
                "matcher_state": "active_abstaining",
                "reason": (
                    f"Distance ({nearest_distance:.3f}) exceeds calibrated threshold "
                    f"({self.tau_abstain:.3f})"
                ),
                "nearest_distance": nearest_distance,
                "candidates": [],
            }

        candidates = []
        metadatas = (
            results["metadatas"][0] if results.get("metadatas") and results["metadatas"] else []
        )
        for meta, dist in zip(metadatas, distances, strict=False):
            candidates.append(
                {
                    "episode_id": meta.get("episode_id", "unknown_ep"),
                    "action_offered": meta.get("action_offered", "open_observation"),
                    "action_performed": meta.get("action_performed")
                    or meta.get("action_offered", "open_observation"),
                    "action_custom_label": meta.get("action_custom_label"),
                    "caregiver_decision": meta.get("caregiver_decision", "accepted"),
                    "performance_status": meta.get("performance_status", "completed"),
                    "outcome_state": meta.get("outcome_state"),
                    "settled_within_sec": meta.get("settled_within_sec"),
                    "child_response": meta.get("child_response", "none"),
                    "response_channel": meta.get("response_channel", "none"),
                    "distance": float(dist),
                }
            )

        return {
            "abstained": False,
            "matcher_state": "active",
            "nearest_distance": nearest_distance,
            "candidates": candidates,
        }
