"""
Project N: Forward-Chaining Temporal Splits & Safety Holdout Manager.
Implements temporal train/validation splitting (train Days 1..T-1, test Day T)
and manages the locked 50-episode safety holdout set per docs/DESIGN.md §7.2.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass
class TemporalSplit:
    """A single forward-chaining temporal train/validation split."""

    split_index: int
    train_episode_ids: list[str]
    eval_episode_ids: list[str]
    split_date: str
    dataset_version: str


class TemporalSplitter:
    """
    Partitions verified episodes temporally to prevent future information leakage,
    while isolating an immutable 50-episode safety holdout set.
    """

    LOCKED_SAFETY_SET_SIZE: int = 50

    def __init__(self, episodes: list[dict[str, Any]]) -> None:
        # Sort chronologically by captured_at
        self.episodes = sorted(
            episodes, key=lambda ep: str(ep.get("captured_at", "1970-01-01T00:00:00Z"))
        )

    def extract_safety_holdout(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        """
        Extracts up to 50 verified safety episodes for the locked evaluation holdout.
        Returns:
            tuple (remaining_episodes, safety_holdout, validation_manifest_hash)
        """
        safety_episodes: list[dict[str, Any]] = []
        regular_episodes: list[dict[str, Any]] = []

        # Prioritize acute distress or pain-evaluated episodes for safety set
        for ep in self.episodes:
            if len(safety_episodes) < self.LOCKED_SAFETY_SET_SIZE and (
                ep.get("pain_cutoff_breached") == 1
                or ep.get("outcome_state") == "escalated"
                or ep.get("action_offered") == "quiet_refuge"
            ):
                safety_episodes.append(ep)
            else:
                regular_episodes.append(ep)

        # If not enough specific distress episodes, fill up to 50 from earliest
        while len(safety_episodes) < min(self.LOCKED_SAFETY_SET_SIZE, len(self.episodes)):
            if regular_episodes:
                safety_episodes.append(regular_episodes.pop(0))
            else:
                break

        # Compute deterministic SHA-256 hash of locked holdout IDs
        manifest_payload = json.dumps(
            sorted([ep["id"] for ep in safety_episodes]), sort_keys=True
        ).encode("utf-8")
        manifest_hash = hashlib.sha256(manifest_payload).hexdigest()

        return regular_episodes, safety_episodes, manifest_hash

    def create_forward_chaining_splits(
        self, min_train_episodes: int = 20, eval_window_size: int = 5
    ) -> list[TemporalSplit]:
        """
        Creates forward-chaining walk-forward temporal splits:
        Fold k: Train Days 1..k, Evaluate on Day k+1.
        """
        regular_episodes, _, _ = self.extract_safety_holdout()
        total = len(regular_episodes)
        if total < min_train_episodes + eval_window_size:
            return []

        splits: list[TemporalSplit] = []

        for split_idx, train_end in enumerate(range(min_train_episodes, total, eval_window_size)):
            eval_end = min(train_end + eval_window_size, total)
            train_eps = [ep["id"] for ep in regular_episodes[:train_end]]
            eval_eps = [ep["id"] for ep in regular_episodes[train_end:eval_end]]
            if not eval_eps:
                break

            split_date = str(regular_episodes[train_end].get("captured_at", "unknown"))
            # Deterministic version hash of train set
            split_hash = hashlib.sha256(
                json.dumps(train_eps, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]

            splits.append(
                TemporalSplit(
                    split_index=split_idx,
                    train_episode_ids=train_eps,
                    eval_episode_ids=eval_eps,
                    split_date=split_date,
                    dataset_version=f"ds_v{split_idx}_{split_hash}",
                )
            )

        return splits
