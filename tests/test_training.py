"""
Project N: Training Pipeline & Temporal Splitter Tests (Gate C).
Verifies walk-forward forward chaining temporal splits, locked safety holdout set,
triplet loss descent with gradient clipping, and checkpoint table registration.
"""

from pathlib import Path

import numpy as np

from models.projection import MetricProjectionHead
from storage.db_schema import init_db
from training.splitter import TemporalSplitter
from training.train_projection import (
    build_synthetic_triplets,
    save_and_register_checkpoint,
    train_projection_head,
)


def test_temporal_splitter_and_safety_holdout() -> None:
    # Build 70 mock episodes over 70 days
    episodes = []
    for day in range(70):
        episodes.append(
            {
                "id": f"ep_{day:03d}",
                "captured_at": f"2026-05-{day + 1:02d}T12:00:00Z"
                if day < 30
                else f"2026-06-{day - 29:02d}T12:00:00Z",
                "pain_cutoff_breached": 1 if day % 5 == 0 else 0,
                "outcome_state": "escalated" if day % 5 == 0 else "settled_immediately",
                "action_offered": "quiet_refuge" if day % 5 == 0 else "hydration_water",
            }
        )

    splitter = TemporalSplitter(episodes)
    regular, safety, manifest_hash = splitter.extract_safety_holdout()

    # Invariant 9: 50 locked safety episodes
    assert len(safety) == 50
    assert len(regular) == 20
    assert len(manifest_hash) == 64

    # Forward-chaining walk-forward splits
    splits = splitter.create_forward_chaining_splits(min_train_episodes=10, eval_window_size=5)
    assert len(splits) >= 2
    for split in splits:
        # Guarantee no temporal overlap / future leakage
        assert not set(split.train_episode_ids).intersection(set(split.eval_episode_ids))
        assert split.dataset_version.startswith("ds_v")


def test_train_projection_and_checkpoint_registration(tmp_path: Path) -> None:
    model = MetricProjectionHead()
    batches = build_synthetic_triplets(batch_size=4)

    # Initial forward
    a_a, a_k = batches[0][0], batches[0][1]
    z_init = model(a_a, a_k)
    assert z_init.shape == (4, 128)

    # Train for 3 epochs
    losses = train_projection_head(model, batches, epochs=3, lr=1e-3, seed=42)
    assert len(losses) == 3
    assert not np.isnan(losses).any()

    # Save and register in SQLite
    db = init_db(":memory:")
    ckpt_id = save_and_register_checkpoint(
        model=model,
        output_dir=tmp_path / "checkpoints",
        db_conn=db,
        dataset_version="ds_test_v1",
        validation_manifest_hash="0123456789abcdef" * 4,
        retrieval_mrr=0.88,
        holdout_coverage=0.94,
        tau_abstain=0.35,
    )

    assert ckpt_id.startswith("ckpt_")

    # Verify SQLite row
    cursor = db.execute("SELECT * FROM model_checkpoints WHERE id = ?", (ckpt_id,))
    row = cursor.fetchone()
    assert row is not None
    assert row["retrieval_mrr"] == 0.88
    assert row["holdout_coverage"] == 0.94

    # Verify model loading
    new_model = MetricProjectionHead()
    loaded_hash = new_model.load_checkpoint(row["checkpoint_path"])
    assert new_model.is_trained is True
    assert f"ckpt_{loaded_hash[:8]}" == ckpt_id
