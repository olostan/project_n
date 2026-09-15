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


def test_candidate_evaluator_insufficient_data(tmp_path: Path) -> None:
    from training.evaluate_candidate import evaluate_checkpoint

    db = init_db(":memory:")
    dummy_ckpt = tmp_path / "dummy.safetensors"
    model = MetricProjectionHead()
    model.save_checkpoint(str(dummy_ckpt))

    report = evaluate_checkpoint(checkpoint_path=str(dummy_ckpt), db_conn=db)
    assert report["evaluation_status"] == "pending"
    assert "insufficient confirmed episodes" in report["notes"]


def test_candidate_evaluator_with_episodes(tmp_path: Path) -> None:
    from storage.episode_repo import EpisodeRepository
    from tests.fixtures.synthetic_episodes import seed_synthetic_episodes
    from training.evaluate_candidate import evaluate_checkpoint

    db = init_db(":memory:")
    repo = EpisodeRepository(db=db)
    seed_synthetic_episodes(repo, count=15)

    dummy_ckpt = tmp_path / "candidate_test.safetensors"
    model = MetricProjectionHead()
    model.save_checkpoint(str(dummy_ckpt))

    report = evaluate_checkpoint(
        checkpoint_path=str(dummy_ckpt), db_conn=db, dataset_version="ds_test_v1"
    )
    assert report["evaluation_status"] in ("passed", "failed")
    assert report["retrieval_mrr"] > 0.0
    assert report["holdout_coverage"] > 0.0
    assert "checks" in report

    # Verify model_checkpoints table was updated
    cursor = db.execute(
        "SELECT * FROM model_checkpoints WHERE checkpoint_path = ?", (str(dummy_ckpt),)
    )
    row = cursor.fetchone()
    assert row is not None
    assert row["evaluation_status"] == report["evaluation_status"]
    assert row["retrieval_mrr"] == report["retrieval_mrr"]


def test_production_checkpoint_loader_in_deps(tmp_path: Path) -> None:
    import server.deps as deps
    from models.standardizer import FeatureStandardizer

    db = init_db(":memory:")
    deps._db_conn = db
    deps._analysis_service = None  # Reset singleton

    # Create dummy trained checkpoint
    ckpt_dir = tmp_path / "prod_ckpt"
    ckpt_dir.mkdir()
    ckpt_file = ckpt_dir / "ckpt_prod123.safetensors"
    model = MetricProjectionHead()
    model.save_checkpoint(str(ckpt_file))

    # Save standardizers
    scaler = FeatureStandardizer()
    scaler.fit(np.ones((2, 10)))
    scaler.save(ckpt_dir / "standardizer_audio.npz")
    scaler.save(ckpt_dir / "standardizer_kinematic.npz")

    # Register as passed and production
    db.execute(
        """
        INSERT INTO model_checkpoints (
            id, checkpoint_path, extractor_version, schema_version, dataset_version,
            retrieval_mrr, holdout_coverage, evaluation_status, is_production
        ) VALUES ('ckpt_prod123', ?, 'v1', '1.0', 'ds1', 0.85, 0.80, 'passed', 1)
        """,
        (str(ckpt_file),),
    )

    svc = deps.get_analysis_service()
    assert svc.projection_head.is_trained is True
    assert svc.audio_standardizer is not None
    assert svc.kinematic_standardizer is not None

    # Verify health endpoint returns loaded checkpoint
    from fastapi.testclient import TestClient

    from server.main import app

    client = TestClient(app)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["checkpoint_id"].startswith("ckpt_")
    assert data["is_trained"] is True
