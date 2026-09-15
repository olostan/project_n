"""
Project N: Apple MLX Metric Projection Head Training Pipeline.
Trains the 128-dimensional metric projection head using triplet margin loss,
gradient clipping, forward-chaining splits, and registers checkpoint metadata.
"""

import argparse
import sqlite3
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np

from models.contracts import ACOUSTIC_LATENT_D, KINEMATIC_LATENT_D
from models.projection import MetricProjectionHead
from storage.db_schema import init_db
from training.splitter import TemporalSplitter


def triplet_margin_loss(
    anchor: mx.array, positive: mx.array, negative: mx.array, margin: float = 0.2
) -> mx.array:
    """Computes cosine/L2 triplet margin loss: max(0, ||a - p||_2 - ||a - n||_2 + margin)."""
    d_pos = mx.sqrt(mx.sum(mx.square(anchor - positive), axis=-1) + 1e-8)
    d_neg = mx.sqrt(mx.sum(mx.square(anchor - negative), axis=-1) + 1e-8)
    loss = mx.maximum(0.0, d_pos - d_neg + margin)
    return mx.mean(loss)


def clip_gradients_l2(grads: Any, max_norm: float = 1.0) -> Any:
    """Clips gradients by global L2 norm: ||g||_2 <= max_norm."""

    def _collect_norms(g: Any) -> list[mx.array]:
        norms: list[mx.array] = []
        if isinstance(g, dict):
            for v in g.values():
                norms.extend(_collect_norms(v))
        elif isinstance(g, list):
            for v in g:
                norms.extend(_collect_norms(v))
        elif isinstance(g, mx.array):
            norms.append(mx.sum(mx.square(g)))
        return norms

    sq_norms = _collect_norms(grads)
    if not sq_norms:
        return grads
    total_norm = mx.sqrt(mx.sum(mx.stack(sq_norms)) + 1e-8)
    scale = mx.minimum(1.0, max_norm / total_norm)

    def _apply_scale(g: Any) -> Any:
        if isinstance(g, dict):
            return {k: _apply_scale(v) for k, v in g.items()}
        elif isinstance(g, list):
            return [_apply_scale(v) for v in g]
        elif isinstance(g, mx.array):
            return g * scale
        return g

    return _apply_scale(grads)


def train_projection_head(
    model: MetricProjectionHead,
    train_batches: list[tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]],
    epochs: int = 5,
    lr: float = 1e-4,
    seed: int = 42,
) -> list[float]:
    """
    Trains projection head on triplet batches with gradient clipping.
    Each batch: (a_audio, a_kin, p_audio, p_kin, n_audio, n_kin).
    """
    mx.random.seed(seed)
    optimizer = opt.AdamW(learning_rate=lr, weight_decay=1e-2)

    def loss_fn(
        m: MetricProjectionHead,
        a_a: mx.array,
        a_k: mx.array,
        p_a: mx.array,
        p_k: mx.array,
        n_a: mx.array,
        n_k: mx.array,
    ) -> mx.array:
        z_a = m(a_a, a_k)
        z_p = m(p_a, p_k)
        z_n = m(n_a, n_k)
        return triplet_margin_loss(z_a, z_p, z_n, margin=0.2)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss_history: list[float] = []

    for _ in range(epochs):
        epoch_losses: list[float] = []
        for batch in train_batches:
            a_a, a_k, p_a, p_k, n_a, n_k = batch
            loss, grads = loss_and_grad_fn(model, a_a, a_k, p_a, p_k, n_a, n_k)
            grads = clip_gradients_l2(grads, max_norm=1.0)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters())
            epoch_losses.append(float(loss.item()))
        if epoch_losses:
            loss_history.append(float(np.mean(epoch_losses)))

    return loss_history


def save_and_register_checkpoint(
    model: MetricProjectionHead,
    output_dir: Path,
    db_conn: sqlite3.Connection,
    dataset_version: str,
    validation_manifest_hash: str,
    retrieval_mrr: float = 0.0,
    holdout_coverage: float = 0.0,
    ece_score: float | None = None,
    evaluation_status: str = "pending",
    tau_abstain: float = 0.35,
    extractor_version: str = "git_v1.0.0",
    schema_version: str = "1.0.0",
    notes: str = "Pending evaluation by evaluate_candidate.py",
) -> str:
    """Saves checkpoint weights and registers entry in model_checkpoints table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_ckpt = output_dir / "temp_eval.safetensors"
    ckpt_hash = model.save_checkpoint(str(temp_ckpt))
    ckpt_id = f"ckpt_{ckpt_hash[:8]}"

    final_path = output_dir / f"{ckpt_id}.safetensors"
    if temp_ckpt.exists():
        temp_ckpt.rename(final_path)

    sql = """
    INSERT OR REPLACE INTO model_checkpoints (
        id, checkpoint_path, extractor_version, schema_version, dataset_version,
        validation_manifest_hash, retrieval_mrr, holdout_coverage, ece_score,
        evaluation_status, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    full_notes = f"{notes} (tau_abstain={tau_abstain})" if "tau_abstain" not in notes else notes
    with db_conn:
        db_conn.execute(
            sql,
            (
                ckpt_id,
                str(final_path),
                extractor_version,
                schema_version,
                dataset_version,
                validation_manifest_hash,
                retrieval_mrr,
                holdout_coverage,
                ece_score,
                evaluation_status,
                full_notes,
            ),
        )

    return ckpt_id


def build_synthetic_triplets(
    batch_size: int = 4, seq_len: int = 5
) -> list[tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]]:
    """Builds synthetic feature triplets for testing and verification."""
    from tests.fixtures.synthetic_episodes import generate_mock_feature_triplets

    return generate_mock_feature_triplets(
        batch_size=batch_size, t_a=seq_len, t_k=seq_len, num_batches=3
    )


def mine_triplets_from_episodes(
    episodes: list[dict[str, Any]],
    batch_size: int = 4,
    seq_len: int = 5,
) -> list[tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]]:
    """
    Mines (anchor, positive, negative) triplet batches from real confirmed episodes.
    Positive pair: matching antecedent and matching outcome state.
    Negative pair: different antecedent or conflicting outcome state.
    """
    valid_eps = [ep for ep in episodes if ep.get("outcome_state") is not None]
    if len(valid_eps) < 10:
        return []

    batches = []
    # Build batches by grouping positive and negative pairs
    for i in range(0, len(valid_eps) - 2, batch_size):
        a_a_list, a_k_list = [], []
        p_a_list, p_k_list = [], []
        n_a_list, n_k_list = [], []

        for j in range(i, min(i + batch_size, len(valid_eps))):
            anchor = valid_eps[j]
            # Find positive candidate: same antecedent, same outcome
            positives = [
                ep
                for k, ep in enumerate(valid_eps)
                if k != j
                and ep.get("antecedent_id") == anchor.get("antecedent_id")
                and ep.get("outcome_state") == anchor.get("outcome_state")
            ]
            # Find negative candidate: different antecedent or escalated vs settled
            negatives = [
                ep
                for k, ep in enumerate(valid_eps)
                if k != j
                and (
                    ep.get("antecedent_id") != anchor.get("antecedent_id")
                    or ep.get("outcome_state") != anchor.get("outcome_state")
                )
            ]

            if not positives or not negatives:
                continue

            # Deterministic feature generation from confirmed observation stats
            def _make_feat(ep: dict[str, Any]) -> tuple[mx.array, mx.array]:
                f0 = float(ep.get("observed_f0_mean") or 260.0)
                rhythm = float(ep.get("observed_motion_rhythm_hz") or 1.0)
                audio_f = mx.ones((seq_len, ACOUSTIC_LATENT_D)) * (f0 / 500.0)
                kin_f = mx.ones((seq_len, KINEMATIC_LATENT_D)) * (rhythm / 3.0)
                return audio_f, kin_f

            a_a, a_k = _make_feat(anchor)
            p_a, p_k = _make_feat(positives[0])
            n_a, n_k = _make_feat(negatives[0])

            a_a_list.append(a_a)
            a_k_list.append(a_k)
            p_a_list.append(p_a)
            p_k_list.append(p_k)
            n_a_list.append(n_a)
            n_k_list.append(n_k)

        if a_a_list:
            batches.append(
                (
                    mx.stack(a_a_list, axis=0),
                    mx.stack(a_k_list, axis=0),
                    mx.stack(p_a_list, axis=0),
                    mx.stack(p_k_list, axis=0),
                    mx.stack(n_a_list, axis=0),
                    mx.stack(n_k_list, axis=0),
                )
            )

    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Project N Metric Projection Head")
    parser.add_argument("--model_size", type=str, default="14b")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--vram_limit", type=float, default=36.0)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="models/checkpoints")
    parser.add_argument("--db_path", type=str, default="project_n.db")
    args = parser.parse_args()

    # Verify device
    if args.device == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        mx.set_default_device(mx.gpu)

    db_conn = init_db(args.db_path)
    from storage.episode_repo import EpisodeRepository

    repo = EpisodeRepository(db=db_conn)
    confirmed = [ep for ep in repo.list_episodes(limit=500) if ep.get("outcome_state")]

    if len(confirmed) < 10:
        print(
            f"Insufficient confirmed episodes with recorded outcomes in database (found {len(confirmed)} < 10). "
            "At least 10 confirmed episodes required for metric projection fine-tuning."
        )
        return

    splitter = TemporalSplitter(confirmed)
    regular_eps, safety_eps, manifest_hash = splitter.extract_safety_holdout()

    batches = mine_triplets_from_episodes(regular_eps, batch_size=args.batch_size)
    if not batches:
        print("Could not mine sufficient valid triplet pairs from regular training split.")
        return

    # Fit FeatureStandardizer on training split features only
    from models.standardizer import FeatureStandardizer

    all_audio = mx.concatenate([b[0] for b in batches], axis=0)
    all_kin = mx.concatenate([b[1] for b in batches], axis=0)
    audio_scaler = FeatureStandardizer().fit(all_audio)
    kin_scaler = FeatureStandardizer().fit(all_kin)

    # Standardize batches
    std_batches = []
    for a_a, a_k, p_a, p_k, n_a, n_k in batches:
        std_batches.append(
            (
                audio_scaler.transform(a_a),
                kin_scaler.transform(a_k),
                audio_scaler.transform(p_a),
                kin_scaler.transform(p_k),
                audio_scaler.transform(n_a),
                kin_scaler.transform(n_k),
            )
        )

    model = MetricProjectionHead()
    start_time = time.time()
    losses = train_projection_head(model, std_batches, epochs=args.epochs)
    duration = time.time() - start_time

    out_p = Path(args.output_dir)
    ckpt_id = save_and_register_checkpoint(
        model=model,
        output_dir=out_p,
        db_conn=db_conn,
        dataset_version="ds_v1.0",
        validation_manifest_hash=manifest_hash,
        evaluation_status="pending",
        notes="Trained on confirmed episodes; pending evaluate_candidate.py validation.",
    )

    # Save standardizers alongside checkpoint
    audio_scaler.save(out_p / "standardizer_audio.npz")
    kin_scaler.save(out_p / "standardizer_kinematic.npz")

    print(
        f"Training completed in {duration:.2f}s. Checkpoint {ckpt_id} saved as pending. "
        f"Final loss: {losses[-1] if losses else 0.0:.4f}"
    )

    # Automatically evaluate candidate checkpoint against holdout episodes
    from training.evaluate_candidate import evaluate_checkpoint

    final_ckpt_path = out_p / f"{ckpt_id}.safetensors"
    if final_ckpt_path.exists():
        eval_report = evaluate_checkpoint(
            checkpoint_path=str(final_ckpt_path),
            db_conn=db_conn,
            dataset_version="ds_v1.0",
        )
        print(f"Candidate evaluation status: {eval_report['evaluation_status']}")


if __name__ == "__main__":
    main()
