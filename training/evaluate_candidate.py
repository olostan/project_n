"""
Project N: Candidate Checkpoint Evaluator & Gated Promotion Pipeline.
Enforces docs/evaluation_protocol.md §5.1 criteria over verified holdout episodes.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

from storage.db_schema import init_db
from storage.episode_repo import EpisodeRepository
from training.splitter import TemporalSplitter

# Protocol §5.1 Prespecified Benchmark Targets
MIN_CONFIRMED_EPISODES: int = 10
TARGET_MRR_MIN: float = 0.65
TARGET_P_AT_3_MIN: float = 0.70
TARGET_COVERAGE_MIN: float = 0.60
TARGET_COVERAGE_MAX: float = 0.85
TARGET_ECE_MAX: float = 0.12
TARGET_MAX_DISTRESS_MISSES: int = 0
TAU_ABSTAIN: float = 0.35


def compute_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Computes cosine similarity between two normalized vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    return float(max(-1.0, min(1.0, dot)))


def compute_expected_calibration_error(
    confidences: list[float], accuracies: list[float], num_bins: int = 5
) -> float:
    """Computes binned Expected Calibration Error (ECE) over confidence scores."""
    if not confidences:
        return 0.0
    total_samples = len(confidences)
    bin_size = 1.0 / num_bins
    ece = 0.0

    for m in range(num_bins):
        bin_lower = m * bin_size
        bin_upper = (m + 1) * bin_size
        bin_indices = [
            i
            for i, c in enumerate(confidences)
            if (bin_lower <= c < bin_upper) or (m == num_bins - 1 and c == 1.0)
        ]
        if not bin_indices:
            continue
        bin_acc = sum(accuracies[i] for i in bin_indices) / len(bin_indices)
        bin_conf = sum(confidences[i] for i in bin_indices) / len(bin_indices)
        ece += (len(bin_indices) / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_b3_baseline_mrr(episodes: list[dict[str, Any]]) -> float:
    """Computes B3 Raw 1-NN Retrieval MRR over unprojected acoustic features."""
    valid_eps = [
        ep
        for ep in episodes
        if ep.get("observed_f0_mean") is not None
        and ep.get("observed_motion_rhythm_hz") is not None
    ]
    if len(valid_eps) < 2:
        return 0.0

    reciprocal_ranks: list[float] = []
    for q_idx, q in enumerate(valid_eps):
        candidates = []
        q_f0 = float(q["observed_f0_mean"])
        q_rhythm = float(q["observed_motion_rhythm_hz"])

        for c_idx, c in enumerate(valid_eps):
            if q_idx == c_idx:
                continue
            c_f0 = float(c["observed_f0_mean"])
            c_rhythm = float(c["observed_motion_rhythm_hz"])
            # Euclidean distance in normalized feature space
            dist = math.sqrt(((q_f0 - c_f0) / 100.0) ** 2 + ((q_rhythm - c_rhythm) / 2.0) ** 2)
            is_relevant = c.get("antecedent_id") == q.get("antecedent_id") and c.get(
                "outcome_state"
            ) == q.get("outcome_state")
            candidates.append((dist, is_relevant))

        candidates.sort(key=lambda x: x[0])
        first_rank = 0
        for rank_idx, (_, is_rel) in enumerate(candidates, start=1):
            if is_rel:
                first_rank = rank_idx
                break
        reciprocal_ranks.append(1.0 / first_rank if first_rank > 0 else 0.0)

    return float(sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0


def evaluate_checkpoint(
    checkpoint_path: str,
    db_conn: sqlite3.Connection,
    dataset_version: str = "ds_v1.0",
    b3_mrr: float | None = None,
) -> dict[str, Any]:
    """
    Executes formal candidate evaluation protocol against confirmed episodes in SQLite.
    Updates model_checkpoints table with real metrics and promotion status.
    """
    repo = EpisodeRepository(db=db_conn)
    all_episodes = repo.list_episodes(limit=500)
    confirmed = [ep for ep in all_episodes if ep.get("outcome_state") is not None]

    ckpt_p = Path(checkpoint_path)
    ckpt_name = ckpt_p.stem
    ckpt_id = ckpt_name if ckpt_name.startswith("ckpt_") else f"ckpt_{ckpt_name[:8]}"

    # Guard 1: Minimum sample threshold
    if len(confirmed) < MIN_CONFIRMED_EPISODES:
        status_payload: dict[str, Any] = {
            "checkpoint_id": ckpt_id,
            "evaluation_status": "pending",
            "retrieval_mrr": 0.0,
            "precision_at_3": 0.0,
            "holdout_coverage": 0.0,
            "ece_score": 0.0,
            "distress_misses": 0,
            "b3_baseline_mrr": b3_mrr or 0.0,
            "notes": (
                f"Evaluation pending: insufficient confirmed episodes (found {len(confirmed)} < {MIN_CONFIRMED_EPISODES})"
            ),
        }
        _update_checkpoint_record(
            db_conn, ckpt_id, checkpoint_path, dataset_version, status_payload
        )
        return status_payload

    # Guard 2: Temporal partition into safety holdout and evaluation queries
    splitter = TemporalSplitter(confirmed)
    regular_eps, safety_eps, manifest_hash = splitter.extract_safety_holdout()
    eval_eps = safety_eps if safety_eps else regular_eps

    # Guard 3: Metric embeddings availability
    eval_with_emb = [ep for ep in eval_eps if ep.get("metric_embedding")]
    if len(eval_with_emb) < 2:
        status_payload = {
            "checkpoint_id": ckpt_id,
            "evaluation_status": "pending",
            "retrieval_mrr": 0.0,
            "precision_at_3": 0.0,
            "holdout_coverage": 0.0,
            "ece_score": 0.0,
            "distress_misses": 0,
            "b3_baseline_mrr": b3_mrr or 0.0,
            "notes": "Evaluation pending: episodes lack metric embeddings",
        }
        _update_checkpoint_record(
            db_conn, ckpt_id, checkpoint_path, dataset_version, status_payload
        )
        return status_payload

    # Compute Retrieval Metrics over holdout queries
    reciprocal_ranks: list[float] = []
    p_at_3_scores: list[float] = []
    covered_queries: int = 0
    confidences: list[float] = []
    accuracies: list[float] = []
    distress_misses: int = 0

    for q_idx, q in enumerate(eval_with_emb):
        q_emb = q["metric_embedding"]
        is_distress = q.get("pain_cutoff_breached") == 1 or q.get("outcome_state") == "escalated"

        candidates = []
        for c_idx, c in enumerate(eval_with_emb):
            if q_idx == c_idx:
                continue
            sim = compute_cosine_similarity(q_emb, c["metric_embedding"])
            dist = 1.0 - sim
            is_relevant = c.get("antecedent_id") == q.get("antecedent_id") and c.get(
                "outcome_state"
            ) == q.get("outcome_state")
            candidates.append((dist, sim, is_relevant, c))

        candidates.sort(key=lambda x: x[0])  # Sort ascending distance
        top_dist = candidates[0][0]
        is_covered = top_dist <= TAU_ABSTAIN
        if is_covered:
            covered_queries += 1

        # Check Distress Miss
        if is_distress:
            # If abstained or top neighbor did not reflect distress escalation
            top_outcome = candidates[0][3].get("outcome_state")
            if not is_covered or top_outcome != "escalated":
                distress_misses += 1

        # Rank of first relevant neighbor
        first_rel_rank = 0
        for r_idx, (_, _, is_rel, _) in enumerate(candidates, start=1):
            if is_rel:
                first_rel_rank = r_idx
                break
        rr = 1.0 / first_rel_rank if first_rel_rank > 0 else 0.0
        reciprocal_ranks.append(rr)

        # Precision@3
        top_3 = candidates[:3]
        rel_in_top_3 = sum(1 for _, _, is_rel, _ in top_3 if is_rel)
        p_at_3_scores.append(rel_in_top_3 / len(top_3) if top_3 else 0.0)

        # Calibration pairs
        conf = max(0.0, min(1.0, 1.0 - top_dist))
        acc = 1.0 if candidates[0][2] else 0.0
        confidences.append(conf)
        accuracies.append(acc)

    mrr = float(sum(reciprocal_ranks) / len(reciprocal_ranks))
    p_at_3 = float(sum(p_at_3_scores) / len(p_at_3_scores))
    coverage = float(covered_queries / len(eval_with_emb))
    ece = compute_expected_calibration_error(confidences, accuracies)

    # Compute or retrieve B3 baseline
    measured_b3 = b3_mrr if b3_mrr is not None else compute_b3_baseline_mrr(eval_with_emb)

    # Gated Promotion Checklist (§5.1)
    checks = {
        "mrr_target": mrr >= TARGET_MRR_MIN,
        "p_at_3_target": p_at_3 >= TARGET_P_AT_3_MIN,
        "coverage_target": TARGET_COVERAGE_MIN <= coverage <= TARGET_COVERAGE_MAX,
        "ece_target": ece <= TARGET_ECE_MAX,
        "zero_distress_misses": distress_misses <= TARGET_MAX_DISTRESS_MISSES,
        "beats_b3_baseline": mrr >= measured_b3,
    }

    if all(checks.values()):
        eval_status = "passed"
        notes = "All Protocol §5.1 validation criteria met."
    else:
        failed_gates = [k for k, v in checks.items() if not v]
        eval_status = "failed"
        notes = f"Failed validation gates: {', '.join(failed_gates)}"

    report: dict[str, Any] = {
        "checkpoint_id": ckpt_id,
        "evaluation_status": eval_status,
        "retrieval_mrr": round(mrr, 4),
        "precision_at_3": round(p_at_3, 4),
        "holdout_coverage": round(coverage, 4),
        "ece_score": round(ece, 4),
        "distress_misses": distress_misses,
        "b3_baseline_mrr": round(measured_b3, 4),
        "manifest_hash": manifest_hash,
        "checks": checks,
        "notes": notes,
    }

    _update_checkpoint_record(db_conn, ckpt_id, checkpoint_path, dataset_version, report)
    return report


def _update_checkpoint_record(
    conn: sqlite3.Connection,
    ckpt_id: str,
    ckpt_path: str,
    dataset_version: str,
    report: dict[str, Any],
) -> None:
    """Updates or inserts model_checkpoints table with computed evaluation results."""
    sql = """
    INSERT INTO model_checkpoints (
        id, checkpoint_path, extractor_version, schema_version, dataset_version,
        validation_manifest_hash, retrieval_mrr, holdout_coverage, ece_score,
        evaluation_status, distress_cases_evaluated, zero_distress_misses, notes
    ) VALUES (
        ?, ?, 'v1.0.0', '1.0.0', ?, ?, ?, ?, ?, ?, ?, ?, ?
    )
    ON CONFLICT(id) DO UPDATE SET
        retrieval_mrr = excluded.retrieval_mrr,
        holdout_coverage = excluded.holdout_coverage,
        ece_score = excluded.ece_score,
        evaluation_status = excluded.evaluation_status,
        distress_cases_evaluated = excluded.distress_cases_evaluated,
        zero_distress_misses = excluded.zero_distress_misses,
        notes = excluded.notes;
    """
    with conn:
        conn.execute(
            sql,
            (
                ckpt_id,
                ckpt_path,
                dataset_version,
                report.get("manifest_hash", "manifest_default"),
                report.get("retrieval_mrr", 0.0),
                report.get("holdout_coverage", 0.0),
                report.get("ece_score", 0.0),
                report["evaluation_status"],
                report.get("distress_misses", 0),
                1 if report.get("distress_misses", 0) == 0 else 0,
                report.get("notes", ""),
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Project N Model Checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--db_path", type=str, default="project_n.db", help="SQLite database path")
    parser.add_argument("--output", type=str, default=None, help="Output path for JSON report")
    parser.add_argument("--strict", action="store_true", help="Exit code 1 if evaluation fails")
    args = parser.parse_args()

    conn = init_db(args.db_path)
    report = evaluate_checkpoint(checkpoint_path=args.checkpoint, db_conn=conn)

    print(json.dumps(report, indent=2))

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.strict and report["evaluation_status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
