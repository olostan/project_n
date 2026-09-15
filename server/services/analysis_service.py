"""
Project N: Analysis Pipeline Orchestration Service (Phase 2).
Coordinates Tier 1 feature extraction, Tier 2 neural metric projection,
acute distress anomaly screening, episodic prototype matching, and 4-layer
output structuring.
"""

import uuid
from typing import Any

import mlx.core as mx
import numpy as np

from extraction.audio_pipeline import extract_acoustic_latent
from extraction.kinematic_pipeline import extract_kinematic_latent
from extraction.signal_quality import evaluate_signal_quality
from extraction.windowing import slice_audio_windows, slice_video_windows
from models.anomaly import AcuteDistressAnomalyDetector
from models.clip_encoder import (
    ClipSequenceAttentionPool,
    generate_sinusoidal_positional_encoding,
)
from models.contracts import (
    CONTROLLED_ACTIONS,
    METRIC_EMBEDDING_D,
)
from models.matcher import EpisodicPrototypeMatcher
from models.projection import MetricProjectionHead
from rag.vector_store import VectorStore
from server.sse_bus import SSEBus
from storage.episode_repo import EpisodeRepository
from storage.vault import VaultManager


class AnalysisService:
    """
    Asynchronous end-to-end multimodal analysis service.
    Processes audio and video streams through the frozen/projected MLX pipeline,
    evaluates clinical anomaly screeners, and produces structured 4-layer outputs.
    """

    def __init__(
        self,
        vault: VaultManager,
        episode_repo: EpisodeRepository,
        vector_store: VectorStore,
        sse_bus: SSEBus,
        projection_head: MetricProjectionHead | None = None,
        clip_encoder: ClipSequenceAttentionPool | None = None,
        prototype_matcher: EpisodicPrototypeMatcher | None = None,
    ) -> None:
        self.vault = vault
        self.episode_repo = episode_repo
        self.vector_store = vector_store
        self.sse_bus = sse_bus

        # Initialize or attach neural components
        self.projection_head = projection_head or MetricProjectionHead()
        self.clip_encoder = clip_encoder or ClipSequenceAttentionPool(d_metric=METRIC_EMBEDDING_D)
        self.prototype_matcher = prototype_matcher or EpisodicPrototypeMatcher(
            tau_abstain=0.35, is_calibrated=False
        )
        self._clip_embeddings: dict[str, list[float]] = {}

    def get_clip_embedding(self, clip_id: str) -> list[float] | None:
        """Returns cached 128-dim metric embedding if computed with a trained checkpoint."""
        return self._clip_embeddings.get(clip_id)

    def analyze_sensory_clip(
        self,
        clip_id: str,
        audio_pcm: np.ndarray,
        video_frames: list[np.ndarray],
        task_id: str | None = None,
        antecedent_id: str = "unknown",
        baseline_stats: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Executes end-to-end analysis on raw audio and video frames.

        Args:
            clip_id: Unique identifier of the encrypted clip.
            audio_pcm: 1D float32 numpy array of raw audio at 48 kHz.
            video_frames: List of RGB video frames at 30 fps.
            task_id: Optional tracking identifier for SSE events.
            antecedent_id: Observed situational antecedent context.
            baseline_stats: Child N's calibrated baseline limits.

        Returns:
            Dictionary containing 4-layer structured output and metadata.
        """
        t_id = task_id or str(uuid.uuid4())

        # 1. Slice into 5.0s contiguous windows
        audio_windows = slice_audio_windows(audio_pcm)
        video_windows = slice_video_windows(video_frames)
        windows_count = max(len(audio_windows), len(video_windows))

        self.sse_bus.publish(
            "processing_progress",
            {
                "task_id": t_id,
                "clip_id": clip_id,
                "stage": "windowing",
                "progress": 0.10,
                "windows_count": windows_count,
            },
        )

        # 2. Extract Acoustic and Kinematic Latents per window
        acoustic_latents: list[np.ndarray] = []
        kinematic_latents: list[np.ndarray] = []
        measured_f0s: list[float] = []
        measured_cpps: list[float] = []
        motion_rhythms: list[float] = []
        guarding_flags: list[bool] = []
        pose_confidences: list[float] = []
        flow_velocities: list[float] = []

        for w_idx in range(windows_count):
            a_win = (
                audio_windows[w_idx]
                if w_idx < len(audio_windows)
                else np.zeros(240000, dtype=np.float32)
            )
            v_win = (
                video_windows[w_idx]
                if w_idx < len(video_windows)
                else [np.zeros((180, 320, 3), dtype=np.uint8) for _ in range(150)]
            )

            # Audio extraction
            x_a, a_metrics = extract_acoustic_latent(a_win)
            acoustic_latents.append(x_a)
            if a_metrics.get("f0_mean_hz", 0.0) > 0:
                measured_f0s.append(float(a_metrics["f0_mean_hz"]))
            measured_cpps.append(float(a_metrics.get("cpp_db", 0.0)))

            # Kinematic extraction
            x_k, k_metrics = extract_kinematic_latent(v_win)
            kinematic_latents.append(x_k)
            motion_rhythms.append(float(k_metrics.get("motion_rhythm_hz", 0.0)))
            guarding_flags.append(bool(k_metrics.get("acute_guarding_detected", False)))
            pose_confidences.append(float(k_metrics.get("mean_pose_confidence", 1.0)))
            flow_velocities.append(float(k_metrics.get("mean_flow_velocity", 0.0)))

        self.sse_bus.publish(
            "processing_progress",
            {
                "task_id": t_id,
                "clip_id": clip_id,
                "stage": "features_extracted",
                "progress": 0.50,
            },
        )

        # 3. Neural Metric Projection & Clip-Level Aggregation
        x_audio_batch = mx.array(np.stack(acoustic_latents, axis=0))
        x_kinematic_batch = mx.array(np.stack(kinematic_latents, axis=0))

        # Project per-window representations to 128-dim metric embeddings
        z_windows = self.projection_head(x_audio_batch, x_kinematic_batch, x_physio=None)

        # Aggregate across windows with sinusoidal positional encodings
        pe = generate_sinusoidal_positional_encoding(windows_count, METRIC_EMBEDDING_D)
        z_windows_seq = mx.expand_dims(z_windows, axis=0)  # (1, W, 128)
        z_clip = self.clip_encoder(z_windows_seq, p_encodings=pe)  # (1, 128)
        mx.eval(z_clip)

        self.sse_bus.publish(
            "processing_progress",
            {
                "task_id": t_id,
                "clip_id": clip_id,
                "stage": "projected",
                "progress": 0.70,
            },
        )

        # 4. Acute Distress Anomaly Screening
        mean_f0 = float(np.mean(measured_f0s)) if measured_f0s else 0.0
        active_cpps = [c for c in measured_cpps if c > 0.0]
        mean_cpp = float(np.mean(active_cpps)) if active_cpps else 0.0
        acute_guarding = any(guarding_flags)
        active_rhythms = [r for r in motion_rhythms if r > 0.0]
        mean_rhythm = float(np.mean(active_rhythms)) if active_rhythms else 0.0

        screener_result = AcuteDistressAnomalyDetector.evaluate(
            measured_features={
                "f0_mean_hz": mean_f0,
                "cpp_db": mean_cpp,
                "acute_guarding_detected": acute_guarding,
            },
            baseline_stats=baseline_stats,
        )

        # 5. Episodic Prototype Matching via Vector Store
        if not self.projection_head.is_trained:
            self.prototype_matcher.is_calibrated = False

        confirmed_coll = self.vector_store.confirmed_episodes
        match_result = self.prototype_matcher.match(
            query_vector=z_clip, confirmed_collection=confirmed_coll
        )

        # 6. Construct 4-Layer Output Separation with Fail-Closed Quality Gates (docs/SPECS.md §4.1, §4.2)
        mean_pose_conf = float(np.mean(pose_confidences)) if pose_confidences else 0.0
        mean_flow_vel = float(np.mean(flow_velocities)) if flow_velocities else 0.0
        quality_report = evaluate_signal_quality(
            audio_pcm=audio_pcm,
            mean_pose_confidence=mean_pose_conf,
            mean_flow_velocity=mean_flow_vel,
        )

        layer1_sensory = {
            "windows_count": windows_count,
            "observed_f0_mean_hz": round(mean_f0, 2),
            "observed_cpp_db": round(mean_cpp, 2),
            "observed_motion_rhythm_hz": round(mean_rhythm, 2),
            "acute_guarding_detected": acute_guarding,
            "signal_quality": quality_report.to_dict(),
        }

        distress_triggered = screener_result.get("distress_anomaly", False)
        retrieved_candidates = match_result.get("candidates") or match_result.get("matches", [])

        if not quality_report.is_acceptable:
            self.sse_bus.publish(
                "signal_quality_abstained",
                {
                    "task_id": t_id,
                    "clip_id": clip_id,
                    "breaches": quality_report.breaches,
                },
            )
            layer2_hypotheses = {
                "status": "abstained",
                "explanation": (
                    "Sensory interpretations suppressed: fail-closed signal quality gate breach: "
                    + "; ".join(quality_report.breaches)
                ),
                "matches": [],
            }
            layer3_dyadic = {
                "suggested_actions": ["open_observation"],
                "rationale": "Sensory signal quality insufficient for reliable behavioral matching.",
            }
            # Precedence Rule (Section C3): Signal quality failure does NOT suppress safety triage,
            # but safety triage reports screener_status: 'not_assessable_low_signal_quality'
            screener_result = {
                "screener_status": "not_assessable_low_signal_quality",
                "distress_anomaly": False,
                "recommendation": (
                    "Sensory signal quality compromised. Automated anomaly screener cannot assess "
                    "distress. Please conduct direct caregiver observation and physical comfort check."
                ),
            }
            distress_triggered = False
        elif distress_triggered:
            layer2_hypotheses = {
                "status": "suppressed_due_to_anomaly",
                "explanation": (
                    "Sensory and behavioral interpretations are suppressed. "
                    "Measured bioacoustic and kinematic excursions exceed personal baseline."
                ),
                "matches": [],
            }
            layer3_dyadic = {
                "suggested_actions": ["hydration_water", "quiet_refuge", "dimmed_lighting"],
                "rationale": "Medical and physical comfort checks take priority over behavioral interventions.",
            }
        else:
            layer2_hypotheses = {
                "status": "active" if not match_result.get("abstained") else "abstained",
                "explanation": match_result.get("reason", "Retrieved historical matches"),
                "matches": retrieved_candidates,
            }
            suggested = (
                ["quiet_refuge", "sensory_break"]
                if "quiet_refuge" in CONTROLLED_ACTIONS
                else ["open_observation"]
            )
            layer3_dyadic = {
                "suggested_actions": suggested,
                "rationale": "Non-distress co-regulatory support routine.",
            }

        layer4_safety = {
            "screener_status": screener_result.get("screener_status"),
            "distress_anomaly_detected": distress_triggered,
            "screener_recommendation": screener_result.get("recommendation"),
            "clinical_disclaimer": (
                "Assistive observational notebook only; does not provide medical diagnosis, "
                "clinical decision-making, or automated prescriptive commands."
            ),
        }

        # Determine version and metric embedding emission
        if self.projection_head.is_trained and self.projection_head.checkpoint_hash:
            encoder_version = f"ckpt_{self.projection_head.checkpoint_hash[:8]}"
            emb_arr = np.array(z_clip[0], dtype=np.float32)
            metric_list: list[float] = [float(x) for x in emb_arr]
            metric_embedding: list[float] | None = metric_list
            self._clip_embeddings[clip_id] = metric_list
        else:
            encoder_version = "uncalibrated_v0"
            metric_embedding = None

        output: dict[str, Any] = {
            "episode_id": clip_id,
            "encoder_version": encoder_version,
            "layer1_sensory": layer1_sensory,
            "layer2_hypotheses": layer2_hypotheses,
            "layer3_dyadic": layer3_dyadic,
            "layer4_safety": layer4_safety,
            "metric_embedding": metric_embedding,
        }

        # 7. Persist Episode to Repository
        self.episode_repo.insert_episode(
            {
                "id": clip_id,
                "vault_uri": f"vault://{clip_id}.enc",
                "encoder_version_id": encoder_version,
                "captured_at": "2026-09-14T00:00:00Z",
                "duration_ms": windows_count * 5000,
                "windows_count": windows_count,
                "observed_f0_mean": mean_f0,
                "observed_motion_rhythm_hz": mean_rhythm,
                "antecedent_id": antecedent_id,
                "action_offered": layer3_dyadic["suggested_actions"][0],
            }
        )

        self.sse_bus.publish(
            "analysis_completed",
            {
                "task_id": t_id,
                "clip_id": clip_id,
                "progress": 1.0,
                "distress_anomaly": distress_triggered,
                "layer4_safety": layer4_safety,
            },
        )

        return output
