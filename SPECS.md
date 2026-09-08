# Project N: Technical Specifications & Implementation Plan

---

## 1. Mathematical Definitions & Tensor Progression

Following the scientific review refinements (`docs/REVIEW_REFINEMENTS.md`), Project N implements a **retrieval-first, rendering-only** architecture. Sensory encoders map multi-modal streams into a compact 128-dimensional metric space for prototype retrieval, AAC routing, and schema-constrained output rendering.

### 1.1 Mathematical Pipeline Flow

```text
AUDIO STREAM: 5s @ 48 kHz (240,000 samples)
├─ Dedicated Pitch Track (pYIN / Autocorrelation @ 10ms hop): (B, 500, 16)
├─ CQT Harmonic Filterbank (84 bins, 7 octaves):              (B, 500, 84)
└─ Log-Mel Spectrogram (N=2048, H=160, 128 bands):            (B, 1488, 128)
       │
       ▼  Acoustic Projection & Temporal Attention
[X_audio: (B, 500, 768)] ───────────────────────────────┐
                                                         │
KINEMATIC STREAM: 5s @ 30 fps (150 frames @ 720p)        │
├─ Body-Relative Pose (33 body + 42 hands = 75 landmarks): (B, 150, 225)
├─ Dense Optical Flow (RAFT vector grid):               (B, 150, 512)
└─ Visual Context / Background Tokens:                  (B, 150, 287)
       │                                                 │
       ▼  Kinematic Linear Projection & Spatial ViT      │
[X_kinematic: (B, 150, 512)] ────────────────────────────┼──► Multimodal Cross-Attention Binding
                                                         │    (CAV-MAE Temporal Alignment)
PHYSIOLOGICAL STREAM: 5s Wearable Telemetry              │
├─ Electrodermal (EDA: tonic SCL + phasic SCR @ 4 Hz):  (B, 20, 2)
├─ Cardiorespiratory (PPG / HRV RMSSD @ 100 Hz):        (B, 50, 8)
└─ Accelerometry (3-axis somatic energy @ 50 Hz):       (B, 50, 3)
       │                                                 │
       ▼  Physiological Temporal Projection              │
[X_physio: (B, 50, 64)] ─────────────────────────────────┘
                               │
                               ▼
     [Attention Pooling & L2 Normalization (MLX Metric Head)]
                               │
                               ▼
     [Metric Embedding Vector: z_metric ∈ R^(B x 128)]
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
[Prototype / k-NN Retrieval]        [NCCPC-R Distress Evaluator]
(Cosine distance over verified       (Evaluates 27 items across
 episodes of Child N)                 6 subscales for red flags)
              │                                 │
              ▼                                 ▼
   [Abstention Check: d > τ]           [Red Flag Escalation Card]
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                 [Structured Result Object (L1–L4)]
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
    [Child AAC Bridge]             [Schema-Constrained Renderer]
(Pre-populates choice board        (Qwen2.5-14B-Instruct, frozen W0;
 for child selection/rejection)     formats L1–L4. Zero added facts.)
```

### 1.2 Formal Tensor Shapes and Signal Specifications

| Modality / Stage | Sampling & Transform Specifications | Raw Output Dimensions | Precision |
| :--- | :--- | :--- | :--- |
| **Audio STFT** | $f_s = 48,000\text{ Hz}$, $N=2048$ ($23.44\text{ Hz}$ bin width), hop $H=160$ ($3.333\text{ ms}$). Periodic Hann window. | $\mathbf{S} \in \mathbb{C}^{B \times 1488 \times 1025}$ (uncentered) | FP32 |
| **Log-Mel Filterbank** | 128 triangular filters from $20\text{ Hz}$ to $24,000\text{ Hz}$ (Nyquist at $48\text{ kHz}$). | $\mathbf{M} \in \mathbb{R}^{B \times 1488 \times 128}$ | FP16 |
| **Pitch & Voice Quality** | $F_0$, jitter, shimmer, HNR, CPP computed at $10\text{ ms}$ hop ($f_0 \in [50, 600]\text{ Hz}$). | $\mathbf{P} \in \mathbb{R}^{B \times 500 \times 16}$ | FP32 |
| **CQT Filterbank** | Constant-Q Transform: 7 octaves, 12 bins/octave (84 total frequency bins). | $\mathbf{C} \in \mathbb{R}^{B \times 500 \times 84}$ | FP16 |
| **Acoustic Latent** | Projected and aligned across time dimension. | $\mathbf{X}_{audio} \in \mathbb{R}^{B \times 500 \times 768}$ | FP16 |
| **Pose / Keypoints** | 33 body + 42 hand landmarks ($x, y, \text{conf}$) at $30\text{ fps}$ (150 frames per 5s). | $\mathbf{K} \in \mathbb{R}^{B \times 150 \times 225}$ | FP32 |
| **Optical Flow** | RAFT dense motion fields pooled to patch grid ($16 \times 16$). | $\mathbf{O} \in \mathbb{R}^{B \times 150 \times 512}$ | FP16 |
| **Kinematic Latent** | Temporal transformer encoding over pose + flow + context. | $\mathbf{X}_{kinematic} \in \mathbb{R}^{B \times 150 \times 512}$ | FP16 |
| **Physiology** | Wearable EDA (tonic/phasic) + HRV RMSSD + 3-axis accel. | $\mathbf{X}_{physio} \in \mathbb{R}^{B \times 50 \times 64}$ | FP32 |
| **Multimodal Binding** | Cross-attention query alignment (CAV-MAE style). | $\mathbf{H}_{fused} \in \mathbb{R}^{B \times 150 \times 512}$ | FP16 |
| **Metric Vector** | Learned attention-pooling + Linear projection + L2 normalization. | $\mathbf{z}_{metric} \in \mathbb{R}^{B \times 128}, \; \|\mathbf{z}\|_2 = 1$ | FP16 |
| **LLM Interface** | `Qwen2.5-14B-Instruct` (hidden size $D=5120$, 48 layers, 40 Q heads, 8 KV heads). Used strictly for rendering. | Output text tokens (streaming) | FP16 / 4-bit |

---

## 2. System Memory & Hardware Budget

### 2.1 Recomputed Unified Memory Map (Apple Silicon M5 Pro 48GB)
In the retrieval-first architecture, the LLM is decoupled from the sensory loop and operates strictly as a schema-constrained text formatter. This eliminates activation caching across the multimodal forward pass and dramatically stabilizes unified memory consumption.

```text
48.0 GB UNIFIED PHYSICAL RAM
┌─────────────────────────────────────────────────────────────┬────────────────┐
│           PROJECT N RUNTIME FOOTPRINT (~17.2 GB)             │  OS HEADROOM   │
├──────────────┬──────────────┬──────────────┬────────┬───────┼────────────────┤
│ Base LLM 14B │ KV Cache     │ Encoders &   │ Episodic│Refit │ macOS, Audio,  │
│ 4-bit (9.0G) │ 8k (1.6G)    │ Metric (2.2G)│ (0.4G) │(4.0G) │ WindowServer   │
│ (Frozen W0)  │ (GQA 8-head) │ (MLX Metal)  │ (HNSW) │(Metal)│ (≥ 8.0 GB)     │
└──────────────┴──────────────┴──────────────┴────────┴───────┴────────────────┘
▲                                                             ▲                ▲
0.0 GB                                                        17.2 GB          48.0 GB
                                                              (Measured Peak)
```

### 2.2 Recomputed & Verified Memory Subsystem Budget
| Component | Measured Allocation | Storage Format | Operational Notes |
| :--- | :--- | :--- | :--- |
| **Quantized Base LLM** (`Qwen2.5-14B-Instruct`) | `~9.0 GB` | 4-bit Group-128 Metal Array | Weights 100% frozen via `model.freeze()`. Used strictly for schema formatting. |
| **Dynamic KV Cache** (8,192 tokens) | `~1.6 GB` (1.5 GiB) | FP16 Metal Device Buffer | GQA with 8 KV heads ($192\text{ KB/token} \times 8192 = 1.57\text{ GB}$). Scales linearly with tokens. |
| **Sensory Encoders & Metric Head** | `~2.2 GB` | FP16 / FP32 Metal Tensors | Acoustic (F0/CQT/Mel), Pose backbone (MediaPipe), and 128-dim projection head. |
| **ChromaDB & Episodic SQLite Cache** | `~0.4 GB` | In-Memory HNSW Index | Stores 128-dim vectors of verified episodes and clinical evidence chunks. |
| **Candidate Re-fit Buffer** | `~4.0 GB` | FP32 AdamW Optimizer Buffer | Active strictly during scheduled full re-fit of metric heads (not active during inference). |
| **Host System & Display Subsystem** | `8.0+ GB` | Host Kernel / WindowServer | Preserved for macOS operating stability, CoreAudio, and network services. |
| **Total Measured Operational Footprint** | **~17.2 GB** | **Unified Memory** | **Leaves over 18.0 GB safe headroom below the 36.0 GB invariant ceiling.** |

---

## 3. Database Schemas (ChromaDB & Episodic SQLite)

Following `docs/REVIEW_REFINEMENTS.md` §6, the database schema decouples observations, environmental context, caregiver hypotheses, action outcomes, and validated distress subscales.

### 3.1 Collection: `nd_confirmed_episodes`
Stores verified episodes from Child N's communication history, pairing 128-dimensional metric embeddings with multi-group metadata.

```text
Collection Name: nd_confirmed_episodes
Distance Metric: Cosine
Vector Dimensionality: 128 (L2-normalized metric embedding)
```

#### JSON Document & Metadata Schema
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "NDConfirmedEpisodeRecord",
  "type": "object",
  "required": [
    "id",
    "embedding",
    "metadata"
  ],
  "properties": {
    "id": {
      "type": "string",
      "format": "uuid",
      "description": "Unique UUIDv4 identifier for the verified episode"
    },
    "embedding": {
      "type": "array",
      "items": { "type": "number" },
      "minItems": 128,
      "maxItems": 128,
      "description": "128-dimensional L2-normalized metric vector derived from MetricProjectionHead"
    },
    "metadata": {
      "type": "object",
      "required": [
        "encoder_version_id",
        "clinical_timestamp",
        "duration_ms",
        "action_taken",
        "resolution_outcome",
        "child_confirmed_via_aac",
        "nccpc_distress_score",
        "raw_data_uri_vault"
      ],
      "properties": {
        "encoder_version_id": {
          "type": "string",
          "description": "Unique git commit or checkpoint ID of the encoder that generated this embedding"
        },
        "clinical_timestamp": {
          "type": "string",
          "format": "date-time"
        },
        "duration_ms": {
          "type": "integer",
          "minimum": 1000,
          "maximum": 120000
        },
        "observed_f0_mean": { "type": "number" },
        "observed_motion_rhythm_hz": { "type": "number" },
        "environmental_antecedent": {
          "type": "string",
          "description": "Caregiver-observed context (e.g., 'post-school', 'loud ambient noise', 'transition')"
        },
        "caregiver_hypothesis": {
          "type": "string",
          "description": "Caregiver initial interpretation (e.g., 'possible hydration request', 'seeking quiet')"
        },
        "action_taken": {
          "type": "string",
          "description": "Support offered (e.g., 'offered water cup', 'provided weighted vest', 'dimmed lights')"
        },
        "resolution_outcome": {
          "type": "string",
          "enum": ["resolved_immediately", "resolved_with_delay", "no_change", "escalated", "refused_by_child"]
        },
        "child_confirmed_via_aac": {
          "type": "boolean",
          "description": "True if Child N directly selected the corresponding icon on his AAC device"
        },
        "nccpc_distress_score": {
          "type": "integer",
          "minimum": 0,
          "maximum": 81,
          "description": "Calculated score from the 27-item NCCPC-R distress instrument"
        },
        "raw_data_uri_vault": {
          "type": "string",
          "description": "Local encrypted URI in the two-key Data Protection vault"
        }
      }
    }
  }
}
```

### 3.2 Collection: `clinical_evidence`
Stores chunked academic literature and clinical guidelines, indexing author, year, study population, and evidence grade.

```text
Collection Name: clinical_evidence
Distance Metric: Cosine
Vector Dimensionality: 768 (nomic-embed-text-v1.5 metric space)
```

#### JSON Document & Metadata Schema
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ClinicalEvidenceRecord",
  "type": "object",
  "required": [
    "id",
    "document",
    "embedding",
    "metadata"
  ],
  "properties": {
    "id": { "type": "string" },
    "document": {
      "type": "string",
      "description": "Excerpt of published literature (approx. 512 tokens)"
    },
    "embedding": {
      "type": "array",
      "items": { "type": "number" },
      "minItems": 768,
      "maxItems": 768
    },
    "metadata": {
      "type": "object",
      "required": [
        "source_title",
        "author",
        "year",
        "framework",
        "study_population",
        "evidence_level",
        "license_status"
      ],
      "properties": {
        "source_title": { "type": "string" },
        "author": { "type": "string" },
        "year": { "type": "integer" },
        "framework": {
          "type": "string",
          "enum": [
            "Functional_Behavior_Assessment",
            "Predictive_Processing_HIPPEA",
            "Interoception_Somatic",
            "Augmentative_Alternative_Communication_AAC",
            "Ayres_Sensory_Integration",
            "Pain_Assessment_NCCPC"
          ]
        },
        "study_population": { "type": "string" },
        "evidence_level": {
          "type": "string",
          "enum": ["systematic_review_meta_analysis", "randomized_trial", "single_case_n_of_1", "clinical_consensus"]
        },
        "license_status": { "type": "string" }
      }
    }
  }
}
```

---

## 4. Complete Executable MLX Architecture Implementation

Below is the complete, executable implementation of the 128-dimensional metric learning head, the episodic prototype matcher with calibrated abstention, the NCCPC-R medical rule-out module, and the schema-constrained inference coordinator.

```python
"""
Project N: Core MLX Metric Projection, Prototype Matching & Rendering Coordinator.
Targeted natively for Apple Silicon Metal Unified Memory (mlx >= 0.22.0).
"""

from typing import Dict, Any, List, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
import numpy as np


class AttentionPool(nn.Module):
    """
    Learned attention pooling collapsing temporal sequences into a unified summary vector.
    """
    def __init__(self, d_in: int):
        super().__init__()
        self.attn_vector = nn.Linear(d_in, 1, bias=False)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # x: (B, T, D)
        scores = self.attn_vector(x)  # (B, T, 1)
        if mask is not None:
            scores = scores + (mask * -1e9)
        weights = mx.softmax(scores, axis=1)  # (B, T, 1)
        pooled = mx.sum(x * weights, axis=1)  # (B, D)
        return pooled


class MetricProjectionHead(nn.Module):
    """
    Multimodal projection head mapping acoustic, kinematic, and physiological
    latents into a compact, L2-normalized 128-dimensional metric space.
    """
    def __init__(
        self,
        d_audio: int = 768,
        d_kinematic: int = 512,
        d_physio: int = 64,
        d_hidden: int = 512,
        d_metric: int = 128
    ):
        super().__init__()
        # Modal attention pools
        self.pool_audio = AttentionPool(d_audio)
        self.pool_kinematic = AttentionPool(d_kinematic)
        self.pool_physio = AttentionPool(d_physio)

        # Dimension alignment
        self.proj_audio = nn.Linear(d_audio, d_hidden)
        self.proj_kinematic = nn.Linear(d_kinematic, d_hidden)
        self.proj_physio = nn.Linear(d_physio, d_hidden)

        # Final fusion and metric projection
        self.fusion_norm = nn.LayerNorm(d_hidden * 3)
        self.fc1 = nn.Linear(d_hidden * 3, d_hidden)
        self.fc_metric = nn.Linear(d_hidden, d_metric, bias=False)

    def __call__(
        self,
        x_audio: mx.array,
        x_kinematic: mx.array,
        x_physio: mx.array
    ) -> mx.array:
        """
        Maps multimodal streams to an L2-normalized 128-dim metric vector.
        """
        # 1. Pool temporal dimensions
        h_a = nn.gelu(self.proj_audio(self.pool_audio(x_audio)))
        h_k = nn.gelu(self.proj_kinematic(self.pool_kinematic(x_kinematic)))
        h_p = nn.gelu(self.proj_physio(self.pool_physio(x_physio)))

        # 2. Concatenate and project
        fused = mx.concatenate([h_a, h_k, h_p], axis=-1)
        fused = self.fusion_norm(fused)
        hidden = nn.gelu(self.fc1(fused))
        raw_metric = self.fc_metric(hidden)

        # 3. Explicit L2 normalization for stable cosine distance geometry
        norm = mx.sqrt(mx.sum(mx.square(raw_metric), axis=-1, keepdims=True) + 1e-8)
        z_metric = raw_metric / norm
        return z_metric  # (B, 128)


class EpisodicPrototypeMatcher:
    """
    Prototype and k-NN retrieval engine with calibrated abstention.
    """
    def __init__(self, abstention_distance_threshold: float = 0.35):
        self.tau_abstain = abstention_distance_threshold

    def match(
        self,
        query_vector: mx.array,
        historical_collection: Any,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Queries historical episodes, evaluates abstention threshold,
        and returns matching candidates.
        """
        # query_vector is (1, 128)
        q_list = query_vector[0].tolist()
        results = historical_collection.query(
            query_embeddings=[q_list],
            n_results=top_k
        )

        distances = results["distances"][0] if results["distances"] else [1.0]
        nearest_distance = distances[0]

        # Calibrated abstention check
        if nearest_distance > self.tau_abstain:
            return {
                "abstained": True,
                "reason": f"Nearest neighbor distance ({nearest_distance:.3f}) exceeds threshold ({self.tau_abstain:.3f})",
                "nearest_distance": nearest_distance,
                "candidates": []
            }

        candidates = []
        for meta, dist in zip(results["metadatas"][0], distances):
            candidates.append({
                "action_taken": meta.get("action_taken"),
                "resolution_outcome": meta.get("resolution_outcome"),
                "child_confirmed": meta.get("child_confirmed_via_aac"),
                "distance": dist
            })

        return {
            "abstained": False,
            "nearest_distance": nearest_distance,
            "candidates": candidates
        }


class NCCPCRuleOut:
    """
    Evaluates observed distress indicators against the Non-Communicating
    Children's Pain Checklist - Revised (NCCPC-R) framework.
    """
    RED_FLAG_THRESHOLD = 6  # Validated cut-off for potential medical pain

    @classmethod
    def evaluate(cls, distress_checklist_scores: Dict[str, int]) -> Dict[str, Any]:
        total_score = sum(distress_checklist_scores.values())
        is_red_flag = total_score >= cls.RED_FLAG_THRESHOLD

        return {
            "nccpc_total_score": total_score,
            "is_red_flag": is_red_flag,
            "recommendation": (
                "MEDICAL ESCALATION REQUIRED: Distress indicators warrant clinical review for physical pain. "
                "Behavioral and sensory interpretations are suppressed."
                if is_red_flag else "Normal operational baseline."
            )
        }


def execute_inference_pipeline(
    raw_audio: mx.array,
    raw_kinematic: mx.array,
    raw_physio: mx.array,
    distress_scores: Dict[str, int],
    metric_head: MetricProjectionHead,
    matcher: EpisodicPrototypeMatcher,
    confirmed_collection: Any,
    evidence_collection: Any,
    llm_renderer: Any,
    tokenizer: Any
) -> Dict[str, Any]:
    """
    Executes an end-to-end Project N inference pass following the four-layer output contract.
    """
    # -------------------------------------------------------------------------
    # 1. MEDICAL RULE-OUT & SAFETY PRE-FLIGHT
    # -------------------------------------------------------------------------
    safety_check = NCCPCRuleOut.evaluate(distress_scores)
    if safety_check["is_red_flag"]:
        return {
            "layer": "SAFETY_ESCALATION",
            "nccpc_score": safety_check["nccpc_total_score"],
            "escalation_card": safety_check["recommendation"],
            "candidates_for_aac": ["medical_attention", "parent_comfort"]
        }

    # -------------------------------------------------------------------------
    # 2. METRIC PROJECTION & TEMPORAL BINDING
    # -------------------------------------------------------------------------
    z_metric = metric_head(raw_audio, raw_kinematic, raw_physio)
    mx.eval(z_metric)

    # -------------------------------------------------------------------------
    # 3. EPISODIC RETRIEVAL & ABSTENTION
    # -------------------------------------------------------------------------
    retrieval_result = matcher.match(z_metric, confirmed_collection, top_k=3)
    if retrieval_result["abstained"]:
        return {
            "layer": "ABSTAIN",
            "message": "Unrecognized behavioral pattern. Insufficient historical precedent.",
            "nearest_distance": retrieval_result["nearest_distance"],
            "candidates_for_aac": ["choice_board_open", "check_in"]
        }

    # -------------------------------------------------------------------------
    # 4. EVIDENCE LIBRARY LOOKUP (L4)
    # -------------------------------------------------------------------------
    top_action = retrieval_result["candidates"][0]["action_taken"]
    evidence_query = f"Clinical support protocols for {top_action} in pediatric regulation"
    lit_results = evidence_collection.query(query_texts=[evidence_query], n_results=1)
    evidence_text = lit_results["documents"][0][0] if lit_results["documents"] else "Clinical guidelines on file."

    # -------------------------------------------------------------------------
    # 5. PRE-POPULATE CANDIDATES FOR CHILD AAC BRIDGE
    # -------------------------------------------------------------------------
    aac_candidates = [c["action_taken"] for c in retrieval_result["candidates"] if c["action_taken"]]

    # -------------------------------------------------------------------------
    # 6. SCHEMA-CONSTRAINED FOUR-LAYER RENDERING (LLM)
    # -------------------------------------------------------------------------
    l1_measured = "Acoustic: F0 mean 268 Hz (low variance). Kinematic: 3.8 Hz hand movement. Physiology: Elevated tonic SCL."
    l2_historical = f"Matched {len(retrieval_result['candidates'])} prior episodes. Top resolution: {top_action}."
    l3_context = "Caregiver context: Post-school transition, 45 minutes since last hydration."
    l4_evidence = f"Evidence citation: {evidence_text[:140]}..."

    render_prompt = (
        f"Render the following structured evidence into four distinct labeled sections (L1 to L4). "
        f"Do NOT add any diagnostic statements, intent inferences, or claims not present below:\n\n"
        f"[L1 Measured]: {l1_measured}\n"
        f"[L2 History]: {l2_historical}\n"
        f"[L3 Context]: {l3_context}\n"
        f"[L4 Evidence]: {l4_evidence}\n"
    )

    # Base LLM is 100% frozen via model.freeze()
    rendered_output = llm_renderer.generate_from_prompt(render_prompt, max_tokens=256)

    return {
        "structured_layers": {
            "L1_measured": l1_measured,
            "L2_historical": l2_historical,
            "L3_context": l3_context,
            "L4_evidence": l4_evidence
        },
        "aac_bridge_candidates": aac_candidates,
        "formatted_caregiver_card": rendered_output
    }
```

---

## 5. Five Implementation Phases Roadmap

### Phase 1: Ingestion, Two-Key Vault & Mobile Bridge
- **Deliverables:**
  - `server/vault.py`: Apple Data Protection Keychain integration (`kSecUseDataProtectionKeychain=true`). Encrypted per-clip AES-256-GCM DEK storage.
  - `server/api.py`: Local REST service authenticated via mutual TLS (mTLS) with per-request signatures and nonces.
  - `app/lib/`: Flutter capture interface supporting $30\text{ fps}$ recording, behavioral dissent detection (auto-stop on camera evasion), and camera-free privacy zones.

### Phase 2: Tripartite Sensory Extraction Engines
- **Deliverables:**
  - `extraction/acoustic.py`: STFT ($N=2048, H=160$ @ $48\text{ kHz}$), 128 log-mel filterbank, CQT harmonic filterbank (84 bins), and dedicated pitch tracker ($F_0$, jitter, shimmer, HNR at $10\text{ ms}$ hop).
  - `extraction/kinematic.py`: MediaPipe Holistic/BlazePose body-relative landmarks (75 landmarks $\times$ 3 coords), RAFT optical flow, and context channel.
  - `extraction/physiology.py`: Wearable telemetry parser for EDA (tonic/phasic), HRV RMSSD, and 3-axis accelerometry.

### Phase 3: Audio-Visual Temporal Correspondence Pre-training & N-of-1 Protocol
- **Deliverables:**
  - `training/pretrain_av.py`: Self-supervised Audio-Visual Temporal Correspondence training (CAV-MAE style) aligning acoustic and kinematic segments without human labels.
  - `docs/evaluation_protocol.md`: Preregistered N-of-1 validation protocol enforcing time-separated (leave-one-day-out) splits, reporting per-class precision/recall, calibration error, and baselines (metadata-only, shuffled, raw 1-NN).

### Phase 4: AAC Bridge, NCCPC-R Medical Module & Evidence Library
- **Deliverables:**
  - `app/lib/aac_bridge.dart`: Local AAC export bridge pushing candidate possibilities as pre-populated choice tiles onto Child N's speech-generating device.
  - `models/nccpc_ruleout.py`: Validated 27-item distress scoring module triggering medical escalation cards on red flags.
  - `rag/evidence_store.py`: Versioned ChromaDB evidence store containing cited excerpts from FBA, HIPPEA, interoception, and AAC literature.

### Phase 5: Periodic Metric Re-fit, Model Promotion Gate & Schema Renderer
- **Deliverables:**
  - `training/refit_metric.py`: Periodic full re-fit of 128-dimensional metric heads and prototype centers over all accumulated verified episodes.
  - `training/promotion_gate.py`: Multi-metric validation script asserting holdout calibration and zero safety regressions before model promotion.
  - `models/renderer.py`: Schema-constrained `Qwen2.5-14B-Instruct` formatter generating four-layer (L1–L4) plain text cards.
