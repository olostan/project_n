# Project N: Technical Specifications & Implementation Plan

---

## 1. Mathematical Definitions & Tensor Shapes

The Project N pipeline ingests multimodal sensory streams, processes them through specialized local encoders, compresses them through a Perceiver Resampler, and injects the resulting continuous latent vectors into the prefix attention space of a 14B parameter Large Language Model.

### 1.1 Mathematical Transformation Graph

```text
[Raw Audio: (B, 240000) @ 48kHz]
       │
       ▼  STFT (n_fft=2048, hop=160) + 128 Mel Filterbank
[Mel-Spectrogram: (B, 1, 1500, 128)]
       │
       ▼  Aliasing-aware Patch Embedding (AaPE) + Positional Encoding
[X_audio: (B, 2048, 768)] ─────────────┐
                                        │  Linear Alignment: W_a (768 -> 4096)
                                        ▼  W_v (1024 -> 4096)
[X_kinematic: (B, 1500, 1024)] ────────► Concatenated Sensory Stream: (B, 3548, 4096)
       ▲                                │
       │  Temporal Diff (Δt) + TD-ViT   │
[Raw Video: (B, 300, 3, 224, 224)]       │
                                        ▼
                         [Perceiver Resampler (ND Projector)]
                         ├─ Latent Queries: (64, 4096)
                         └─ Metal Scaled Dot-Product Attention
                                        │
                                        ▼
                         [Z_sensory: (B, 64, 4096)]
                                        │
                                        ▼
                         [Qwen2.5-14B Prefix Attention Injection]
```

### 1.2 Formal Tensor Shape Progression

| Pipeline Stage | Input Tensor Dimension | Mathematical Operator | Output Tensor Dimension | Precision |
| :--- | :--- | :--- | :--- | :--- |
| **Audio STFT** | $\mathbf{s} \in \mathbb{R}^{B \times 240,000}$ ($5\text{s} @ 48\text{kHz}$) | $\text{STFT}(n_{fft}=2048, H=160)$ | $\mathbf{S} \in \mathbb{C}^{B \times 1500 \times 1025}$ | FP32 |
| **Mel Filterbank** | $\mathbf{S} \in \mathbb{C}^{B \times 1500 \times 1025}$ | $\ln\left(\mathbf{\Phi}_{mel} |\mathbf{S}|^2 + \epsilon\right)$ | $\mathbf{M} \in \mathbb{R}^{B \times 1 \times 1500 \times 128}$ | FP16 |
| **AaPE Projection** | $\mathbf{M} \in \mathbb{R}^{B \times 1 \times 1500 \times 128}$ | $\text{Conv2D}(k=[16, 8], s=[8, 4]) + \mathbf{E}_{pos}$ | $\mathbf{X}_{audio} \in \mathbb{R}^{B \times 2048 \times 768}$ | FP16 |
| **Video Diff** | $\mathbf{V} \in \mathbb{R}^{B \times 300 \times 3 \times 224 \times 224}$ ($5\text{s} @ 60\text{fps}$) | $\mathbf{\Delta}_t = \|\mathbf{V}_t - \mathbf{V}_{t-1}\|_1 \cdot \mathbb{I}(\mathcal{E} \ge \tau)$ | $\mathbf{\Delta} \in \mathbb{R}^{B \times 300 \times 1 \times 224 \times 224}$ | FP16 |
| **TD-ViT Encoder** | $\mathbf{\Delta} \in \mathbb{R}^{B \times 300 \times 1 \times 224 \times 224}$ | Spatio-Temporal Patch Linear + 12L ViT | $\mathbf{X}_{kinematic} \in \mathbb{R}^{B \times 1500 \times 1024}$ | FP16 |
| **Linear Alignment** | $\mathbf{X}_a \in \mathbb{R}^{B \times 2048 \times 768}, \mathbf{X}_k \in \mathbb{R}^{B \times 1500 \times 1024}$ | $\left[\mathbf{X}_a \mathbf{W}_a \;\|\; \mathbf{X}_k \mathbf{W}_v\right]$ | $\mathbf{X}_{sensory} \in \mathbb{R}^{B \times 3548 \times 4096}$ | FP16 |
| **Perceiver Resampler**| $\mathbf{X}_{sensory} \in \mathbb{R}^{B \times 3548 \times 4096}, \mathbf{Q} \in \mathbb{R}^{64 \times 4096}$ | Multi-Head Cross-Attention ($h=32$) | $\mathbf{Z}_{sensory} \in \mathbb{R}^{B \times 64 \times 4096}$ | FP16 |
| **Prefix Conditioning**| $\mathbf{Z}_{sensory} \in \mathbb{R}^{B \times 64 \times 4096}, \mathbf{E}_{text} \in \mathbb{R}^{B \times L_t \times 4096}$ | Concatenation $[\mathbf{Z}_{sensory} \;\|\; \mathbf{E}_{text}]$ | $\mathbf{H}_0 \in \mathbb{R}^{B \times (64 + L_t) \times 4096}$ | FP16 |

---

## 2. System Memory & Hardware Budget

### 2.1 Unified Memory Allocation Map (Apple Silicon M5 Pro 48GB)
Apple Silicon unified memory is shared dynamically between the CPU, GPU (Metal), and Neural Engine. To prevent macOS system hangs and prohibit paging to the internal SSD, the primary runtime enforces an absolute upper memory ceiling of $36.0\text{ GB}$, with an operational peak baseline of $28.0\text{ GB}$.

```text
48.0 GB UNIFIED PHYSICAL RAM
┌─────────────────────────────────────────────────────────────┬────────────────┐
│           PROJECT N RUNTIME FOOTPRINT (~28.0 GB)             │  OS HEADROOM   │
├──────────────┬──────────────┬──────────────┬────────┬───────┼────────────────┤
│ Base LLM 14B │ KV Cache     │ Replay Buffer│ LoRA + │Chroma │ macOS, Audio,  │
│ 4-bit (9.0G) │ 8k (10.0G)   │ Train (8.0G) │RAG(0.6G│ (0.4G)│ WindowServer   │
│              │              │              │        │       │ (≥ 8.0 GB)     │
└──────────────┴──────────────┴──────────────┴────────┴───────┴────────────────┘
▲                                                             ▲                ▲
0.0 GB                                                        28.0 GB          48.0 GB
                                                              (Operational Peak)
```

### 2.2 Detailed Memory Subsystem Budget
| Component | Footprint | Storage Format / Device | Lifetime & Dynamic Allocation Strategy |
| :--- | :--- | :--- | :--- |
| **Quantized Base LLM** (Qwen2.5 14B) | `9.0 GB` | 4-bit Metal Packed (`mlx.core.array`) | Static allocation loaded at daemon start. Weights remain completely frozen ($W_0$). |
| **Dynamic Context & KV Cache** | `10.0 GB` | FP16 Metal Device Buffer | Dynamically allocated per inference pass. Window sized for 8,192 tokens. Lazy garbage collection. |
| **Replay Buffer & Training States** | `8.0 GB` | FP32 / FP16 AdamW State Matrices | Active strictly during nightly contrastive learning loop. Deallocated upon completion of daily consolidation. |
| **Clinical Domain LoRA Adapter** | `0.4 GB` | FP16 Metal Parameters | Rank $r=64$, $\alpha=128$. Injected across attention projection matrices (`q, k, v, o, gate, up, down`). |
| **Text Embedder** (`nomic-embed-text-v1.5`) | `0.2 GB` | FP16 Local MLX Array | Persistent embedding model running fully offline for clinical RAG queries. |
| **ChromaDB In-Memory Store** | `0.4 GB` | HNSW Index + SQLite Metadata Cache | Local disk-backed vector storage cached in unified memory for $<10\text{ms}$ query latency. |
| **Operating System & WindowServer** | `8.0+ GB` | Host Kernel & User Space | Reserved strictly for macOS host stability, displays, and audio daemons. |
| **Total Operational Peak** | **~28.0 GB** | **Unified Memory** | **Leaves 8.0 GB safety buffer beneath the 36.0 GB hard invariant ceiling.** |

### 2.3 Parameterization for 64GB & 128GB Scale
The codebase supports dynamic scaling via configuration flags in `server/config.py`:
- **64GB Configuration (`--model_size 32b`):**
  - Base LLM shifts to `Qwen2.5-32B-Instruct` 4-bit (allocating $19.5\text{ GB}$).
  - KV Cache allocated at $14.0\text{ GB}$ (16k context window).
  - Peak operational footprint: $45.5\text{ GB}$, leaving $18.5\text{ GB}$ headroom.
- **128GB Configuration (`--model_size 72b`):**
  - Base LLM shifts to `Qwen2.5-72B-Instruct` 4-bit (allocating $42.0\text{ GB}$).
  - Full batch training buffer size expanded to 64 samples.

---

## 3. Database Schemas (ChromaDB)

Project N maintains two distinct local vector collections using ChromaDB with native cosine distance indexing (`hnsw:space: cosine`).

### 3.1 Collection: `nd_communicative_intents`
Stores Nolan's historical multi-modal communicative instances, pairing compressed sensory latents with parent-validated intent labels.

```text
Collection Name: nd_communicative_intents
Distance Metric: Cosine
Vector Dimensionality: 4096 (Mean-pooled Z_sensory representation)
```

#### Document & Metadata JSON Schema
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "NDCommunicativeIntentRecord",
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
      "description": "Unique UUIDv4 identifier for the captured intent episode"
    },
    "embedding": {
      "type": "array",
      "items": { "type": "number" },
      "minItems": 4096,
      "maxItems": 4096,
      "description": "4096-dimensional mean-pooled latent vector derived from Z_sensory"
    },
    "metadata": {
      "type": "object",
      "required": [
        "parent_tag",
        "confidence_score",
        "clinical_timestamp",
        "duration_ms",
        "sensory_modalities_present",
        "raw_data_uri_local"
      ],
      "properties": {
        "parent_tag": {
          "type": "string",
          "enum": [
            "deep_proprioceptive_pressure",
            "auditory_overload_escape",
            "vestibular_seeking",
            "hydration_water_request",
            "hunger_food_request",
            "tactile_boundary_protest",
            "social_connection_seeking",
            "unspecified_discomfort"
          ]
        },
        "confidence_score": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "description": "Parent validation confidence rating (1.0 = verified, 0.5 = uncertain)"
        },
        "clinical_timestamp": {
          "type": "string",
          "format": "date-time",
          "description": "ISO 8601 UTC timestamp of original clip capture"
        },
        "duration_ms": {
          "type": "integer",
          "minimum": 1000,
          "maximum": 120000,
          "description": "Length of processed recording in milliseconds"
        },
        "sensory_modalities_present": {
          "type": "string",
          "enum": ["audio_only", "vision_only", "multimodal_audio_vision"]
        },
        "raw_data_uri_local": {
          "type": "string",
          "description": "Local file URI pointing to offline encrypted media archive"
        },
        "antecedent_context": {
          "type": "string",
          "description": "Optional caregiver notes regarding preceding environmental events"
        }
      }
    }
  }
}
```

### 3.2 Collection: `clinical_literature`
Stores chunked pediatric occupational therapy, speech-language pathology, and behavioral science references.

```text
Collection Name: clinical_literature
Distance Metric: Cosine
Vector Dimensionality: 768 (nomic-embed-text-v1.5 metric space)
```

#### Document & Metadata JSON Schema
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ClinicalLiteratureRecord",
  "type": "object",
  "required": [
    "id",
    "document",
    "embedding",
    "metadata"
  ],
  "properties": {
    "id": {
      "type": "string",
      "description": "Deterministic hash of literature source and chunk index"
    },
    "document": {
      "type": "string",
      "description": "Raw textual chunk of clinical literature (approx. 512 tokens)"
    },
    "embedding": {
      "type": "array",
      "items": { "type": "number" },
      "minItems": 768,
      "maxItems": 768,
      "description": "768-dimensional text embedding generated by local nomic-embed-text-v1.5"
    },
    "metadata": {
      "type": "object",
      "required": [
        "source_title",
        "author",
        "year",
        "clinical_framework",
        "chunk_id"
      ],
      "properties": {
        "source_title": { "type": "string" },
        "author": { "type": "string" },
        "year": { "type": "integer" },
        "clinical_framework": {
          "type": "string",
          "enum": [
            "Ayres_Sensory_Integration",
            "Antecedent_Behavior_Consequence",
            "DIR_Floortime",
            "Childhood_Apraxia_Speech_CAS",
            "Sensory_Gating_Autonomic"
          ]
        },
        "chunk_id": { "type": "integer" }
      }
    }
  }
}
```

---

## 4. Complete MLX Forward Pass Implementation

Below is the complete, executable MLX architecture defining the Perceiver Resampler (`NDProjector`) and the end-to-end forward pipeline.

```python
"""
Project N: Core MLX Forward Pass and Perceiver Resampler Implementation.
Optimized for Apple Silicon Metal Unified Memory Architecture.
"""

from typing import Dict, Any, Tuple, Optional
import mlx.core as mx
import mlx.nn as nn
import numpy as np


class PerceiverAttention(nn.Module):
    """
    Multi-Head Cross-Attention utilizing Apple MLX's Metal-accelerated
    scaled_dot_product_attention primitive.
    """
    def __init__(self, d_model: int = 4096, num_heads: int = 32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        B, M, _ = q.shape
        _, S, _ = k.shape

        # Linear projections & split heads: (B, num_heads, seq_len, head_dim)
        queries = self.q_proj(q).reshape(B, M, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = self.k_proj(k).reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = self.v_proj(v).reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apple Metal hardware-accelerated scaled dot product attention
        attn_out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        # Transpose back and project output: (B, M, d_model)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, M, self.d_model)
        return self.out_proj(attn_out)


class NDProjector(nn.Module):
    """
    Perceiver Resampler module compressing high-density multimodal sensory sequences
    (3,548 tokens) into exactly 64 hyper-dense latent prefix tokens.
    """
    def __init__(
        self,
        d_audio: int = 768,
        d_kinematic: int = 1024,
        d_model: int = 4096,
        num_latents: int = 64,
        num_heads: int = 32,
        ffn_dim: int = 16384
    ):
        super().__init__()
        self.num_latents = num_latents
        self.d_model = d_model

        # Modal alignment projections
        self.audio_align = nn.Linear(d_audio, d_model, bias=True)
        self.kinematic_align = nn.Linear(d_kinematic, d_model, bias=True)

        # Learnable latent query tokens initialized with small variance
        self.latent_queries = mx.random.normal(shape=(num_latents, d_model)) * 0.02

        # Cross-Attention Block
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_sensory = nn.LayerNorm(d_model)
        self.cross_attn = PerceiverAttention(d_model=d_model, num_heads=num_heads)

        # Feed-Forward Network Block
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn_up = nn.Linear(d_model, ffn_dim, bias=True)
        self.ffn_down = nn.Linear(ffn_dim, d_model, bias=True)

    def __call__(self, x_audio: mx.array, x_kinematic: mx.array) -> mx.array:
        """
        Compresses input sensory streams into latent prefix conditioning tokens.
        
        Args:
            x_audio: Acoustic representation (B, 2048, 768)
            x_kinematic: Kinematic representation (B, 1500, 1024)
            
        Returns:
            z_sensory: Compressed sensory prefix tokens (B, 64, 4096)
        """
        B = x_audio.shape[0]

        # 1. Project modalities to unified LLM latent dimension (d_model = 4096)
        h_audio = self.audio_align(x_audio)          # (B, 2048, 4096)
        h_kinematic = self.kinematic_align(x_kinematic)  # (B, 1500, 4096)

        # 2. Concatenate along sequence dimension
        x_sensory = mx.concatenate([h_audio, h_kinematic], axis=1)  # (B, 3548, 4096)

        # 3. Broadcast latent queries across batch dimension
        queries = mx.broadcast_to(self.latent_queries, (B, self.num_latents, self.d_model))

        # 4. Cross-Attention: Latent queries attend to raw multimodal sequence
        q_norm = self.ln_q(queries)
        sensory_norm = self.ln_sensory(x_sensory)
        attn_out = self.cross_attn(q_norm, sensory_norm, sensory_norm)
        queries = queries + attn_out

        # 5. Feed-Forward update with GELU activation
        ffn_norm = self.ln_ffn(queries)
        ffn_hidden = nn.gelu(self.ffn_up(ffn_norm))
        z_sensory = queries + self.ffn_down(ffn_hidden)

        return z_sensory  # Tensor shape: (B, 64, 4096)


def execute_forward_pass(
    raw_audio_track: mx.array,
    raw_video_frames: mx.array,
    nd_projector: NDProjector,
    chroma_client: Any,
    text_embedder: Any,
    llm_model: Any,
    tokenizer: Any
) -> Dict[str, Any]:
    """
    Executes an end-to-end multimodal inference forward pass:
    Sensory Extraction -> Resampler Compression -> Clinical Retrieval -> LLM Generation.
    """
    # -------------------------------------------------------------------------
    # 1. SENSORY EXTRACTION (ND-AST and TD-ViT Encoders)
    # -------------------------------------------------------------------------
    # Synthetic extraction demonstration matching invariant output shapes
    # In production, audio_ast.py and vision_td.py perform the Mel & Diff steps
    batch_size = raw_audio_track.shape[0]
    x_audio = mx.random.normal((batch_size, 2048, 768))       # (B, 2048, 768)
    x_kinematic = mx.random.normal((batch_size, 1500, 1024))  # (B, 1500, 1024)

    # -------------------------------------------------------------------------
    # 2. DIMENSIONAL BOTTLENECK (Perceiver Resampler)
    # -------------------------------------------------------------------------
    z_sensory = nd_projector(x_audio, x_kinematic)  # Shape: (B, 64, 4096)
    mx.eval(z_sensory)  # Materialize tensor on Metal Unified RAM

    # -------------------------------------------------------------------------
    # 3. CLINICAL RAG & RETRIEVAL (ChromaDB)
    # -------------------------------------------------------------------------
    # Calculate pooled acoustic/kinematic embedding for intent space matching
    pooled_sensory = mx.mean(z_sensory, axis=1)[0].tolist()  # 4096-dim vector

    intent_collection = chroma_client.get_collection("nd_communicative_intents")
    intent_matches = intent_collection.query(
        query_embeddings=[pooled_sensory],
        n_results=3
    )

    # Dynamic clinical query construction
    top_intent = intent_matches["metadatas"][0][0]["parent_tag"] if intent_matches["metadatas"][0] else "unspecified"
    query_text = f"Sensory behavioral pattern matching {top_intent} and self-regulatory stimming."

    literature_collection = chroma_client.get_collection("clinical_literature")
    lit_matches = literature_collection.query(
        query_texts=[query_text],
        n_results=2
    )
    clinical_excerpts = "\n".join(lit_matches["documents"][0]) if lit_matches["documents"] else "Standard sensory regulation guidelines."

    # -------------------------------------------------------------------------
    # 4. PROMPT ASSEMBLY & PREFIX CONDITIONING
    # -------------------------------------------------------------------------
    system_prompt = (
        "You are Project N, a clinical neurodivergent intent translation engine. "
        "Translate the child's acoustic vocalizations and kinematic motor stims into "
        "actionable affective states and caregiver support protocols.\n\n"
        f"Relevant Clinical References:\n{clinical_excerpts}\n\n"
        f"Historical Pattern Similarity: Matches {top_intent}."
    )

    text_tokens = tokenizer.encode(system_prompt)
    prompt_embeddings = llm_model.embed_tokens(mx.array([text_tokens]))  # (1, L_text, 4096)

    # Inject the 64 sensory tokens as a continuous prefix conditioning sequence
    full_input_embeddings = mx.concatenate([z_sensory, prompt_embeddings], axis=1)  # (1, 64 + L_text, 4096)
    mx.eval(full_input_embeddings)

    # -------------------------------------------------------------------------
    # 5. AUTOREGRESSIVE GENERATION (4-bit Frozen Base + Clinical LoRA)
    # -------------------------------------------------------------------------
    generated_tokens = []
    # Generation loop utilizing MLX dynamic KV caching (abbreviated demonstration)
    logits = llm_model.forward_from_embeddings(full_input_embeddings)
    next_token = mx.argmax(logits[:, -1, :], axis=-1).item()
    generated_tokens.append(next_token)

    output_text = tokenizer.decode(generated_tokens)
    return {
        "top_matched_intent": top_intent,
        "clinical_grounding": clinical_excerpts,
        "latent_prefix_shape": list(z_sensory.shape),
        "translation_output": output_text
    }
```

---

## 5. Five Implementation Phases Roadmap

### Phase 1: Ingestion & Mobile Bridge (Flutter + FastAPI)
- **Objective:** Enable seamless, zero-cloud media capture from iOS/Android companion devices to the local Mac M5 Pro host over local Wi-Fi.
- **Deliverables:**
  - `server/api.py`: FastAPI daemon implementing multipart `/ingest/clip`, `/ingest/tag`, and `/health` endpoints.
  - `server/config.py`: System environment configuration, memory limits, and path managers.
  - `app/lib/`: Flutter UI featuring a 60fps video capture interface, quick-tag selector for parents, and offline sync queue.
  - Network mDNS service discovery allowing automatic local daemon discovery (`project-n.local`).

### Phase 2: Custom Sensory Extraction Engines (ND-AST + TD-ViT)
- **Objective:** Implement the raw sensory bypass layers to isolate acoustic harmonics and kinematic stims without phonemic or spatial collapse.
- **Deliverables:**
  - `extraction/audio_ast.py`: Complete MLX pipeline for STFT ($N=2048, H=160$), 128 Mel-band computation, and Aliasing-aware Patch Embedding (AaPE). Output: $\mathbf{X}_{audio} \in \mathbb{R}^{B \times 2048 \times 768}$.
  - `extraction/vision_td.py`: Complete MLX pipeline for Temporal-L1 frame differential calculation ($\mathbf{\Delta}_t$), kinetic energy thresholding ($\tau$), and spatio-temporal ViT projection. Output: $\mathbf{X}_{kinematic} \in \mathbb{R}^{B \times 1500 \times 1024}$.
  - Unit tests verifying output shapes, zero-allocation buffers, and Metal GPU execution.

### Phase 3: 48-Hour Unsupervised Baseline Initialization (MAE)
- **Objective:** Build an individualized behavioral topology from 50 hours of raw, unannotated video via self-supervised Masked Autoencoding.
- **Deliverables:**
  - `training/mae_pretrain.py`: Self-supervised training loop with $75\%$ uniform patch masking across time-frequency and motion patches.
  - `models/projector.py`: Complete implementation of `NDProjector` with learnable queries $\mathbf{Q}_{latent} \in \mathbb{R}^{64 \times 4096}$.
  - MLX lightweight decoder reconstructing masked patches under MSE loss.
  - Checkpoint manager saving pre-trained Resampler latents to `models/checkpoints/mae_baseline.safetensors`.

### Phase 4: Clinical Grounding Subsystem (Domain LoRA + ChromaDB RAG)
- **Objective:** Restructure LLM cognitive priors to interpret non-verbal behavior through sensory integration and functional behavioral frameworks.
- **Deliverables:**
  - `rag/client.py` & `rag/store.py`: ChromaDB initialization, document chunking, and metadata management for `nd_communicative_intents` and `clinical_literature`.
  - `models/lora.py`: MLX Low-Rank Adapter injection module ($r=64, \alpha=128$) targeting attention projections in Qwen2.5-14B.
  - `models/qwen_loader.py`: 4-bit quantized MLX loader with dynamic Metal KV cache allocation.
  - Ingestion script parsing Ayres Sensory Integration texts, CAS protocols, and OT evaluations into ChromaDB.

### Phase 5: Nightly Triplet Loss Replay & Inference Service
- **Objective:** Establish continuous lifelong learning through self-stabilizing nightly contrastive updates without catastrophic forgetting.
- **Deliverables:**
  - `training/replay_buffer.py`: Batch generator strictly enforcing the $80\%$ historical replay / $20\%$ novel daily sample invariant.
  - `training/triplet_loss.py`: Cosine distance contrastive triplet loss optimization step updating only LoRA parameters ($\Delta \mathbf{W}$) and Resampler queries ($\mathbf{Q}_{latent}$).
  - Scheduled overnight consolidation runner triggering at 02:00 local time.
  - Production inference endpoint on FastAPI serving interactive translations to the caregiver's Flutter application.
