# Project N: Multimodal Communication Support & Behavioral Insight Pipeline

[![Platform: Apple Silicon](https://img.shields.io/badge/Platform-Apple%20Silicon%20(M5%20Pro)-black?style=flat-square&logo=apple)](https://developer.apple.com/metal/)
[![Engine: Apple MLX](https://img.shields.io/badge/Engine-Apple%20MLX%200.22+-blue?style=flat-square)](https://github.com/ml-explore/mlx)
[![Memory: 48GB Unified RAM](https://img.shields.io/badge/Memory-48GB%20Unified%20RAM%20(36GB%20Cap)-green?style=flat-square)]()
[![Privacy: 100% Offline Two-Key Vault](https://img.shields.io/badge/Privacy-100%25%20Offline%20Two--Key%20Vault-red?style=flat-square)]()
[![Safety: AAC Bridge & NCCPC-R](https://img.shields.io/badge/Safety-AAC%20Bridge%20%26%20NCCPC--R-purple?style=flat-square)]()

---

## 1. Mission Statement & Product Contract

**Project N is a 100% offline, caregiver- and child-controlled communication-support and behavioral insight assistant** engineered for minimally speaking neurodivergent children (exemplified by Child N, a 7-year-old minimally speaking child).

### What Project N Is
- **An Evidence-Grounded Assistant:** It observes multi-modal recordings, references Child N's personal history of verified episodes, queries a local clinical evidence library, and surfaces calibrated possibilities alongside ways to confirm or reject them.
- **Child-Authored via AAC:** Possibilities are routed directly into Augmentative and Alternative Communication (AAC) speech-generating devices or choice boards as pre-populated options for the child to select, confirm, or reject.
- **Separated into Four Clear Evidence Layers:** Every output is strictly delineated into four distinct layers:
  1. **Layer 1 (L1 - Measured Observation):** Directly measured acoustic, kinematic, and physiological dynamics.
  2. **Layer 2 (L2 - Comparable History):** Prior caregiver-confirmed episodes with matching metric signatures and their recorded resolution outcomes.
  3. **Layer 3 (L3 - Context & Antecedents):** Caregiver-recorded environmental context, timing, meals, and transitions.
  4. **Layer 4 (L4 - Cited Evidence Library):** Versioned, cited academic excerpts documenting author, date, study population, and evidence quality.

### What Project N Is NOT
- It is **NOT** a diagnostic system. It never asserts internal states, autonomic conditions, or medical diagnoses as facts.
- It is **NOT** an autonomous "intent translator." It does not produce unconstrained narrative translations of internal child consciousness.
- It does **NOT** replace medical review. Somatic distress is evaluated via the validated Non-Communicating Children's Pain Checklist – Revised (**NCCPC-R**), with red flags triggering immediate escalation cards.

---

## 2. Inverted Target Architecture (Retrieval-First, Rendering-Only)

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              LOCAL MULTIMODAL CAPTURE                                 │
│                                                                                        │
│   Companion Client (Flutter iOS / Android / Local Sensor Bridge)                       │
│   ├─ Video & Audio: 30 fps @ 720p/1080p, 48kHz WAV audio                               │
│   ├─ Wearable Physiology: Electrodermal Activity (EDA), Heart Rate Variability (HRV),   │
│   │  and 3-axis Accelerometry (providing objective autonomic correlates)              │
│   └─ Dissent Protocol: Turning away or covering camera immediately halts capture        │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Local LAN mTLS + Signed Requests & Nonce
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      TWO-KEY ENCRYPTION VAULT & PER-USER HELPER                        │
│                                                                                        │
│   Apple macOS Signed LaunchAgent (User Context / Data Protection Keychain)             │
│   ├─ Processing Key: Per-clip AES-256-GCM DEK for automatic processing while locked    │
│   └─ Touch ID Review Key: Required to view raw media, inspect timelines, or export     │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                      ┌─────────────────────┼─────────────────────┐
                      │                     │                     │
                      ▼                     ▼                     ▼
┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
│     ACOUSTIC PIPELINE     │ │    KINEMATIC PIPELINE     │ │   PHYSIOLOGICAL PIPELINE    │
│                           │ │                           │ │                           │
│ ├─ Explicit Pitch Track:  │ │ ├─ Body-Relative Pose:    │ │ ├─ Electrodermal (EDA):   │
│ │  F0, Jitter, Shimmer,   │ │    MediaPipe Holistic /   │ │    Tonic / phasic skin    │
│ │  HNR (~10ms hop)        │ │    BlazePose landmarks    │ │    conductance response   │
│ ├─ Harmonic Filterbank:   │ │ ├─ Dense Optical Flow:    │ │ ├─ Cardiorespiratory:     │
│ │  CQT / ERB filterbank   │ │    RAFT motion vectors    │ │    HRV / pulse intervals  │
│ └─ Texture: 128 Log-Mel   │ │ └─ Context: Object & room │ │ └─ Accelerometry: Body    │
│    bands (broadband)      │ │    (TD-ViT as ablation)   │ │    motion & posture       │
└─────────────┬─────────────┘ └─────────────┬─────────────┘ └─────────────┬─────────────┘
              │                             │                             │
              └──────────────────────┬──────┴─────────────────────────────┘
                                     ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   ENCODER BINDING & METRIC EMBEDDING HEAD (MLX)                        │
│                                                                                        │
│   Multimodal Binding & Metric Projection                                               │
│   ├─ Audio-Visual Temporal Correspondence (CAV-MAE / AVC cross-modal binding)         │
│   ├─ Attention-Pooled & L2-Normalized Metric Representation (d = 128)                  │
│   └─ Metric Loss: Supervised Prototypical / Contrastive Loss over verified episodes    │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                  RETRIEVAL, CALIBRATED MATCHING & ABSTENTION ENGINE                    │
│                                                                                        │
│   Local In-Memory Vector & Episode Store (ChromaDB / SQLite)                           │
│   ├─ Historical Match: k-NN / Prototype retrieval over confirmed Child N episodes      │
│   ├─ Calibrated Multi-Label Probabilities & Prediction Sets                            │
│   ├─ ABSTENTION RULE: If d(nearest) > threshold, output "Unrecognized Pattern"        │
│   └─ RED-FLAG CHECK: If NCCPC-R distress threshold exceeded, output MEDICAL ESCALATION│
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Structured Result Object (L1, L2, L3, L4)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       CHILD-AUTHORED AAC BRIDGE & OUTPUT RENDERING                     │
│                                                                                        │
│   Primary Channel: AAC Bridge (Speech-Generating Device / Choice Board)               │
│   └─ Pre-populates candidate options for Child N to select, confirm, or reject         │
│                                                                                        │
│   Caregiver Interface: Schema-Constrained Renderer (Qwen2.5-14B-Instruct)              │
│   ├─ LLM is used ONLY for formatting structured result into four visually distinct     │
│   │  evidence layers (L1-L4). Model weights are 100% frozen via model.freeze().        │
│   └─ Strict Invariant: Prohibited from inventing labels, causes, or diagnoses.         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Hardware Prerequisites & Measured Memory Profile

Project N runs locally on Apple Silicon (tested on M5 Pro 48GB unified memory) without network dependencies.

### Hardware Envelope
- **Processor:** Apple Silicon M-series (M2/M3/M4/M5 Pro/Max).
- **Physical Memory:** 48 GB Unified RAM (parameterized for 64GB/128GB workstations).
- **Hard Memory Ceiling:** $\le 36.0\text{ GB}$ (zero disk swapping permitted).
- **Operational Peak Baseline:** $\le 28.0\text{ GB}$ (leaving $\ge 8.0\text{ GB}$ for macOS system services).

### Measured Operational Allocations
Because the LLM is decoupled from the continuous inference path and reserved for schema-constrained rendering, memory utilization is lean and predictable:

| Subsystem Component | VRAM Footprint | Allocation & Lifecycle Strategy |
| :--- | :--- | :--- |
| **Quantized Base LLM** (`Qwen2.5-14B-Instruct` 4-bit) | `~9.0 GB` | Loaded in 4-bit packed Metal arrays; weights 100% frozen via `model.freeze()`. |
| **Dynamic KV Cache** (GQA 8 KV heads @ 8k tokens) | `~1.6 GB` | Grouped Query Attention scales linearly; footprint is ~1.6 GB for full 8,192 context. |
| **Sensory Encoders & Metric Head** (MLX) | `~2.2 GB` | Pitch tracker, CQT, pose backbone, and 128-dim metric projection head. |
| **Episodic Store & ChromaDB** (Local SQLite/HNSW) | `~0.4 GB` | Versioned embeddings of verified episodes and clinical evidence excerpts. |
| **Candidate Evaluation & Refit Buffer** | `~4.0 GB` | Periodic full re-fit of metric heads and prototype centers (non-continuous). |
| **Operating System & WindowServer** | `8.0+ GB` | Preserved for macOS host stability, displays, and audio routing daemons. |
| **Total Operational Consumption** | **~17.2 GB** | **Substantially below the 28.0 GB baseline and 36.0 GB ceiling.** |

---

## 4. Repository Directory Structure

```text
project_n/
├── app/                  # Companion mobile application (Flutter iOS/Android)
│   ├── lib/              # Capture UI, dissent controls, AAC bridge interface
│   └── test/             # App unit and widget tests
├── docs/                 # Architecture, clinical governance, and review documents
│   ├── REVIEW_REFINEMENTS.md  # Comprehensive multi-reviewer scientific audit
│   └── evaluation_protocol.md # Preregistered N-of-1 study and validation protocol
├── extraction/           # Modular sensory extraction engines
│   ├── acoustic.py       # Pitch tracking (F0, jitter, shimmer, HNR) + CQT + Log-Mel
│   ├── kinematic.py      # MediaPipe/BlazePose body-relative landmarks + optical flow
│   └── physiology.py     # Wearable EDA, HRV, and accelerometry feature extractors
├── models/               # Metric learning, episodic retrieval, and rendering
│   ├── metric_head.py    # 128-dim attention-pooled projection & prototypical head
│   ├── resampler.py      # Audio-visual temporal correspondence encoder
│   ├── qwen_loader.py    # Quantized Qwen2.5-14B-Instruct loader (frozen W0)
│   └── renderer.py       # Schema-constrained four-layer (L1-L4) plain text formatter
├── rag/                  # Clinical evidence library & versioned retrieval store
│   ├── evidence_store.py # Versioned academic literature chunks with scope metadata
│   └── intent_store.py   # Caregiver-confirmed episodes with encoder version lineage
├── server/               # Local backend daemon & two-key encrypted storage
│   ├── api.py            # Local REST endpoints (mTLS authenticated)
│   ├── config.py         # Hardware thresholds, paths, and tunable hyperparameters
│   └── vault.py          # Two-key keychain integration (Data Protection Keychain)
├── training/             # Metric training, episodic re-fitting & model promotion
│   ├── pretrain_av.py    # Self-supervised audio-visual binding pre-training
│   ├── refit_metric.py   # Periodic full re-fit of prototype metric representations
│   └── promotion_gate.py # Multi-metric validation and safety regression tests
├── AGENTS.md             # Autonomous AI engineer directives and conventions
├── DESIGN.md             # Theoretical, neurobiological, and epistemic design
├── INVARIANTS.md         # True non-negotiable invariants vs tunable defaults
├── SPECS.md              # Technical specifications, schemas, and MLX pseudo-code
└── README.md             # Project overview, architecture, and onboarding guide
```

---

## 5. Quickstart Guide

### Prerequisites
- Apple Silicon Mac running macOS 15.0+ (Sequoia or later).
- Python 3.11 or 3.12 (`python3 --version`).
- Native Apple MLX framework (`pip install mlx>=0.22.0`).
- FFmpeg with VideoToolbox support (`brew install ffmpeg`).

### Step 1: Environment Setup
```bash
git clone https://github.com/olostan/project_n.git
cd project_n

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install mlx>=0.22.0 numpy scipy soundfile chromadb fastapi uvicorn pydantic mediapipe
```

### Step 2: Verify Apple MLX Acceleration and Memory Queries
```bash
python -c "
import mlx.core as mx
print('MLX Default Device:', mx.default_device())
print('Active Memory (GB):', mx.get_active_memory() / 1e9)
print('Peak Memory (GB):  ', mx.get_peak_memory() / 1e9)
"
```

### Step 3: Initialize the Encrypted Vault and Vector Stores
```bash
python -c "
import chromadb
client = chromadb.PersistentClient(path='./chroma_db')
intents = client.get_or_create_collection('nd_confirmed_episodes', metadata={'hnsw:space': 'cosine'})
evidence = client.get_or_create_collection('clinical_evidence', metadata={'hnsw:space': 'cosine'})
print('Initialized Collections:', client.list_collections())
"
```

### Step 4: Run the Multi-Modal Verification Test
```bash
python -c "
from models.metric_head import MetricProjectionHead
import mlx.core as mx
head = MetricProjectionHead(d_audio=768, d_pose=512, d_physio=64, d_metric=128)
z = head(mx.zeros((1, 200, 768)), mx.zeros((1, 150, 512)), mx.zeros((1, 50, 64)))
print('Metric Vector Output Shape:', z.shape)
assert z.shape == (1, 128), 'Shape mismatch!'
"
```

---

## 6. Safety, Epistemics & Governance

1. **Child Authorship First:** Possibilities are routed to Child N via an AAC bridge. His confirmed selections or behavioral gestures outrank all adult interpretations.
2. **Medical Precedence:** Pain or acute distress evaluated under NCCPC-R triggers medical review escalations, superseding any behavioral interpretation.
3. **Calibrated Abstention:** If a sensory pattern is novel or ambiguous ($d > \text{threshold}$), the system explicitly responds with *"Unrecognized pattern"* and offers observational or AAC-based exploration.
4. **No Autonomous Deployment:** New models are promoted strictly through the gated pipeline defined in [`INVARIANTS.md`](file:///Users/olostan/code/project_n/INVARIANTS.md) and [`docs/REVIEW_REFINEMENTS.md`](file:///Users/olostan/code/project_n/docs/REVIEW_REFINEMENTS.md).
