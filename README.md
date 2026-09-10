# Project N: Multimodal Communication Support & Behavioral Insight Pipeline

[![Platform: Apple Silicon](https://img.shields.io/badge/Platform-Apple%20Silicon%20(M5%20Pro)-black?style=flat-square&logo=apple)](https://developer.apple.com/metal/)
[![Engine: Apple MLX](https://img.shields.io/badge/Engine-Apple%20MLX%200.22+-blue?style=flat-square)](https://github.com/ml-explore/mlx)
[![Memory: 48GB Unified RAM](https://img.shields.io/badge/Memory-48GB%20Unified%20RAM%20(36GB%20Cap)-green?style=flat-square)]()
[![Privacy: 100% Offline Two-Key Vault](https://img.shields.io/badge/Privacy-100%25%20Offline%20Two--Key%20Vault-red?style=flat-square)]()
[![Clients: Flutter (Android/iOS) + React Dashboard](https://img.shields.io/badge/Clients-Flutter%20%2B%20React%20SPA-blueviolet?style=flat-square)]()
[![Safety: AAC Bridge & NCCPC-R](https://img.shields.io/badge/Safety-AAC%20Bridge%20%26%20NCCPC--R-purple?style=flat-square)]()

---

## 1. Mission Statement & Product Contract

**Project N is a 100% offline, caregiver- and child-controlled communication-support and behavioral insight assistant** engineered for minimally speaking neurodivergent children (exemplified by Child N, a 7-year-old minimally speaking child).

### Core Capabilities
- **Acoustic & Motion Matching Without Words:** Level 3 non-verbal vocalizations and repetitive motor stims are analyzed using raw acoustic physics (pitch contour, jitter, shimmer, harmonic overtones) and body-relative pose tracking. The system matches sounds and movements directly against Child N's past verified episodes without ever forcing idiosyncratic signals into clumsy English descriptions.
- **Store-and-Forward Offline Mobile Capture:** Record clips at playgrounds, outdoor parks, or Occupational Therapy (OT) clinics on an Android or iOS phone without Wi-Fi. Clips are stored in an encrypted local outbox and automatically flushed to your Mac upon returning home.
- **Child-Authored via AAC Bridge:** Inferred possibilities are pre-populated onto Child N's speech-generating device or AAC choice board as candidate options for him to select, confirm, or reject. His selection outranks all adult interpretations.
- **Medical Distress Rule-Out (NCCPC-R):** Acute distress is evaluated using the validated 27-item Non-Communicating Children’s Pain Checklist – Revised, with red flags immediately triggering medical review escalations.
- **Local Mac Caregiver Dashboard:** A modern React + Tailwind web interface running locally on your Mac (`http://localhost:8080`), featuring live unified memory monitoring, an interactive video inspector with synchronized pitch/pose graphs, a 2D UMAP cluster map of learned behaviors, and model promotion gates.

---

## 2. End-to-End System Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              MOBILE COMPANION CAPTURE                                  │
│                                                                                        │
│   Flutter Mobile Client (Android 12+ & iOS 17+)                                        │
│   ├─ Capture Engine: 30 fps @ 720p/1080p Video + 48kHz WAV Audio                       │
│   ├─ Optional Physiology: Wearable sensor telemetry (EDA/HRV) - masked if absent       │
│   ├─ Offline Outbox: Encrypted SQLite queue for recordings at playgrounds & OT clinics │
│   └─ Background Sync: Auto-flushes over mTLS when home Wi-Fi connects to Mac helper   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Local LAN mTLS + Signed Requests & Nonces
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        LOCAL MAC SERVER & TWO-KEY VAULT DAEMON                         │
│                                                                                        │
│   Python 3.11+ / FastAPI + Uvicorn (Native Apple Silicon Unified Memory)               │
│   ├─ LaunchAgent Helper: Background encryption using Data Protection Keychain (SecItem)│
│   ├─ SSE Event Bus: Streams live progress, memory metrics & inference stages           │
│   ├─ Relational Storage: Local SQLite for episode metadata, outbox tracking, and logs  │
│   └─ Vector Storage: Local ChromaDB collections for metric vectors & clinical evidence │
└───────────────────────┬───────────────────────────────────────────┬────────────────────┘
                        │                                           │
                        ▼                                           ▼
┌──────────────────────────────────────────┐    ┌────────────────────────────────────────┐
│   LOCAL CAREGIVER DASHBOARD (REACT SPA)  │    │      NEURAL ENGINE CORE (APPLE MLX)    │
│                                          │    │                                        │
│   React 18 + Tailwind CSS (Bundled SPA)  │    │   Apple MLX (mlx.core, mlx.nn on Metal)│
│   ├─ Live Memory & Telemetry Gauge       │    │   ├─ Acoustic: F0/Jitter/Shimmer/CQT   │
│   ├─ Episode Timeline & Filterable Diary │    │   ├─ Kinematic: MediaPipe / RAFT flow  │
│   ├─ Video Inspector (Pitch + Pose Sync) │    │   ├─ Metric Head: 128-dim L2 Vector    │
│   ├─ 2D UMAP Behavioral Cluster Browser  │    │   ├─ Episodic Matcher & Abstention Gate│
│   └─ Model Promotion & Safety Holdouts   │    │   └─ Qwen2.5-14B-Instruct Renderer     │
└──────────────────────────────────────────┘    └────────────────────────────────────────┘
```

---

## 3. Technology Stack Rationale

| Layer | Technology Selected | Engineering Rationale |
| :--- | :--- | :--- |
| **Backend Daemon** | **Python 3.11+ / FastAPI** | Direct zero-copy shared memory access to Apple MLX C++ Metal arrays. Avoids IPC memory duplication inherent in a Go sidecar. |
| **Machine Learning Engine** | **Apple MLX 0.22+** | Native Metal Performance Shaders (MPS) acceleration on Apple Silicon unified memory. |
| **Caregiver Dashboard UI** | **React 18 + Tailwind CSS** | Fast, responsive single-page application bundled locally into FastAPI static files. Zero external cloud CDNs. |
| **Real-Time Client Stream**| **Server-Sent Events (SSE)**| Lightweight, unidirectional HTTP streaming pushing live pipeline progress, unified memory gauges, and LLM text generation. |
| **Mobile Companion Client**| **Flutter 3.24+ (Dart)** | Single codebase compiling natively to Android and iOS. Hardware-backed security via Android Keystore and iOS Keychain. |
| **Local Vector Database** | **ChromaDB (Persistent)** | Embedded in-process vector store with HNSW cosine indexing for 128-dim metric vectors and 768-dim text chunks. |
| **Relational Metadata Store**| **SQLite (WAL mode)** | Lightweight, zero-configuration embedded relational database for episodic histories and mobile outbox queues. |

---

## 4. Hardware Prerequisites & Measured Memory Profile

Project N runs locally on Apple Silicon (tested on M5 Pro with 48GB Unified Memory) with zero external internet dependencies.

### Measured Memory Allocation (Operational Peak: ~17.2 GB)
Because the 14B base LLM is decoupled from the continuous inference path and reserved strictly for schema-constrained rendering, memory utilization is lean and predictable:

| Subsystem Component | VRAM Footprint | Allocation & Lifecycle Strategy |
| :--- | :--- | :--- |
| **Quantized Base LLM** (`Qwen2.5-14B-Instruct` 4-bit) | `~9.0 GB` | Loaded in 4-bit packed Metal arrays; weights 100% frozen via `model.freeze()`. |
| **Dynamic KV Cache** (GQA 8 KV heads @ 8k tokens) | `~1.6 GB` | Grouped Query Attention scales linearly; footprint is ~1.6 GB for full 8,192 context. |
| **Sensory Encoders & Metric Head** (MLX) | `~2.2 GB` | Pitch tracker, CQT, pose backbone, and 128-dim projection head. |
| **Episodic Store & ChromaDB** (Local SQLite/HNSW) | `~0.4 GB` | Versioned embeddings of verified episodes and clinical evidence excerpts. |
| **Candidate Evaluation & Refit Buffer** | `~4.0 GB` | Periodic full re-fit of metric heads and prototype centers (non-continuous). |
| **Operating System & WindowServer** | `8.0+ GB` | Preserved for macOS host stability, displays, and audio routing daemons. |
| **Total Operational Consumption** | **~17.2 GB** | **Substantially below the 28.0 GB baseline and 36.0 GB ceiling.** |

---

## 5. Repository Directory Structure

```text
project_n/
├── app/                  # Mobile companion application (Flutter Android & iOS)
│   ├── android/          # Native Android configuration with Keystore security
│   ├── ios/              # Native iOS configuration with Keychain security
│   └── lib/              # Flutter UI: Camera capture, offline SQLite queue, AAC bridge
├── docs/                 # Architecture, clinical governance, and review documents
│   ├── REVIEW_REFINEMENTS.md  # Comprehensive multi-reviewer scientific audit
│   ├── evaluation_protocol.md # Preregistered N-of-1 validation protocol & baselines
│   └── report-source.md       # Raw peer review source document
├── extraction/           # Modular sensory extraction engines (MLX native)
│   ├── acoustic.py       # F0/jitter/shimmer/HNR @ 10ms hop + CQT 84-bin + 128 Log-Mel
│   ├── kinematic.py      # MediaPipe Holistic torso-normalized landmarks + RAFT flow
│   └── physiology.py     # Optional wearable EDA (tonic/phasic), HRV, and accelerometry
├── models/               # Metric learning, episodic retrieval, and rendering
│   ├── metric_head.py    # 128-dim attention-pooled projection & missing-modality gate
│   ├── resampler.py      # Audio-visual temporal correspondence encoder (CAV-MAE)
│   ├── qwen_loader.py    # Quantized Qwen2.5-14B-Instruct loader (frozen W0)
│   └── renderer.py       # Schema-constrained four-layer (L1–L4) plain text formatter
├── rag/                  # Clinical evidence library & versioned retrieval store
│   ├── evidence_store.py # Versioned academic literature chunks with scope metadata
│   └── intent_store.py   # Caregiver-confirmed episodes with encoder version lineage
├── server/               # Local backend daemon & two-key encrypted storage
│   ├── api.py            # Local REST endpoints (mTLS authenticated)
│   ├── sse.py            # Real-time Server-Sent Events (SSE) streaming bus
│   ├── config.py         # Hardware thresholds, paths, and tunable hyperparameters
│   ├── vault.py          # Apple Data Protection Keychain integration (SecItem)
│   └── static/           # Bundled React + Tailwind Caregiver Web Dashboard
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

## 6. Safety, Epistemics & Governance

1. **Child Authorship First:** Possibilities are routed to Child N via an AAC bridge. His confirmed selections or behavioral gestures outrank all adult interpretations.
2. **Medical Precedence:** Pain or acute distress evaluated under NCCPC-R triggers medical review escalations, superseding any behavioral interpretation.
3. **Calibrated Abstention:** If a sensory pattern is novel or ambiguous ($d > \tau_{abstain}$), the system explicitly responds with *"Unrecognized pattern"* and offers observational or AAC-based exploration.
4. **No Autonomous Deployment:** New models are promoted strictly through the gated pipeline defined in [`INVARIANTS.md`](file:///Users/olostan/code/project_n/INVARIANTS.md) and [`docs/evaluation_protocol.md`](file:///Users/olostan/code/project_n/docs/evaluation_protocol.md).
