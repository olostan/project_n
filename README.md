# Project N: Multimodal Intent Translation for Non-Verbal Autism

[![Platform: Apple Silicon](https://img.shields.io/badge/Platform-Apple%20Silicon%20(M5%20Pro)-black?style=flat-square&logo=apple)](https://developer.apple.com/metal/)
[![Engine: Apple MLX](https://img.shields.io/badge/Engine-Apple%20MLX%200.22+-blue?style=flat-square)](https://github.com/ml-explore/mlx)
[![Memory: 48GB Unified RAM](https://img.shields.io/badge/Memory-48GB%20Unified%20RAM%20(36GB%20Cap)-green?style=flat-square)]()
[![Privacy: 100% Offline Local-First](https://img.shields.io/badge/Privacy-100%25%20Offline%20Local--First-red?style=flat-square)]()
[![License: Proprietary / Research Core](https://img.shields.io/badge/License-Research%20Core-purple?style=flat-square)]()

---

## 1. Mission Statement

Project N democratizes non-verbal intent translation through local-first, privacy-preserving multimodal intelligence tailored to individualized neurodivergent communication. 

Commercial and open-weight foundation models suffer from an inherent neurotypical inductive bias. Standard automatic speech recognition (ASR) pipelines discard raw acoustic resonance, tonal hums, and guttural vocalizations as "untranscribable noise," while standard vision transformers (ViTs) spatially pool away high-frequency repetitive motor stims (such as rapid wrist rotation, finger-flicking, or postural shifts). For a Level 3 non-verbal autistic individual—specifically exemplified by Nolan, a 7-year-old minimally speaking child—this idiosyncratic sensory "noise" forms the entire substrate of expressive communication.

Project N provides a zero-cloud, strictly offline architecture executing locally on Apple Silicon (M5 Pro, 48GB Unified Memory). By pairing custom raw sensory encoders with a Perceiver Resampler bottleneck, domain-specific clinical grounding, and nightly self-stabilizing contrastive replay loops, Project N decodes subtle somatic and vocal expressions into clear, actionable intent for parents and caregivers.

---

## 2. End-to-End System Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CLIENT INTERFACE                                     │
│                                                                                        │
│   Flutter Mobile Client (iOS / Android / macOS Local Target)                           │
│   ├─ Capture Engine: 30s–120s High-Framerate Video (1080p @ 60fps) + 48kHz WAV Audio   │
│   └─ Micro-Feedback: Voice Notes, Antecedent Tags, Intent Corrections, Affect Sliders   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Local LAN Wi-Fi / mDNS HTTP REST
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        LOCAL INGESTION & FASTAPI DAEMON (MACOS)                        │
│                                                                                        │
│   FastAPI Service (Uvicorn / Asynchronous Buffer / Metal Acceleration)                 │
│   ├─ Ingestion Pipeline: Multipart clip validation, zero-copy buffer handoff          │
│   └─ FFmpeg Stream Demuxing: High-precision audio extraction & frame diff preprocessing│
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      │                                           │
                      ▼                                           ▼
┌───────────────────────────────────────────┐ ┌──────────────────────────────────────────┐
│          ACOUSTIC SENSORY PATH            │ │          KINEMATIC SENSORY PATH          │
│                                           │ │                                          │
│   ND-AST (Spectrogram Transformer)        │ │   TD-ViT (Temporal Difference ViT)       │
│   ├─ Raw Audio (44.1kHz / 48kHz WAV)      │ │   ├─ Video Stream (60fps Decoded Frames) │
│   ├─ STFT: n_fft=2048, hop=160 (~3.6ms)   │ │   ├─ Temporal-L1 Diff: Δt = ||zt-zt-1||1 │
│   ├─ Mel Filterbank: 128 Filter Bands     │ │   ├─ Threshold Filter: Drops Static Bkgd │
│   ├─ Aliasing-aware Patch Embedding (AaPE)│ │   ├─ Isolates Stims (3Hz-8Hz Motor Stims)│
│   └─ Output: X_audio ∈ R^(B x 2048 x 768) │ │   └─ Output: X_kinematic ∈ R^(B x 1500 x │
│                                           │ │                                     1024)│
└─────────────────────┬─────────────────────┘ └────────────────────┬─────────────────────┘
                      │                                            │
                      └─────────────────────┬──────────────────────┘
                                            │ Concat: [X_audio, X_kinematic] (~3548 tokens)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                 DIMENSIONAL BOTTLENECK: PERCEIVER RESAMPLER                            │
│                                                                                        │
│   Apple MLX NDProjector (Cross-Attention Compression Engine)                           │
│   ├─ Latent Queries: M = 64 Learnable Vectors (D = 4096)                               │
│   ├─ Multi-Head Cross-Attention (scaled_dot_product_attention on Apple Metal M5 Pro)   │
│   └─ Output Bottleneck: Z_sensory ∈ R^(B x 64 x 4096)                                  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Prefix Conditioning Vector Injection
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                 CLINICAL REASONING & INFERENCE CORE (MLX 14B BASE)                     │
│                                                                                        │
│   4-bit Quantized Foundation LLM (Qwen2.5-Omni / Qwen2.5-14B-Instruct)                │
│   ├─ Base Weights (W0): 100% Frozen in Metal Unified Memory (4-bit, ~9.0 GB)           │
│   ├─ Clinical Domain LoRA: Low-Rank Adapter (r=64, α=128, ~400 MB trainable)           │
│   │   └─ Ingests: Sensory Integration, Antecedent-Behavior-Consequence (ABC) Frameworks│
│   ├─ Dynamic Retrieval-Augmented Generation (ChromaDB Vector Store)                    │
│   │   ├─ Collection 1: nd_communicative_intents (Child's Historical Vector Lexicon)    │
│   │   └─ Collection 2: clinical_literature (Ayres, Greenspan, Functional Analysis)    │
│   └─ Context Generation: Child State Diagnosis, Intent Hypothesis & Caregiver Guidance │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               CONTINUOUS ADAPTATION & NIGHTLY LEARNING PIPELINE                        │
│                                                                                        │
│   Nightly Self-Stabilizing Triplet Loss Optimization                                  │
│   ├─ Anchor / Positive / Negative Triplet Formulation with Cosine Distance Metric     │
│   ├─ Replay Buffer Invariant: 80% Historical Replay Anchors / 20% Novel Daily Samples  │
│   └─ Target Updates: Confined exclusively to LoRA weights (ΔW) and Perceiver Queries   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Hardware Prerequisites & Memory Footprint

Project N is engineered specifically for Apple Silicon hardware, leveraging Apple MLX for unified memory computation and native Metal Shading Language (MSL) acceleration.

### Tested Configuration
- **Hardware Target:** Apple Silicon M5 Pro (or M2/M3/M4 Max/Pro equivalents).
- **Unified Memory:** 48 GB Unified RAM.
- **Operating Memory Ceiling:** Maximum allocated VRAM $\le 36.0\text{ GB}$.
- **Operational Peak Baseline:** $\le 28.0\text{ GB}$ (guaranteeing $\ge 8.0\text{ GB}$ remaining for macOS, WindowServer, audio routing, and OS services).
- **Zero-Swapping Invariant:** Hard threshold forbidding dynamic disk swapping during forward passes or nightly gradient steps.

### Memory Allocation Matrix
| Subsystem Component | VRAM Footprint | Allocation & Optimization Rationale |
| :--- | :--- | :--- |
| **Quantized Base LLM** (Qwen2.5 14B 4-bit) | `9.0 GB` | 4-bit Group-128 quantization via MLX format (~0.5 GB / billion parameters). |
| **Dynamic Context & KV Cache** | `10.0 GB` | MLX lazy evaluation graph, 8k sequence window, Metal page recycling. |
| **Training States & Replay Buffer** | `8.0 GB` | AdamW optimizer states restricted to LoRA and Resampler parameters ($r=64$). |
| **Clinical Domain LoRA Adapter** | `0.4 GB` | Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. |
| **Local Text Embedder** (`nomic-embed-text-v1.5`) | `0.2 GB` | Offline text embedding engine running under MLX runtime (<150 MB static). |
| **ChromaDB Vector Store (In-Memory)** | `0.4 GB` | Persistent SQLite + HNSW index caching child-specific intent topologies. |
| **System Headroom (macOS + Desktop)** | `8.0+ GB` | Preserved for host operating system stability and interactive UI responsiveness. |
| **Total Operational Consumption** | **~28.0 GB** | **Leaves 8.0 GB safe headroom below the 36.0 GB ceiling.** |

*Note: For systems upgraded to 64GB or 128GB Unified Memory, base model architecture expands to 32B parameters (`Qwen2.5-32B-Instruct` 4-bit, allocating 19.5 GB static weights) via the `--model_size 32b` flag.*

---

## 4. Repository Directory Structure

```text
project_n/
├── app/                  # Flutter mobile capture, telemetry & caregiver review interface
│   ├── android/          # Android platform runner configuration
│   ├── ios/              # iOS platform runner configuration
│   └── lib/              # Flutter UI: Live recording, clip tagging & intent timeline
├── docs/                 # Clinical references, formal architecture docs & safety logs
├── extraction/           # ND-AST and TD-ViT sensory extraction layers (MLX native)
│   ├── audio_ast.py      # Neurodivergent Audio Spectrogram Transformer (AaPE)
│   └── vision_td.py      # Temporal Difference Vision Transformer (L1 differential)
├── models/               # Model definitions, projector, LoRA, and LLM orchestration
│   ├── lora.py           # MLX LoRA injection module (r=64, alpha=128)
│   ├── projector.py      # Perceiver Resampler cross-attention bottleneck (64 tokens)
│   └── qwen_loader.py    # Quantized 14B/32B base model loader and KV manager
├── rag/                  # Clinical document ingestion, chunking, and ChromaDB vector engine
│   ├── client.py         # Persistent ChromaDB client and local embedding wrapper
│   └── store.py          # Collections: nd_communicative_intents & clinical_literature
├── server/               # FastAPI local inference and ingestion service
│   ├── api.py            # Local REST endpoints for video upload, review, and inference
│   └── config.py         # Hardware thresholds, memory monitor, and path configurations
├── training/             # 48-Hour unsupervised baseline MAE & nightly Triplet Loss loop
│   ├── mae_pretrain.py   # Self-supervised 75% masked autoencoder pipeline
│   ├── replay_buffer.py  # 80/20 historical replay vs. novel sample generator
│   └── triplet_loss.py   # Cosine distance contrastive training step
├── AGENTS.md             # Autonomous AI engineer directives and documentation protocols
├── DESIGN.md             # Deep architectural, neurobiological, and mathematical design
├── INVARIANTS.md         # Non-negotiable system, privacy, memory, and model invariants
├── SPECS.md              # Exhaustive technical specifications and implementation roadmap
└── README.md             # Project vision, architecture overview, and onboarding guide
```

---

## 5. Quickstart Guide

### Prerequisites
- macOS 15.0+ (Sequoia or later) on Apple Silicon (M-series Pro/Max/Ultra).
- Xcode Command Line Tools installed: `xcode-select --install`.
- Python 3.11 or Python 3.12 (managed via `pyenv` or `conda`).
- FFmpeg compiled with Apple Metal VideoToolbox support: `brew install ffmpeg`.
- Flutter 3.24+ (if working on the mobile companion app).

### Step 1: Clone and Set Up Virtual Environment
```bash
git clone https://github.com/olostan/project_n.git
cd project_n

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Step 2: Install MLX and Core Dependencies
```bash
pip install mlx>=0.22.0
pip install numpy scipy chromadb fastapi uvicorn pydantic soundfile librosa torchvision
```

### Step 3: Run Baseline Extraction and Hardware Verification Test
Verify that Apple Silicon GPU acceleration via Metal Performance Shaders is active and sensory extractors instantiate within memory constraints:

```bash
python -c "
import mlx.core as mx
import mlx.nn as nn
print('MLX Default Device:', mx.default_device())
a = mx.ones((1000, 1000))
b = mx.ones((1000, 1000))
c = mx.matmul(a, b)
mx.eval(c)
print('Metal Sanity Check Passed. C[0,0] =', c[0,0].item())
"
```

### Step 4: Initialize the Local ChromaDB Vector Store
```bash
python -c "
import chromadb
from chromadb.config import Settings
client = chromadb.PersistentClient(path='./chroma_db')
intents = client.get_or_create_collection(
    name='nd_communicative_intents',
    metadata={'hnsw:space': 'cosine'}
)
literature = client.get_or_create_collection(
    name='clinical_literature',
    metadata={'hnsw:space': 'cosine'}
)
print('Initialized Local Chroma Collections. Count:', client.list_collections())
"
```

### Step 5: Launch Local Ingestion & Inference Daemon
```bash
uvicorn server.api:app --host 127.0.0.1 --port 8080 --reload
```
The server will bind to `127.0.0.1:8080`, exposing the local ingest endpoints for the Flutter client and running inference exclusively over local Metal-accelerated MLX pipelines.

---

## 6. Development Workflow & Agent Guidelines

All modifications to this repository are governed by strict operational rules:
1. Review [`INVARIANTS.md`](file:///Users/olostan/code/project_n/INVARIANTS.md) before introducing any architectural or dependency modifications. Zero cloud leaks or standard phonemic ASR integrations are tolerated.
2. Ensure automated agents follow the protocol detailed in [`AGENTS.md`](file:///Users/olostan/code/project_n/AGENTS.md).
3. Detailed mathematical definitions, tensor shapes, and MLX pseudo-code reside in [`SPECS.md`](file:///Users/olostan/code/project_n/SPECS.md).
4. Full biological grounding, sensory encoder bypass philosophy, and continuous adaptation logic are documented in [`DESIGN.md`](file:///Users/olostan/code/project_n/DESIGN.md).
