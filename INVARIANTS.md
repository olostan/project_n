# Project N: Strict Architectural & System Invariants

---

## Preamble
This document establishes the inviolable system, mathematical, memory, privacy, and architectural invariants governing **Project N**. Every human contributor, automated script, and autonomous AI coding agent (e.g., Antigravity, Cursor, Claude Code) operating on this codebase is legally and technically bound by these rules.

A violation of an invariant constitutes an immediate regression and build failure. Automated test harnesses and pull-request filters must assert these invariants prior to merging or executing code.

---

## Invariant 1: Privacy & Offline Execution Invariant

### Formal Rule
**Zero external network telemetry. All model weights, inference pipelines, embeddings, audio tracks, video clips, metadata, and generated outputs MUST process strictly offline on local hardware.**

### Specifications
1. **Forbidden APIs & SDKs:** Inclusion or invocation of remote cloud AI APIs (including OpenAI, Anthropic, Google Cloud Vertex/Gemini, AWS Bedrock, HuggingFace Inference API, or any remote telemetry collector) is strictly banned across all production and test modules.
2. **Local Loopback Isolation:** The FastAPI daemon and mobile companion services must bind strictly to local network interfaces (`127.0.0.1` or LAN `192.168.x.x` / `project-n.local` via mDNS). No outbound WAN traffic may be initiated by any backend process.
3. **Sensitive Media Protection:** Nolan's raw video recordings and audio tracks are sensitive pediatric health records. They must reside in local encrypted directories and never be serialized into unencrypted shared volumes or public Git commits.
4. **Offline Assertions:** The automated test suite must execute inside a network-sandboxed environment (e.g., `pytest` run with blocked socket connections except localhost).

---

## Invariant 2: Hardware Memory Cap (36.0 GB Hard Invariant)

### Formal Rule
**Peak VRAM / Unified Memory consumption must NEVER exceed 36.0 GB on 48GB Unified Memory Apple Silicon hardware. Zero page swapping to the host SSD is permitted during training or inference.**

### Specifications
1. **Peak Operational Baseline:** The operational baseline during active inference must remain $\le 28.0\text{ GB}$, ensuring at least $8.0\text{ GB}$ of headroom below the $36.0\text{ GB}$ ceiling and preserving $\ge 8.0\text{ GB}$ for macOS host processes (WindowServer, CoreAudio, system daemons).
2. **Zero Swapping Assertion:** Memory allocation must not cause swap activity. Swap usage before and after execution must be checked via `sysctl vm.swapusage`. If `swapins` or `swapouts` increase during a forward pass or nightly training cycle, the process must immediately abort.
3. **MLX Graph Garbage Collection:** Long-running loops must strategically trigger `mx.eval()` and garbage collection to release dead intermediate nodes from the Metal unified memory computation graph.
4. **Out-of-Memory (OOM) Protection:** All batch sizes, context window lengths, and tensor allocations must be parameterized and clamped within safe upper bounds.

---

## Invariant 3: Foundational LLM Weight Freezing Invariant

### Formal Rule
**The foundational Large Language Model weights ($\mathbf{W}_0$) must remain 100% frozen in memory. Weight updates are strictly confined to the Perceiver Resampler projection matrices and Low-Rank Adapters ($\Delta \mathbf{W} = \frac{\alpha}{r} \mathbf{B} \mathbf{A}$).**

### Specifications
1. **Frozen State:** In the MLX parameter hierarchy, all parameters belonging to the base LLM (`model.layers.*`) must have their `trainable` flags set to `False`.
2. **Quantization Integrity:** The base LLM must be loaded in 4-bit quantized format directly into Metal unified memory. De-quantization of $\mathbf{W}_0$ into FP16/FP32 for fine-tuning is prohibited.
3. **Trainable Parameter Budget:** Trainable parameters during nightly adaptation must never exceed:
   - Perceiver Resampler (`NDProjector`): $64 \times 4096$ latent queries + cross-attention + FFN ($\approx 68\text{M}$ parameters).
   - Clinical Domain LoRA ($r=64, \alpha=128$): Adapter projections ($\approx 85\text{M}$ parameters).
   - Total Trainable Footprint: $\le 160\text{M}$ parameters ($\approx 320\text{ MB}$ in FP16), keeping AdamW states strictly within the $8.0\text{ GB}$ training budget.

---

## Invariant 4: Inductive Bias Invariant (Sensory Bypass)

### Formal Rule
**Standard phonemic speech-to-text engines (e.g., Whisper, Wav2Vec2) and standard spatial vision classification heads must NEVER be placed in the primary sensory extraction loop. Raw time-frequency Mel-spectrograms and Temporal-L1 kinematic differentials are mandatory.**

### Specifications
1. **The ASR Ban:** No component may convert raw vocalizations into intermediate phonemes, words, or text transcripts before presenting the signal to the multimodal projector. Standard ASR obliterates micro-pitch, harmonic resonance, and tonal hums.
2. **The Spatial Pooling Ban:** Video inputs must not be pre-processed using spatial pooling that discards high-frequency motor stims. Temporal differential masking ($\mathbf{\Delta}_t = \|\mathbf{Z}_t - \mathbf{Z}_{t-1}\|_1$) with threshold filtering ($\tau$) is mandatory to isolate rapid motor dynamics from static background pixels.
3. **Direct Latent Ingestion:** The 64 compressed sensory tokens ($\mathbf{Z}_{sensory}$) produced by the Perceiver Resampler must be injected directly into the LLM prefix conditioning space. Text-only intermediate descriptions ("Child is moving hands and humming") are prohibited as sensory proxies.

---

## Invariant 5: Replay Buffer Invariant (The 80/20 Rule)

### Formal Rule
**Every nightly training batch MUST adhere strictly to the 80/20 rule: exactly 80% historical replay vectors and 20% novel daily vectors. Training on unbuffered novel data alone is strictly prohibited.**

### Specifications
1. **Batch Allocation Ratio:** For every optimization batch of size $B = 10$:
   - Exactly 8 samples ($80\%$) must be drawn from the historical ChromaDB intent collection (`nd_communicative_intents`), incorporating both random historical exemplars and hard-mined negative anchors.
   - Exactly 2 samples ($20\%$) must be drawn from the newly collected, parent-tagged recordings of the current 24-hour cycle.
2. **Catastrophic Forgetting Prevention:** Violating the 80/20 ratio causes catastrophic forgetting, warping previously calibrated latent clusters. If fewer than 2 novel samples exist on a given day, replay continues with $100\%$ historical self-reinforcement. If novel samples exceed the daily limit, they must be partitioned into multiple 80/20 batches.
3. **Contrastive Triplet Consistency:** Contrastive Triplet Loss ($\mathcal{L}_{triplet}$) must enforce positive and negative pairings across both historical and novel instances using the cosine distance metric:
   $$\mathcal{L}_{triplet} = \max\left(d(\mathbf{a}, \mathbf{p}) - d(\mathbf{a}, \mathbf{n}) + 0.25, \; 0\right)$$

---

## Invariant 6: Apple MLX Native Framework Invariant

### Formal Rule
**Core tensor operations, forward passes, sensory encoders, Resampler modules, and training loops MUST target Apple MLX (`mlx.core`, `mlx.nn`). PyTorch and CUDA runtime dependencies are forbidden in the primary execution path.**

### Specifications
1. **Framework Purity:** Production files under `extraction/`, `models/`, `training/`, and `server/` must import `mlx.core as mx` and `mlx.nn as nn`. PyTorch (`torch`), Torchvision (`torchvision`), or CUDA-specific primitives are banned from the primary execution pipeline.
2. **Metal Acceleration:** All attention blocks must invoke `mx.fast.scaled_dot_product_attention` to leverage Apple Silicon Metal Performance Shaders (MPS).
3. **Lazy Evaluation Discipline:** Agents and developers must prevent unbounded graph growth by applying `mx.eval()` at deterministic synchronization boundaries (loss evaluations, log steps, and batch transitions).
4. **Data Interchange Exception:** Standard CPU-bound ingestion tools (`numpy`, `scipy`, `soundfile`, `librosa`, `av`, `ffmpeg`) are permitted for initial audio/video file demuxing prior to converting arrays into `mx.array`.

---

## Enforcement and Verification Table

| Invariant ID | Target Subsystem | Automated Verification Command | Action on Failure |
| :--- | :--- | :--- | :--- |
| **INV-1** | Network / Telemetry | `pytest tests/test_offline_sandbox.py` | Immediate process abort; reject PR |
| **INV-2** | Memory / Hardware | `python tests/verify_memory_ceiling.py --max_gb 36.0` | Abort forward pass; release GPU cache |
| **INV-3** | Weight Freezing | `python tests/verify_frozen_weights.py` | Refuse gradient update step |
| **INV-4** | Inductive Bias | `pytest tests/test_sensory_pipeline.py` | Reject intermediate text descriptions |
| **INV-5** | Replay Buffer | `python -m training.replay_buffer --verify_ratio 0.8` | Halt nightly training loop |
| **INV-6** | MLX Native | `grep -rn "import torch" extraction/ models/ training/` | Fail lint check; forbid commit |
