# Project N: Architectural & System Invariants

---

## Preamble
This document establishes the system, architectural, mathematical, security, and governance boundaries governing **Project N**. Every contributor, automated script, and autonomous AI coding agent (e.g., Antigravity, Cursor, Claude Code) operating on this codebase is strictly bound by these rules.

Following the comprehensive architectural review (`docs/REVIEW_REFINEMENTS.md`), this document makes an explicit distinction between:

1. **True Non-Negotiable Invariants (§1):** Hard system, safety, privacy, and architectural constraints that must never be violated.
2. **Tunable Empirical Defaults (§2):** Research and training hyperparameters that are explicitly configurable and subject to experimental ablation.

---

## 1. True Non-Negotiable Invariants

### Invariant 1: Privacy, Offline Execution & Encryption Vault
**Zero external network telemetry. All media, features, embeddings, and clinical records MUST process strictly offline on local hardware.**

- **No Cloud AI APIs:** Inclusion or invocation of remote cloud AI APIs (OpenAI, Anthropic, Google Cloud Vertex/Gemini, AWS Bedrock, HuggingFace Inference API, or any remote telemetry collector) is strictly prohibited across all modules.
- **Zero-Cloud Dashboard:** The local Caregiver Web Dashboard (React + Tailwind) must bundle all JavaScript, CSS, font, and asset dependencies locally. Loading remote CDNs or external web resources is forbidden.
- **Two-Key Vault Posture:** 
  - Media, features, and metadata are encrypted at rest using per-clip random AES-256-GCM Data Encryption Keys (DEKs).
  - Background ingestion and processing while the Mac screen is locked are executed via a signed per-user LaunchAgent helper using Apple's Data Protection Keychain (`SecItem` with `kSecUseDataProtectionKeychain=true`).
  - Sensitive operations (revealing raw video, exporting data, viewing timelines, altering retention, pairing new devices) require explicit Touch ID / user-presence authentication via a separate private review key.
- **Transport & Mobile Outbox Security:** 
  - Companion apps (Flutter Android & iOS) communicate with the Mac helper over mutual TLS (mTLS) with per-request signatures and nonces over local Wi-Fi. mDNS discovery or LAN IP address presence alone is never treated as authentication.
  - Device credentials must reside in hardware-backed storage: **Android Keystore** on Android devices and **iOS Keychain** on iOS devices (never in application preferences or shared storage).
  - Clips captured offline away from home (playgrounds, outdoor parks, OT clinic sessions) are encrypted locally using AES-256 in the client SQLite outbox before being flushed to the Mac.
- **Child Assent & Dissent:** The capture system must respect behavioral dissent (e.g., turning away or covering the camera immediately terminates recording). Camera-free private zones (bathrooms, bedrooms) are enforced in software. Untagged footage is automatically purged after a configurable retention window.
- **Child Identity & Data Separation:** The project documentation transparently acknowledges the personal parental motivation and dedicated origin for Nolan Shybanov (7-year-old non-verbal autistic child). However, at the engineering, runtime, data storage, and modeling layers, strict pseudonymization and health data protection are enforced: all runtime biometric databases, local SQLite outboxes, media encryption containers, inference logs, and benchmark datasets strictly refer to **Child N** and contain zero identifiable personal health information (PHI). Under no circumstances are raw video/audio recordings, facial embeddings, or clinical diagnostic records committed to Git or transmitted over external networks.

### Invariant 2: Measured Hardware Memory Ceiling
**Peak unified memory allocation must NEVER cause page swapping to SSD and must remain bounded by a measured ceiling on the target hardware.**

- On 48 GB Unified Memory systems, peak operational allocation must remain strictly $\le 36.0\text{ GB}$, with an operational baseline $\le 28.0\text{ GB}$, preserving $\ge 12.0\text{ GB}$ ($48\text{ GB} - 36\text{ GB}$) for macOS system stability.
- Memory budgets must reflect measured benchmark envelopes (active Metal allocations, dynamic cache, context window, and model weights) rather than unverified static assertions.

### Invariant 3: Base LLM Weight Freezing
**Base Large Language Model weights ($\mathbf{W}_0$) must remain 100% frozen in memory. Weight updates are strictly confined to lightweight connectors, adapters, or metric heads.**

- In Apple MLX, base model freezing must be enacted via `model.freeze()` (never no-op attribute mutations like `param.trainable = False`).
- Base LLM weights are loaded in quantized format directly into Metal unified memory. De-quantization into FP16/FP32 for base weight fine-tuning is prohibited.

### Invariant 4: Sensory Bypass & Inductive Bias
**Phonemic speech-to-text transcriptions (e.g., Whisper text tokens, CTC phoneme decoding) and text-only intermediate captioning bottlenecks are strictly prohibited in the primary sensory decoding path.**

- **Acoustic Front End:** The system must preserve raw acoustic dynamics. Explicit pitch tracking (F0), voice quality metrics (jitter, shimmer, HNR, spectral tilt), harmonic filterbanks (CQT/ERB), and broadband log-mel representations are permitted and required. Pitch tracking is an acoustic measurement, not an ASR phoneme decoder.
- **Pretrained Encoders as Experimental Arms:** Pretrained acoustic representations (e.g., BEATs, AudioMAE, or frozen intermediate encoder layers) and pretrained vision backbones may be evaluated as competing experimental arms against custom-trained encoders. The ban applies to *phonemic transcription and text-only intermediate bottlenecks*, not to pretrained acoustic representations.
- **Kinematic Extraction:** Pixel differencing alone must not be relied upon due to camera motion vulnerability. Pose/keypoint landmarks (e.g., MediaPipe Holistic, BlazePose) form the primary motion substrate, complemented by optical flow and context.
- **Optional Physiological Sensing:** Wearable physiological telemetry (EDA, HRV) is an optional auxiliary channel. The metric learning head must support graceful degradation (modal masking) when sensors are absent.

### Invariant 5: No Autonomous Model Deployment (Gated Promotion)
**No model checkpoint or adapter may be deployed to caregiver-facing inference automatically or via unverified overnight gradient updates.**

- Continuous adaptation requires an explicit promotion gate conforming to [`evaluation_protocol.md`](evaluation_protocol.md):

  1. Candidate models are trained on accumulated, verified records.
  2. Candidates are evaluated against time-separated and context-stratified holdout sets plus a locked "never-train" safety set.
  3. Predeclared criteria must be satisfied: improved calibration error, coverage, and per-class precision/recall, with **zero safety regressions**.
  4. Explicit caregiver (and clinical team) review and sign-off are required prior to promoting candidate weights to production.
  5. Full rollback capability, model lineage, and parameter versioning are maintained.

### Invariant 6: Output Truthfulness, Non-Diagnostic Framing & Four-Layer Separation
**The system is a caregiver- and child-controlled communication-support tool, NOT a diagnostic device or intent translator. It must never present an inferred state, pain assessment, or intent as fact.**

- **Four-Layer Traceability:** Every system output must be explicitly partitioned into four visually distinct layers:
  - **Layer 1 (L1 - Measured Observation):** Directly measured acoustic, kinematic, and physiological features (e.g., vocalization duration, F0 mean/variance, motion periodicity).
  - **Layer 2 (L2 - Comparable History):** Historical episodes from Child N's verified records exhibiting similar metric embeddings and their recorded outcomes.
  - **Layer 3 (L3 - Context & Antecedents):** Caregiver-provided notes regarding transitions, environment, timing, and caregiver-observed antecedents.
  - **Layer 4 (L4 - Evidence Library):** Versioned, cited excerpts from published literature, documenting author, year, study population, and evidence level.
- **Abstention as a First-Class State:** If nearest-neighbor distance in metric space exceeds a calibrated threshold or if signal quality is compromised, the system must **abstain** ("unrecognized pattern") and suggest observational or AAC-based exploratory options.
- **Forbidden Terminology:** Outputs and documentation must not use deterministic terms such as "child state diagnosis", "translating into intent", or "caregiver treatment protocols".

### Invariant 7: Medical Safety Protocol & Human-Controlled Triage Priority
**Potential physical pain and medical emergencies must always take priority over behavioral and sensory interpretations. Automated models must never attempt to diagnose medical etiology or pretend to score multi-item clinical instruments from brief video clips.**

- **Automated Acute Distress Screener:** The automated pipeline inspects acoustic and kinematic features strictly to detect *acute signal anomalies* (e.g., extreme pitch excursion exceeding personal baseline limits, sustained high-amplitude crying, or acute physical flinching). When an anomaly is detected, the system does not diagnose a medical condition; it presents an immediate prompt: *"Acute distress anomaly detected. Please perform your family's pediatrician-approved physical safety and comfort check."*
- **Human-Controlled Clinical Pain Screening:** Standardized pain observation instruments (such as the Non-Communicating Children's Pain Checklist, NCCPC-PV [Breau et al., 2002, *Anesthesiology*, doi:10.1097/00000542-200203000-00007; cut-off $\ge 11$ for moderate-to-severe postoperative pain] or outpatient NCCPC-R [Breau et al., 2002, *Pain*, cut-off $\ge 6$ for pain]) require 10 minutes of direct caregiver observation across 27 items. If the caregiver suspects somatic pain, they complete this checklist manually in the app. Automated models must never synthesize artificial checklist scores or claim automated clinical validation.
- **Priority over Behavioral Inference:** When acute physical distress or a caregiver-reported medical red flag is active, behavioral pattern matching and sensory hypotheses are paused or explicitly subordinated to physical comfort and medical consultation.
- **Non-Diagnostic Boundary:** The system must never claim to identify specific medical causes (e.g., ear infection, dental abscess, GI reflux) or issue medical diagnoses. All health escalations are directed to the child's pediatrician or emergency medical services.

### Invariant 8: Caregiver & Therapist Insight Priority & Child Agency
**The core purpose of Project N is to support parents and therapists in noticing observable behavioral patterns, reviewing historical co-regulatory precedents, and exploring practical, low-risk scaffolding strategies. The system does not act as an automated AAC generator, nor does it claim access to the child's internal mental state.**

- **Dual-Perspective Output:** The primary deliverable is an accessible four-layer analysis: measured acoustic/kinematic patterns (L1), historical precedents and observed associations (L2), situational context (L3), and evidence-based exploratory possibilities (L4), switchable between **Parent View** (warm, everyday English co-regulatory ideas) and **Therapist View** (full bioacoustic and sensory telemetry).
- **Epistemic Humility:** Outputs must explicitly distinguish between direct physical observations and exploratory hypotheses. The system never claims to "know what the child feels" or declare causal certainty.
- **Independent Child Communication:** When the child communicates via an independent external AAC speech device, communication board, or intentional motor gestures (reach, push away), that child-authored bid serves as primary observational ground truth. Project N does not generate or dispatch synthetic AAC vocabulary as its core function.

### Invariant 9: Lineage, Versioning & Evaluation Set Immutability
**Training data, retrieval indices, and evaluation benchmarks must be strictly segregated.**

- Every stored embedding must record the exact encoder checkpoint ID that generated it. When encoders are updated, retrieval indices must be re-embedded from a versioned corpus snapshot to prevent silent embedding drift.
- A locked, immutable benchmark evaluation set and safety set must remain isolated from mutable daily training buffers.

### Invariant 10: Apple MLX Native Framework Purity
**Core tensor transformations, forward passes, and training loops must execute natively on Apple MLX (`mlx.core`, `mlx.nn`).**

- Use `mlx.core` and `mlx.nn`. Metal Performance Shaders attention must be called via `mx.fast.scaled_dot_product_attention`.
- Memory tracking must use `mx.get_active_memory()` and `mx.get_peak_memory()`. Model freezing must use `model.freeze()`.
- Standard CPU-bound demuxing, video decoding, and extraction libraries (`numpy`, `scipy`, `soundfile`, `librosa`, `opencv-python`, and `mediapipe`) are permitted for media ingest, pose landmark extraction, and DSP feature preparation prior to MLX array conversion. All neural network forward passes, attention blocks, loss functions, and metric projection heads must execute natively on Apple MLX without PyTorch or CUDA dependencies.

---

## 2. Tunable Empirical Defaults (Configurable Hyperparameters)

The following parameters are empirical design defaults subject to systematic ablation and optimization, not rigid invariants:

| Parameter | Baseline Default | Ablation / Search Range | Notes |
| :--- | :--- | :--- | :--- |
| **Replay Buffer Ratio** | 80% historical / 20% novel | 50/50 to 90/10, or full-history re-fit | Replaced nightly SGD with periodic full re-fit of metric heads. |
| **Masking Ratio** | 75% uniform | 50% to 85% | Pre-training ablation parameter for self-supervised encoders. |
| **Resampler Latent Count** | 64 query tokens | 16, 32, 64, 128 tokens | Evaluated against cross-modal retrieval fidelity and memory. |
| **LoRA Rank & Scale** | $r=64, \alpha=128$ | $r \in [16, 32, 64], \alpha = 2r$ | Confined strictly to output rendering and schema alignment. |
| **Metric Embedding Dimension** | 128 dimensions | 64 to 256 dimensions | Compact metric space for prototype/k-NN retrieval. |
| **Triplet / Contrastive Margin** | $\alpha_{margin} = 0.25$ | 0.1 to 0.5 | Evaluated with supervised contrastive / prototypical loss. |
| **Capture Frame Rate & Res** | 30 fps @ 720p | 15 fps to 30 fps, 720p/1080p | 30 fps provides Nyquist coverage for 3–6 Hz motor stims; sequence downsampling (Mondal & Washington 2026) reduces redundant compute. |
| **Audio STFT Window & Hop** | $N=2048$, $H=160$ (48 kHz) | $N \in [1024, 2048], H \in [160, 240]$ | Accompanied by 10 ms hop F0/voice quality tracking. |
| **Abstention Distance Cutoff** | 95th percentile holdout | Calibrated per-class threshold | Governs coverage vs error rate trade-off. |

---

## 3. Enforcement & Verification Summary

| Invariant ID | Target Subsystem | Automated Verification Command | Action on Failure |
| :--- | :--- | :--- | :--- |
| **INV-1** | Network Sandbox | `pytest tests/test_offline_sandbox.py` | Immediate process abort; reject PR |
| **INV-2** | Memory Ceiling | `python tests/verify_memory_ceiling.py --max_gb 36.0` | Abort execution; release Metal cache |
| **INV-3** | Weight Freezing | `python tests/verify_frozen_weights.py` | Refuse candidate promotion |
| **INV-4** | Sensory Bypass | `pytest tests/test_sensory_pipeline.py` | Reject phonemic/text bottlenecks |
| **INV-5** | Model Promotion | `python -m training.evaluate_candidate --strict` | Block deployment if safety regresses |
| **INV-6** | Output Traceability | `pytest tests/test_output_schema.py` | Reject unformatted or diagnostic text |
| **INV-7** | Medical Rule-Out | `pytest tests/test_nccpc_escalation.py` | Enforce medical referral on red flags |
| **INV-8** | AAC Authorship | `pytest tests/test_aac_routing.py` | Ensure candidate choices route to child |
| **INV-9** | Version Lineage | `python -m rag.verify_lineage` | Prevent querying across encoder versions |
| **INV-10**| MLX Purity | `grep -rn "import torch" extraction/ models/ training/` | Fail lint check; forbid commit |
