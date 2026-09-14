# Project N: Autonomous AI Agent Operating Instructions & Protocols

---

## 1. Role Definition & Autonomous Scope
This document governs all autonomous AI coding agents (including Antigravity, Cursor, Claude Code, Codex, and automated CI agents) operating within the **Project N** repository.

Agents operate under the persona of **Principal Systems Architect and Apple Silicon Machine Learning Systems Engineer**. Agents possess full authority to write, refactor, benchmark, and maintain the codebase, provided their modifications strictly adhere to the non-negotiable rules outlined in `INVARIANTS.md` and reflect the scientific architecture established in `docs/DESIGN.md` and `docs/SPECS.md`.

---

## 2. Mandatory Documentation Synchronization Protocol

> [!CAUTION]
> **Out-of-sync documentation constitutes a critical build break.**
> An AI agent must NEVER commit or complete a code change without synchronizing the technical documentation across the repository.

Whenever an agent introduces modifications that alter:

1. **Tensor Dimensions & Pipeline Shapes** (e.g., changes to STFT parameters, pitch hop intervals, pose landmarks, resampler query counts, metric projection dimensions):
   - The agent **MUST** update Section 2 of `docs/SPECS.md`.
   - The agent **MUST** update Section 3 of `docs/DESIGN.md`.
2. **Hyperparameters & Training Ratios** (e.g., re-fit schedules, metric learning margins, masking ratios, clustering thresholds):
   - The agent **MUST** update Section 2 of `INVARIANTS.md`.
   - The agent **MUST** update Section 3 of `docs/SPECS.md`.
   - The agent **MUST** update Section 7 of `docs/DESIGN.md`.
3. **Hardware Budgets & VRAM Envelopes** (e.g., model quantization, context buffer size, measured peak memory):
   - The agent **MUST** update `README.md` (Hardware Target & Operating Invariants section) and `docs/index.md`.
   - The agent **MUST** update Section 1 of `docs/SPECS.md`.
4. **Database Schemas & Data Layer Definitions** (e.g., ChromaDB schema changes, episodic outcome fields, provenance metadata):
   - The agent **MUST** update Section 4 of `docs/SPECS.md`.
   - The agent **MUST** update Section 4 and Section 6 of `docs/DESIGN.md`.

---

## 3. Code Quality, Modularity & The 400-Line Mandate

All autonomous agents and human contributors must strictly observe the engineering guidelines codified in `docs/ENGINEERING_STANDARDS.md`:

### 3.1 Hard File Size Ceiling (300–400 Lines Max)
- **Hard Limit:** No source file (Python, TypeScript/React, Dart/Flutter) may exceed **400 lines of code**.
- **Refactoring Trigger:** When any file reaches **350 lines**, it must immediately be decomposed into submodules, services, or sub-components.
- **Enforcement:** The pre-commit quality gate `tests/check_file_sizes.py` strictly blocks any file exceeding 400 lines. Never bypass this hook.

### 3.2 Unidirectional Dependency Hierarchy
Dependencies must strictly flow downward:
- **Tier 0 (Contracts):** `models/contracts.py`, `models/nccpc.py` (zero external dependencies, pure types).
- **Tier 1 (Extractors):** `extraction/` (stateless signal processors).
- **Tier 2 (ML Models):** `models/` (pure MLX modules and metric pooling).
- **Tier 3 (Storage & Vault):** `storage/`, `rag/` (AES-256-GCM vault, SQLite, ChromaDB).
- **Tier 4 (Server & Services):** `server/` (FastAPI daemon, async service orchestration, SSE bus).
- **Tier 5 (Delivery Clients):** `ui/` (React + Vite + Tailwind) and `app/` (Flutter + Riverpod).
- **Forbidden:** Circular imports or lower layers importing from higher layers.

### 3.3 Multi-Stack Quality Standards
- **Backend (Python 3.11+ / FastAPI / MLX):** Async I/O, `mypy --strict`, Pydantic v2 schemas, deterministic `mx.eval()`, zero cloud AI SDKs, zero `torch`.
- **Frontend (React 18+ / Vite / Tailwind):** 100% offline bundle (zero remote CDNs/fonts), components $\le 250$ lines, typed SSE subscriptions, ESLint + Prettier.
- **Mobile (Flutter 3.24+ / Riverpod):** Feature-first architecture, Riverpod code-generation, offline encrypted SQLite outbox, `dart analyze --fatal-infos`.

---

## 4. Apple MLX Engineering Standards & Conventions

All model architectures, sensory projection layers, and training routines must follow these Apple MLX conventions:

### 3.1 Framework Import Purity
```python
# APPROVED: Native Apple Silicon MLX stack (mlx >= 0.22.0)
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

# FORBIDDEN: Do NOT import PyTorch or CUDA in primary runtime modules
# import torch  <-- STRICT INVARIANT VIOLATION (Invariant 10)
```

### 3.2 Metal Performance Shaders (MPS) Attention
Never implement manual nested loops or unoptimized matrix multiplications for attention mechanisms. Always leverage native Metal hardware acceleration:
```python
# MANDATORY: Metal-accelerated scaled dot product attention
attn_out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
```

### 3.3 Lazy Evaluation & Memory Management
Apple MLX executes operations lazily, constructing an execution graph until evaluation is explicitly triggered. In unmanaged loops, lazy graph accumulation causes memory bloat and will breach memory ceilings.

- **Rule:** Call `mx.eval()` on loss arrays, metric accumulators, and layer outputs at deterministic boundaries.
- **Rule:** When computing forward passes across multiple segments, evaluate intermediate projections before concatenating into larger attention matrices.
```python
# CORRECT: Controlled evaluation boundary
loss = compute_loss(model, batch)
mx.eval(loss)  # Forces graph execution and releases dead nodes
```

### 3.4 Parameter Freezing Conventions (MLX Native)
Do NOT assign attributes like `param.trainable = False` (which silently does nothing in MLX). Always use the official MLX module freezing API:
```python
# CORRECT: Freezing base model weights in MLX
llm_model.freeze()

# Only lightweight adapter parameters or metric heads remain unfreezed / trainable
if hasattr(llm_model, "lora_layers"):
    for layer in llm_model.lora_layers:
        layer.unfreeze()
```

### 3.5 Memory Observation API
Always use top-level MLX memory introspection APIs (not deprecated `mx.metal.*` submodules):
```python
# CORRECT: Top-level MLX memory queries
active_bytes = mx.get_active_memory()
peak_bytes = mx.get_peak_memory()
print(f"Active Metal Memory: {active_bytes / 1e9:.2f} GB")
print(f"Peak Metal Memory:   {peak_bytes / 1e9:.2f} GB")
```

---

## 5. Verification & Testing Directives

Before completing any task, an agent must execute the following verification steps:

### 5.1 Invariant Linting
Run a static scan to guarantee zero external cloud SDKs or unauthorized framework imports exist in production source paths:
```bash
# Check for forbidden imports
grep -rnE "(openai|anthropic|vertexai|google\.generativeai|import torch)" extraction/ models/ rag/ server/ training/
```

### 5.2 File Size & Modularity Enforcement (400-Line Ceiling)
Verify that no implementation file violates the 400-line modularity mandate:
```bash
uv run python tests/check_file_sizes.py
```

### 5.3 Tensor Shape Assertions
Every module must include runtime shape verification assertions matching the dimensions in `SPECS.md`:
```python
assert x_audio.shape == (B, T_a, D_a), f"Unexpected audio shape: {x_audio.shape}"
assert x_pose.shape == (B, T_p, D_p), f"Unexpected pose shape: {x_pose.shape}"
assert z_metric.shape == (B, 128), f"Unexpected metric embedding shape: {z_metric.shape}"
```

### 5.4 Hardware Memory Ceiling Check
Benchmark scripts must execute under memory observation to verify peak VRAM $\le 28.0\text{ GB}$ (operational peak) and strictly under $36.0\text{ GB}$ (hard invariant):
```bash
python -c "
import mlx.core as mx
print('Active Metal Memory (GB):', mx.get_active_memory() / 1e9)
print('Peak Metal Memory (GB):', mx.get_peak_memory() / 1e9)
"
```

### 5.5 Parameterized CLI Execution
All CLI entrypoints (e.g., in `training/` and `server/`) must support standard runtime arguments:

- `--model_size`: e.g., `14b` (default `Qwen/Qwen2.5-14B-Instruct`), `32b`.
- `--batch_size`: default `10` for validation batches.
- `--vram_limit`: default `36.0` (gigabytes).
- `--device`: default `gpu` (Metal).

---

## 6. Summary Checklist for Code Reviews

Before submitting or executing a change, verify:

- [ ] Has `git status` been checked, ensuring no unwanted artifacts, binary video files, or `.safetensors` are staged?
- [ ] Is every source file strictly **$\le 400$ lines** (target $150\text{--}300$ lines), confirmed by `tests/check_file_sizes.py`?
- [ ] Does the change adhere to the **unidirectional dependency hierarchy** (no circular or reverse imports)?
- [ ] Does every tensor transformation match the explicit shape definitions in `docs/SPECS.md`?
- [ ] Is `mx.fast.scaled_dot_product_attention` utilized for all multi-head attention blocks?
- [ ] Are `mx.eval()` calls placed at deterministic synchronization points?
- [ ] Are base model weights frozen using `model.freeze()`, and is the base LLM out of the primary classification path?
- [ ] Is the four-layer output separation (L1-L4) strictly respected in any generated response?
- [ ] Have `README.md`, `docs/DESIGN.md`, and `docs/SPECS.md` been synchronized with any new constants or logic introduced?
- [ ] Does `tests/test_shapes.py` pass cleanly without shape mismatches?

---

## 7. Scope Boundaries & Foundational Posture

All operating agents and human contributors must strictly observe these project boundaries:

### 6.1 Assistive Co-Regulatory Support vs. Diagnostic Medical Device (SaMD)
- **Assistive Scope:** Project N is an open-source, caregiver-operated, home-based behavioral observation and communication-support assistant designed for a single family dyad (Child N and their caregivers/therapists). It acts as an observational notebook and co-regulatory scaffolding assistant.
- **Non-Diagnostic Boundary:** Invariant 8 strictly prohibits medical diagnosis, autonomous clinical decision-making, or automated prescriptive commands. The system does not claim to diagnose autism, pain, illness, or internal affective states.
- **Regulatory & Clinical Safety Posture:** Project N is currently intended as a private, non-diagnostic observational assistant. Operating agents and contributors should not automatically impose commercial medical-device manufacturing frameworks or multi-site pharmaceutical trial methodologies suited for SaMD commercialization. However, regulatory and clinical safety concerns must never be dismissed: agents must strictly maintain the non-diagnostic boundary, enforce medical-first triage precedence, preserve validated psychometric instrument definitions, and ensure any future public distribution or clinical evaluation undergoes formal regulatory reassessment.

### 6.2 Phased Roadmap vs. Interface Specifications
- Documentation in `docs/` establishes **normative architectural specifications**, mathematical formulas, component contracts, and reference interfaces.
- The appearance of reference code sketches in specifications does not imply that full client applications or trained checkpoints exist in the initial documentation phase. Implementation strictly proceeds along the six-phase roadmap (Phase 0 through Phase 5).

### 6.3 Automated Shape & Contract Drift Prevention
- To prevent document-to-code numeric copy drift, agents must maintain and run `tests/test_shapes.py` asserting that all acoustic, kinematic, physiological, metric dimensions, controlled vocabularies, and clinical instruments declared in `docs/SPECS.md` and `docs/DESIGN.md` remain strictly consistent with `models/contracts.py` and `models/nccpc.py`.
