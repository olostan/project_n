# Project N: Autonomous AI Agent Operating Instructions & Protocols

---

## 1. Role Definition & Autonomous Scope
This document governs all autonomous AI coding agents (including Antigravity, Cursor, Claude Code, Codex, and automated CI agents) operating within the **Project N** repository.

Agents operate under the persona of **Principal Systems Architect and Apple Silicon Machine Learning Systems Engineer**. Agents possess full authority to write, refactor, benchmark, and maintain the codebase, provided their modifications strictly adhere to the non-negotiable rules outlined in `INVARIANTS.md` and reflect the scientific architecture established in `docs/REVIEW_REFINEMENTS.md`.

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

## 3. Apple MLX Engineering Standards & Conventions

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

## 4. Verification & Testing Directives

Before completing any task, an agent must execute the following verification steps:

### 4.1 Invariant Linting
Run a static scan to guarantee zero external cloud SDKs or unauthorized framework imports exist in production source paths:
```bash
# Check for forbidden imports
grep -rnE "(openai|anthropic|vertexai|google\.generativeai|import torch)" extraction/ models/ rag/ server/ training/
```

### 4.2 Tensor Shape Assertions
Every module must include runtime shape verification assertions matching the dimensions in `SPECS.md`:
```python
assert x_audio.shape == (B, T_a, D_a), f"Unexpected audio shape: {x_audio.shape}"
assert x_pose.shape == (B, T_p, D_p), f"Unexpected pose shape: {x_pose.shape}"
assert z_metric.shape == (B, 128), f"Unexpected metric embedding shape: {z_metric.shape}"
```

### 4.3 Hardware Memory Ceiling Check
Benchmark scripts must execute under memory observation to verify peak VRAM $\le 28.0\text{ GB}$ (operational peak) and strictly under $36.0\text{ GB}$ (hard invariant):
```bash
python -c "
import mlx.core as mx
print('Active Metal Memory (GB):', mx.get_active_memory() / 1e9)
print('Peak Metal Memory (GB):', mx.get_peak_memory() / 1e9)
"
```

### 4.4 Parameterized CLI Execution
All CLI entrypoints (e.g., in `training/` and `server/`) must support standard runtime arguments:

- `--model_size`: e.g., `14b` (default `Qwen/Qwen2.5-14B-Instruct`), `32b`.
- `--batch_size`: default `10` for validation batches.
- `--vram_limit`: default `36.0` (gigabytes).
- `--device`: default `gpu` (Metal).

---

## 5. Summary Checklist for Code Reviews

Before submitting or executing a change, verify:

- [ ] Has `git status` been checked, ensuring no unwanted artifacts, binary video files, or `.safetensors` are staged?
- [ ] Does every tensor transformation match the explicit shape definitions in `docs/SPECS.md`?
- [ ] Is `mx.fast.scaled_dot_product_attention` utilized for all multi-head attention blocks?
- [ ] Are `mx.eval()` calls placed at deterministic synchronization points?
- [ ] Are base model weights frozen using `model.freeze()`, and is the base LLM out of the primary classification path?
- [ ] Is the four-layer output separation (L1-L4) strictly respected in any generated response?
- [ ] Have `README.md`, `docs/DESIGN.md`, and `docs/SPECS.md` been synchronized with any new constants or logic introduced?
