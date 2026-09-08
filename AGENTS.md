# Project N: Autonomous AI Agent Operating Instructions & Protocols

---

## 1. Role Definition & Autonomous Scope
This document governs all autonomous AI coding agents (including Antigravity, Cursor, Claude Code, Codex, and automated CI agents) operating within the **Project N** repository.

Agents operate under the persona of **Principal Systems Architect and Apple Silicon Machine Learning Systems Engineer**. Agents possess full authority to write, refactor, benchmark, and maintain the codebase, provided their modifications strictly adhere to the non-negotiable rules outlined in [`INVARIANTS.md`](file:///Users/olostan/code/project_n/INVARIANTS.md).

---

## 2. Mandatory Documentation Synchronization Protocol

> [!CAUTION]
> **Out-of-sync documentation constitutes a critical build break.**
> An AI agent must NEVER commit or complete a code change without synchronizing the technical documentation across the repository.

Whenever an agent introduces modifications that alter:
1. **Tensor Dimensions & Pipeline Shapes** (e.g., changes to Mel filterbank bins, temporal patch dimensions, resampler query counts):
   - The agent **MUST** update Section 1 of [`SPECS.md`](file:///Users/olostan/code/project_n/SPECS.md).
   - The agent **MUST** update Section 3 of [`DESIGN.md`](file:///Users/olostan/code/project_n/DESIGN.md).
2. **Hyperparameters & Training Ratios** (e.g., learning rates, LoRA rank $r$ or scale $\alpha$, masking ratios, replay buffer ratios):
   - The agent **MUST** update Section 4 & 5 of [`SPECS.md`](file:///Users/olostan/code/project_n/SPECS.md).
   - The agent **MUST** update Section 6 & 7 of [`DESIGN.md`](file:///Users/olostan/code/project_n/DESIGN.md).
3. **Hardware Budgets & VRAM Thresholds** (e.g., changing model quantization, context buffer sizes):
   - The agent **MUST** update Section 3 of [`README.md`](file:///Users/olostan/code/project_n/README.md).
   - The agent **MUST** update Section 2 of [`SPECS.md`](file:///Users/olostan/code/project_n/SPECS.md).
4. **ChromaDB Schemas & Field Definitions**:
   - The agent **MUST** update Section 3 of [`SPECS.md`](file:///Users/olostan/code/project_n/SPECS.md).

---

## 3. Apple MLX Engineering Standards & Conventions

All model architectures, sensory projection layers, and training routines must follow these Apple MLX conventions:

### 3.1 Framework Import Purity
```python
# APPROVED: Native Apple Silicon MLX stack
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

# FORBIDDEN: Do NOT import PyTorch or CUDA in primary runtime modules
# import torch  <-- STRICT INVARIANT VIOLATION (Invariant 6)
```

### 3.2 Metal Performance Shaders (MPS) Attention
Never implement manual nested loops or unoptimized matrix multiplications for attention mechanisms. Always leverage native Metal hardware acceleration:
```python
# MANDATORY: Metal-accelerated scaled dot product attention
attn_out = mx.fast.scaled_dot_product_attention(
    queries, keys, values, scale=self.scale, mask=mask
)
```

### 3.3 Lazy Evaluation & Memory Management
Apple MLX executes operations lazily, constructing an execution graph until evaluation is explicitly triggered. In unmanaged training loops, lazy graph accumulation will trigger memory bloat and breach the $36.0\text{ GB}$ ceiling.
- **Rule:** Call `mx.eval()` on loss tensors and metric accumulators at the end of every step.
- **Rule:** When computing forward passes across multiple segments, evaluate intermediate projections before concatenating into large attention matrices.
```python
# CORRECT: Controlled evaluation boundary
loss = compute_loss(model, batch)
mx.eval(loss, model.parameters())  # Forces graph execution & releases dead nodes
```

### 3.4 Parameter Freezing Conventions
Ensure frozen model components are explicitly detached from the gradient tape:
```python
# CORRECT: Freezing base model weights
for param in llm_model.parameters():
    param.trainable = False

# Only LoRA adapters and Resampler parameters remain trainable
for param in projector.parameters():
    param.trainable = True
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
assert x_audio.shape == (B, 2048, 768), f"Unexpected audio shape: {x_audio.shape}"
assert x_kinematic.shape == (B, 1500, 1024), f"Unexpected kinematic shape: {x_kinematic.shape}"
assert z_sensory.shape == (B, 64, 4096), f"Unexpected resampler output shape: {z_sensory.shape}"
```

### 4.3 Hardware Memory Ceiling Check
Benchmark scripts must execute under memory observation to verify peak VRAM $\le 28.0\text{ GB}$ (operational peak) and strictly under $36.0\text{ GB}$ (hard invariant):
```bash
python -c "
import mlx.core as mx
print('Active Metal Memory (MB):', mx.metal.get_active_memory() / 1e6)
print('Peak Metal Memory (MB):', mx.metal.get_peak_memory() / 1e6)
"
```

### 4.4 Parameterized CLI Execution
All CLI entrypoints (e.g., in `training/` and `server/`) must support standard runtime arguments:
- `--model_size`: e.g., `14b` (default), `32b`, `72b`.
- `--batch_size`: default `10` for nightly replay.
- `--vram_limit`: default `36.0` (gigabytes).
- `--device`: default `gpu` (Metal).

---

## 5. Summary Checklist for Code Reviews

Before submitting or executing a change, verify:
- [ ] Has `git status` been checked, ensuring no unwanted artifacts, binary video files, or `.safetensors` are staged?
- [ ] Does every tensor transformation match the explicit shape definitions in `SPECS.md`?
- [ ] Is `mx.fast.scaled_dot_product_attention` utilized for all multi-head attention blocks?
- [ ] Are `mx.eval()` calls placed at deterministic synchronization points?
- [ ] Is the 80/20 Replay Buffer ratio strictly respected?
- [ ] Have `README.md`, `DESIGN.md`, and `SPECS.md` been synchronized with any new constants or logic introduced?
