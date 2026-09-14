# Contributing to Project N

Thank you for your interest in contributing to **Project N**!

Project N is an open-source, caregiver-operated, home-based behavioral observation and co-regulatory scaffolding assistant designed for minimally verbal and non-speaking autistic individuals (specifically designed around Child N and their family dyad). The system runs 100% locally on Apple Silicon hardware with zero cloud telemetry.

---

## 1. Guiding Principles & Scope Boundaries

Before contributing, please review our core architectural documents:
- [`INVARIANTS.md`](INVARIANTS.md): Non-negotiable system, privacy, and clinical guardrails.
- [`AGENTS.md`](AGENTS.md): Engineering standards and documentation synchronization protocols.
- **Technical Specifications (`docs/SPECS.md`):** Complete mathematical definitions, tensor shapes, and component interfaces.
- **Theoretical Design (`docs/DESIGN.md`):** Detailed neurobiology, acoustic physics, and continuous adaptation architecture.
- **Scientific Whitepaper (`docs/WHITE_PAPER.md`):** Clinical foundations for SLPs, OTs, and autism researchers.

### Assistive Co-Regulatory Support vs. Diagnostic Medical Device
- **Assistive Scope:** Project N serves as an observational notebook and co-regulatory scaffolding assistant. It aims to help caregivers and therapists notice patterns, track empirical settle rates following supportive actions, and ground possibilities in validated research.
- **Non-Diagnostic Boundary (Invariant 8):** The system strictly prohibits automated medical diagnosis, clinical decision-making, or prescriptive commands. It does not diagnose pain, illness, or autism.
- **Human Triage Precedence (Invariant 7):** Caregiver observations, physical comfort checks, and clinical red flags always take precedence over automated sensor inference.

---

## 2. Who Can Contribute?

We welcome contributions across several domains:

1. **Speech-Language Pathologists (SLPs) & Occupational Therapists (OTs):**
   - Refinements to evidence-based sensory strategies and co-regulatory routines.
   - Enhancements to AAC multimodal integration and transactional communication modeling (SCERTS framework).
   - Validation of observational instruments and dyadic interaction protocols.

2. **Neurodivergent Advocates & Family Caregivers:**
   - Feedback on neurodiversity-affirming terminology, child dignity, and assent/dissent modeling.
   - Usability feedback on the Parent View and caregiver dashboard interfaces.

3. **Apple Silicon & Systems ML Engineers:**
   - Optimizations for native Apple MLX (`mlx>=0.22.0`) and Metal Performance Shaders (MPS).
   - High-performance bioacoustic (F0, CQT) and kinematic (pose, optical flow) feature extractors.
   - Offline SQLite/SQLCipher and ChromaDB vector retrieval pipelines.

---

## 3. Engineering & Code Standards

All code contributions must adhere to our automated quality gates:

### 3.1 Framework Purity (Invariant 10)
- **Approved:** Native Apple MLX (`import mlx.core as mx`, `import mlx.nn as nn`).
- **Forbidden:** No cloud AI SDKs (`openai`, `anthropic`, `google.generativeai`, `vertexai`) and no PyTorch/CUDA in runtime modules.

### 3.2 Offline Privacy Boundary (Invariant 1)
- Zero external network telemetry. All media processing, vector search, and LLM inference must execute strictly offline on local hardware.
- Encrypted storage at rest using AES-256-GCM and SQLCipher.

### 3.3 Documentation Synchronization Protocol
- Any PR that modifies tensor dimensions, hyperparameters, schemas, or API endpoints **must** update the corresponding sections in `docs/SPECS.md`, `docs/DESIGN.md`, and `models/contracts.py`.

---

## 4. Local Development & Pre-Commit Verification

### Setup
```bash
# Clone the repository
git clone https://github.com/olostan/project_n.git
cd project_n

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and setup pre-commit hooks
uv sync --extra dev
uv run pre-commit install
```

### Running Test Suite
Before submitting a pull request, ensure all local verification checks pass:

```bash
# 1. Run architecture shape and contract assertions
uv run python tests/test_shapes.py

# 2. Verify all relative documentation markdown links
uv run python tests/check_markdown_links.py

# 3. Verify academic citation DOIs against Crossref/DataCite
uv run python tests/verify_citations.py

# 4. Run Ruff linter and code formatter
uv run ruff check .
uv run ruff format --check .

# 5. Run Mypy strict type checker
uv run mypy models tests

# 6. Run MkDocs strict build
uv run mkdocs build --strict

# 7. Run full pre-commit suite
uv run pre-commit run --all-files
```

---

## 5. Pull Request Guidelines

1. **Focused Changes:** Keep PRs targeted to specific features, bug fixes, or documentation refinements.
2. **Commit Messages:** Follow standard conventional commit format:
   - `feat(extraction): implement 320x180 optical flow pre-downscaling`
   - `fix(triage): evaluate caregiver red flags before sensor quality gate`
   - `docs(specs): clarify clip-to-window temporal sequence pooling`
3. **No Sensitive Data:** Never commit raw video, audio clips, facial biometric data, or personal identifying information.

---

## 6. License

By contributing to Project N, you agree that your contributions will be licensed under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
