# Project N: Multimodal Communication Support & Behavioral Insight Pipeline

[![Platform: Apple Silicon](https://img.shields.io/badge/Platform-Apple%20Silicon%20(M5%20Pro)-black?style=flat-square&logo=apple)](https://developer.apple.com/metal/)
[![Engine: Apple MLX](https://img.shields.io/badge/Engine-Apple%20MLX%200.22+-blue?style=flat-square)](https://github.com/ml-explore/mlx)
[![Memory: 48GB Unified RAM](https://img.shields.io/badge/Memory-48GB%20Unified%20RAM%20(36GB%20Cap)-green?style=flat-square)]()
[![Privacy: 100% Offline Two-Key Vault](https://img.shields.io/badge/Privacy-100%25%20Offline%20Two--Key%20Vault-red?style=flat-square)]()
[![Clients: Flutter (Android/iOS) + React Dashboard](https://img.shields.io/badge/Clients-Flutter%20%2B%20React%20SPA-blueviolet?style=flat-square)]()
[![Docs: Live Research Site](https://img.shields.io/badge/Docs-olostan.github.io%2Fproject__n-blue?style=flat-square)](https://olostan.github.io/project_n/)

> [!IMPORTANT]
> 🌐 **Live Documentation & Research Website:**
> **[https://olostan.github.io/project_n/](https://olostan.github.io/project_n/)**
> *Explore interactive system diagrams, zoomable architectures, clinical evaluation protocols, and technical specifications.*

---

## A Note from the Project Initiator

> *"I am [Valentyn Shybanov](https://olostan.me/), a software engineer, systems architect, and the father of Nolan, a 7-year-old completely non-verbal boy with Level 3 autism. Every single day of my life is shaped by the profound love, challenge, and heartbreak of trying to understand my son. Nolan is non-verbal. Completely.*
>
> *My son cannot use spoken words, but he is never silent. He communicates continuously: through subtle pitch inflections in his throat, micro-tremors in his hands, bodily orientations, and rhythms of movement. Traditional foundation models and commercial cloud AI discard these signals as 'meaningless background noise.' But to me, as his father, that 'noise' is his entire voice.*
>
> *I started Project N not as a commercial product, not to make money, and not to promote a startup. I started it out of a father’s deep desire to understand his child. I want to dedicate my twenty-plus years of engineering experience, systems knowledge, and machine learning skills to build a free, open-source, local-first tool that can help parents like me and the dedicated therapists who support our children.*
>
> *If you are a speech-language pathologist, an occupational therapist, an autism researcher, or an engineer who cares about non-verbal communication: I invite you with an open heart to review this work, critique it, and help us make it better."*
>
> — **Valentyn Shybanov** (Father & Project Initiator · [olostan.me](https://olostan.me/) · [GitHub](https://github.com/olostan))

---

## What is Project N?

**Project N is a 100% offline, privacy-preserving behavioral insight and communication-support assistant** engineered specifically for completely non-verbal neurodivergent children and the parents and therapists who care for them.

Instead of trying to force non-verbal vocalizations into synthetic English speech or making ungrounded claims about what a child is "thinking", Project N:
1. **Analyzes raw bioacoustic physics:** Tracks vocal pitch ($F_0$), harmonic overtones (CQT), vocal strain, and body kinematics (pose frequency, optical flow) without phonemic collapse.
2. **Models the dyadic interaction loop:** Grounded in the **SCERTS framework** and transactional communication models, observing how adult scaffolding and child responses interact.
3. **Provides dual-perspective insight:** Offers **Parent View** (gentle exploratory possibilities to investigate and low-risk co-regulatory ideas grounded in past successes) and **Therapist View** (dense bioacoustic and kinematic telemetry for clinical sessions).
4. **Prioritizes medical safety:** Runs automated acoustic/motion anomaly screening to prompt pediatrician-approved comfort checks before any behavioral hypotheses are considered.
5. **Guarantees total privacy:** Runs 100% locally on Apple Silicon (Apple MLX) with zero cloud network telemetry and two-key encrypted storage.

---

## Documentation Directory (`docs/`)

All scientific whitepapers, clinical protocols, architectural designs, and engineering specifications reside in the **[`docs/`](docs/)** directory and are served on the [live documentation website](https://olostan.github.io/project_n/):

| Document | Focus & Target Audience |
| :--- | :--- |
| [📄 **Scientific Whitepaper**](docs/WHITE_PAPER.md) | Clinical foundations, dyadic transactional model, SCERTS alignment, and bioacoustics for SLPs, OTs, and autism researchers. |
| [📋 **Evaluation Protocol**](docs/evaluation_protocol.md) | Prospective single-participant longitudinal evaluation design, forward-chaining splits, and model promotion criteria. |
| [🧠 **Theoretical Architecture & Design**](docs/DESIGN.md) | Detailed neurobiology, acoustic physics, pose kinematics, literature grounding, and continuous adaptation. |
| [📐 **Technical Specifications**](docs/SPECS.md) | Mathematical definitions, tensor shapes, REST endpoints, SSE event streams, and executable MLX implementations. |
| [🛡️ **System & Safety Invariants**](INVARIANTS.md) | Non-negotiable repository guardrails: 100% offline boundary, zero cloud SDKs, frozen base LLM, and encrypted storage. |
| [🔍 **Multi-Reviewer Scientific Audit**](docs/REVIEW_REFINEMENTS.md) | Comprehensive finding-by-finding peer review and corrective action matrix. |
| [🤖 **Autonomous Agent Directives**](AGENTS.md) | Engineering standards, Apple MLX memory management conventions, and documentation synchronization rules. |

---

## Getting Started & Local Development

### Prerequisites

- **Hardware:** Apple Silicon Mac (M-series with 32 GB+ Unified Memory; 48 GB recommended).
- **Operating System:** macOS 14.0 (Sonoma) or newer.
- **Python Environment:** Python 3.11+ with the [`uv`](https://github.com/astral-sh/uv) package manager.

### 1. Clone the Repository

```bash
git clone https://github.com/olostan/project_n.git
cd project_n
```

### 2. Browse Documentation Locally

To browse the complete documentation site with interactive diagrams, full-screen zoom, and search:

```bash
uv run --with mkdocs --with mkdocs-material mkdocs serve
```
Then open [http://localhost:8000](http://localhost:8000) in your web browser.

### 3. Verify Citations & Documentation Integrity

To run the automated Crossref and DataCite metadata verification suite across all academic citations:

```bash
uv run python tests/verify_citations.py
```

To test building the documentation in strict mode (ensuring zero broken links or markdown warnings):

```bash
uv run mkdocs build --strict
```

### 4. Linters, Type Checking & Pre-Commit Quality Gates

Project N enforces strict quality and safety gates on every commit (`pre-commit`) and push (`pre-push`):
- **Code & Style:** `ruff` (linter + formatter) with strict rule sets.
- **Static Types:** `mypy` in full `--strict` mode.
- **Safety Invariant 10:** Zero external cloud AI SDKs (`openai`, `anthropic`, `vertexai`, `google.generativeai`) and zero PyTorch imports in runtime modules.
- **Docs & Links:** Relative markdown link verification (`tests/check_markdown_links.py`) and `mkdocs build --strict`.
- **Academic Citations:** Crossref and DataCite DOI verification prior to pushing.

To set up the development environment and install the git hooks:

```bash
# Install dependencies using uv
uv sync --all-extras

# Install git pre-commit and pre-push hooks
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

To execute all quality checks manually across all files:

```bash
uv run pre-commit run --all-files
```

---

## Repository Structure

```
project_n/
├── README.md                      # Project portal & onboarding (this file)
├── INVARIANTS.md                  # Non-negotiable safety guardrails (offline, privacy, frozen LLM)
├── AGENTS.md                      # AI coding agent operating directives & MLX standards
├── pyproject.toml                 # Project metadata, dependencies, ruff & strict mypy config
├── .pre-commit-config.yaml        # Git pre-commit & pre-push quality gates
├── mkdocs.yml                     # Documentation site configuration
├── docs/                          # The single source of truth for all documentation
│   ├── index.md                   # Live documentation website homepage
│   ├── WHITE_PAPER.md             # Scientific whitepaper
│   ├── evaluation_protocol.md     # Single-participant longitudinal evaluation protocol
│   ├── DESIGN.md                  # Theoretical design & architecture
│   ├── SPECS.md                   # Technical specifications & executable MLX pipelines
│   ├── INVARIANTS.md              # Documentation mirror of root invariants
│   ├── REVIEW_REFINEMENTS.md      # Multi-reviewer scientific audit matrix
│   ├── report-source.md           # Preserved original audit critique
│   ├── stylesheets/               # Theme styling
│   └── javascripts/               # Theme scripts & MathJax
├── tests/
│   ├── check_markdown_links.py    # Automated relative markdown link validator
│   └── verify_citations.py        # Automated Crossref/DataCite DOI verifier
├── app/                           # Cross-platform Flutter mobile client (offline outbox)
├── extraction/                    # Bioacoustic (F0, CQT) & kinematic feature extraction
├── models/                        # Native Apple MLX neural encoders & metric projection
├── rag/                           # ChromaDB episodic memory & clinical knowledge base
├── server/                        # Local FastAPI backend daemon (SSE event bus)
└── training/                      # MLX training scripts & forward-chaining splits
```

---

## Implementation Roadmap

Project N is currently in **Phase 1 (Multimodal Feature Extraction & Ingestion Foundations)**:
- [x] Scientific rationale, clinical dyadic foundations, and whitepaper published.
- [x] Architecture, system invariants, and technical specifications formalized.
- [x] Prespecified evaluation protocol and safety promotion gates established.
- [ ] Phase 1: Local Mac backend daemon (FastAPI), encrypted vault, and acoustic feature pipeline.
- [ ] Phase 2: Metric projection head, episodic prototype retrieval, and ChromaDB integration.
- [ ] Phase 3: Flutter companion app with offline outbox and pairing.
- [ ] Phase 4: Local React dashboard and clinic-to-home knowledge transfer.
- [ ] Phase 5: Controlled clinical evaluation and prospective single-participant longitudinal benchmark.

---

## Safety & Operating Invariants

Any engineer or autonomous AI agent modifying this codebase is bound by the rules in [`INVARIANTS.md`](INVARIANTS.md) and [`AGENTS.md`](AGENTS.md):
- **100% Local Execution:** Zero external cloud AI SDKs (`openai`, `anthropic`, `vertexai`, `google.generativeai`).
- **Framework Purity:** 100% native Apple MLX (`mlx.core`, `mlx.nn`). Zero PyTorch imports in runtime modules.
- **Strict VRAM Budget:** Peak memory $\le 28.0\text{ GB}$ (operational peak), hard ceiling $\le 36.0\text{ GB}$.
- **Frozen Base LLM:** Base language model is 100% frozen with `model.freeze()` and kept out of primary classification paths.
- **Child Agency:** Independent child communication is treated strictly as an observed outcome, never auto-generated machine speech.
