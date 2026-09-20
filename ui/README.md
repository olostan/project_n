# Project N: Local Caregiver Web UI Dashboard (`ui/`)

The `ui/` package is a high-performance, single-page React 18 + Tailwind CSS desktop dashboard designed for local execution on Apple Silicon hardware. It provides caregivers and clinical therapists with real-time system observability, multimodal synchronized inspection, validated pain triage, clinical knowledge grounding, and model governance.

---

## 1. Architectural Principles & Invariants

- **Invariant 1 (100% Offline & Zero-Cloud):** Absolutely no remote API calls, cloud telemetry, analytics trackers, or external CDN requests. All typography (`@fontsource/inter`, `@fontsource/jetbrains-mono`) and assets are bundled locally into the output directory.
- **Invariant 6 (Four-Layer Explanatory Separation):** Explicitly partitions all observational outputs into:
  - **L1:** Measured acoustic and kinematic features.
  - **L2:** Historical verified analogues and cluster centroids.
  - **L3:** Contextual antecedents (routines, hunger, noise, transitions).
  - **L4:** Calibrated hypotheses with explicit confidence intervals and non-diagnostic notices.
- **Invariant 7 (Medical Safety Precedence):** When validated pain cut-offs (NCCPC-PV $\ge 11$, NCCPC-R $\ge 7$) are met, all behavioral interpretations are immediately locked out in favor of pediatric physical comfort protocols.
- **Invariant 8 (Dual Perspective & Child Agency):**
  - **Parent View (Default):** Warm, plain-English translation with concrete co-regulatory calming suggestions.
  - **Therapist View:** Technical bioacoustic figures ($F_0$, CPP), kinematic flow metrics, SCERTS/Ayres SI domain mapping, and linked peer-reviewed literature citations.
  - **Child Agency:** Explicit capture of Child N's intentional communicative acts (reaching, pointing, vocalizing, independent AAC selection).
- **Engineering Modularity:** Adheres strictly to the $\le 400$-line ceiling (with presentation components $\le 250$ lines) and unidirectional dependency flow.

---

## 2. Directory Structure

```
ui/
├── index.html                   # Zero-CDN HTML root with local font bundling
├── package.json                 # React 18, Vite 5, Tailwind CSS, Lucide icons
├── vite.config.ts               # Local dev proxy, asset hashing, and build output
├── tailwind.config.js           # Palette tokens, typography, and dark-mode config
├── src/
│   ├── api/                     # Type-safe API clients (Fetch wrapper)
│   │   ├── client.ts            # Base HTTP client with error typing
│   │   ├── clipsApi.ts          # Media retrieval & video stream URLs
│   │   ├── episodesApi.ts       # Episode querying and outcome logging
│   │   ├── factsApi.ts          # Child facts & clinician note confirmation
│   │   ├── modelsApi.ts         # Model checkpoint inspection & promotion
│   │   ├── nccpcApi.ts          # NCCPC evaluation submissions
│   │   └── telemetryApi.ts      # Hardware telemetry and active status
│   ├── types/                   # TypeScript interfaces mirroring backend contracts
│   │   ├── episodes.ts          # Episode, L1-L4 representations, Outcomes
│   │   ├── events.ts            # SSE ingestion and task tracking payloads
│   │   ├── facts.ts             # Child profile facts and confirmation state
│   │   ├── models.ts            # Checkpoint metadata, ECE bins, promotion
│   │   ├── nccpc.ts             # NCCPC-PV/R instruments, subscales, items
│   │   └── telemetry.ts         # Metal unified VRAM, thermal, pipeline state
│   ├── hooks/                   # Custom React lifecycle & streaming hooks
│   │   ├── useSSEStream.ts      # Resilient Server-Sent Events subscription
│   │   ├── useSystemTelemetry.ts# Hardware metrics polling with fallback
│   │   ├── useEpisodes.ts       # Filterable episode list with caching
│   │   ├── useSynchronizedPlayback.ts # 30 fps video/audio/canvas scrubbing
│   │   └── useNCCPCForm.ts      # Real-time NCCPC scoring and threshold gating
│   ├── components/              # Modular presentation components
│   │   ├── common/              # Buttons, Badges, Modals, NonDiagnosticBanner
│   │   ├── layout/              # TopHeaderBar, SidebarNav
│   │   ├── telemetry/           # VRAMGauge, ThermalIndicator, PipelineTaskTracker
│   │   ├── diary/               # EpisodeCard, EpisodeFilters, OutcomeBadge, UploadClipModal
│   │   ├── inspector/           # VideoPlayer, SkeletalCanvasOverlay, AudioPitchTrack,
│   │   │                        # FourLayerCard, ParentViewContent, TherapistViewContent,
│   │   │                        # OutcomeLoggerModal
│   │   ├── triage/              # MedicalEscalationBanner, ScoreIndicator, NCCPCSubscaleSection
│   │   ├── lexicon/             # UMAPCanvas, ClusterTooltip, LexiconLegend
│   │   ├── facts/               # FactItemCard, FactFilters, CreateFactModal
│   │   └── promotion/           # CheckpointRow, CalibrationCurve, PromotionConfirmModal
│   ├── pages/                   # Top-level screen assemblies
│   │   ├── LiveTelemetryPage.tsx# Hardware memory & streaming ingestion tracker
│   │   ├── DiaryPage.tsx        # Historical episode browser & search
│   │   ├── InspectorPage.tsx    # Multimodal video, canvas, audio & L1-L4 card
│   │   ├── PainTriagePage.tsx   # Digital NCCPC-PV/R checklist & escalation
│   │   ├── LexiconPage.tsx      # 2D UMAP cluster visualization
│   │   ├── FactsLibraryPage.tsx # Clinical RAG & caregiver confirmation gate
│   │   └── PromotionGatePage.tsx# Model calibration audit & promotion sign-off
│   ├── App.tsx                  # Tab navigation & top-level layout coordinator
│   ├── main.tsx                 # React DOM mount point
│   └── index.css                # Tailwind directives & local font bindings
└── dist/                        # Compiled production bundle served by FastAPI
```

---

## 3. Screen Topology & Functional Overview

| Screen | Primary Route | Key Features |
|---|---|---|
| **Live Telemetry** | `telemetry` | Real-time Metal VRAM meter ($\le 36.0\text{ GB}$), thermal state monitor, live pipeline SSE tracker. |
| **Episode Diary** | `diary` | Timeline list, antecedent filtering, duration and outcome badges, direct inspector launch. |
| **Video Inspector** | `inspector` | Frame-by-frame 30 fps playback, 33 MediaPipe pose landmarks, $F_0$ pitch curve, dual Parent/Therapist view, outcome logger. |
| **Pain Triage** | `triage` | Interactive NCCPC-PV / NCCPC-R assessment, live score computation, automatic medical lockouts. |
| **Behavioral Lexicon**| `lexicon` | 2D interactive projection of 128-d metric embeddings, action resolution color legend, cluster tooltips. |
| **Facts Library** | `facts` | Child facts management, sensory trigger tagging, fail-closed caregiver confirmation gate. |
| **Promotion Gate** | `promotion` | Validation MRR, ECE reliability diagram across 5 bins, safety regression audit, one-click rollback. |

---

## 4. Development & Build Commands

### Prerequisites
- Node.js 18+ and npm 9+

### Local Development Server
To launch the Vite development server with API proxying to the local FastAPI daemon (`http://127.0.0.1:8080`):
```bash
cd ui
npm install
npm run dev
```

### Production Build & Bundling
To build the production-ready, self-contained offline bundle:
```bash
cd ui
npm run build
```
This writes optimized HTML, CSS, JavaScript, and font assets to `ui/dist/`.

---

## 5. Backend Daemon Integration

The compiled bundle in `ui/dist/` is served directly by the Python FastAPI daemon (`server/main.py`):
- Static build assets are mounted at `/assets` -> `ui/dist/assets`.
- Root and client routes (`/`, `/telemetry`, `/diary`, `/inspector`, etc.) are resolved via the SPA index fallback handler (`/{full_path:path}`).
- All API routes under `/api/*` take strict precedence and are never intercepted by the static fallback handler.
