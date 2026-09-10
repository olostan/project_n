# Project N: Architectural, Neurobiological & Systems Engineering Design Document

---

## 1. Biological and Neurological Foundations

### 1.1 Neurobiology & Communication in Non-Verbal Individuals
Autism Spectrum Disorder (ASD) encompasses heterogeneous neurodevelopmental profiles. For non-verbal children—specifically conceptualized around Nolan (Child N), a 7-year-old completely non-verbal autistic boy—expressive spoken language is absent. Childhood apraxia of speech (CAS), atypical oral-motor coordination, or motor-planning challenges frequently co-occur, dissociating cognitive capacity from phonetic articulation.

Crucially, **the absence of verbal speech does not indicate the absence of language, cognition, agency, or communicative intent**. The scientific grounding for Project N rests on several foundational principles:

- **Heterogeneous Connectivity Profiles:** Classical models proposed local hyper-connectivity coupled with long-range hypo-connectivity. Contemporary neuroimaging indicates that functional and structural connectivity alterations are highly heterogeneous; long-range underconnectivity is reasonably supported, while local microcircuit hyper-reactivity varies widely across individuals. Project N rejects one-size-fits-all neurological assumptions, adopting an individualized (N-of-1) measurement posture.
- **Sensorimotor & Cerebellar Coordination:** Group-level post-mortem and imaging studies show morphological variations in cerebellar Purkinje cells, which contribute to predictive motor sequencing and sensorimotor integration. However, attributing specific internal cerebellar mechanics to an individual child from a video clip is an unlicensed causal jump. Project N treats motor differences as observable kinematic patterns rather than direct indicators of neural circuit failure.
- **Autonomic Regulation & Objective Physiological States:** Autonomic nervous system (ANS) tone—including sympathetic arousal and vagal modulation—fluctuates in response to sensory demands. However, respiratory sinus arrhythmia (RSA) and vagal tone findings in autism are complex and non-uniform. Inferring autonomic hyper-arousal solely from surface movement or acoustic volume is circular; an objective assessment requires direct physiological sensing (electrodermal activity, heart rate variability, and accelerometry).

### 1.2 Sensory Processing Differences & Stimming Functions
Sensory processing in autistic individuals frequently diverges across auditory, proprioceptive, vestibular, and tactile domains:

1. **Proprioceptive & Vestibular Seeking:** Atypical sensory threshold gating can lead individuals to seek intense vestibular or proprioceptive input to achieve somatic equilibrium (e.g., rhythmic rocking, vertical jumping, or rapid hand/wrist stimming).
2. **Auditory Hyper-Reactivity & Gating Differences:** Thalamocortical gating differences can reduce acoustic habituation. Ambient noises (such as mechanical hums or overlapping voices) can register as acute somatic distress rather than ignorable background sound.
3. **Stimming as Active Self-Regulation & Predictability Seeking:**
   Motor stimming (e.g., 3 Hz to 6 Hz wrist rotation, finger-flicking) and tonal vocalizations are not purposeless pathology to be suppressed. Contemporary cognitive neuroscience—notably the **HIPPEA framework** (High Inflexible Precision of Prediction Errors in Autism; Van de Cruys et al., 2014)—conceptualizes stereotypic movements as strategies to generate highly predictable sensory feedback in an uncertain or overwhelming environment. Stims fulfill plural, context-dependent roles: down-regulating hyper-arousal, up-regulating under-stimulated sensory pathways, expressing joy, or communicating engagement.

### 1.3 Paralinguistic Structure of Idiosyncratic Vocalizations
In the absence of phonemic speech, communicative and affective states are conveyed through non-verbal acoustic signals:

- **Continuous Tonal Hums & Pitch Dynamics:** Sustained vocalizations contain measurable fundamental frequency ($F_0$), harmonic spacing, jitter, and shimmer. Within an individual child, shifts in pitch trajectory and vocal effort correlate with internal homeostatic state and communicative intent.
- **Harmonic Decay & Voice Quality:** Micro-pitch variations, harmonic-to-noise ratio (HNR), and spectral tilt carry affective valence. However, as established in the scientific audit (`docs/REVIEW_REFINEMENTS.md`), standard filterbanks cannot resolve fine micro-pitch without specialized acoustic tracking.

---

## 2. Epistemic Validity, the "Rosetta Stone" Fallacy & Grounding

### 2.1 The Emotion-Inference Critique & Epistemic Traps
A core risk in automated affective computing is the assumption that facial movements or vocal acoustic properties map uniformly to internal emotional states. As established by Barrett et al. (2019), emotional expressions are profoundly context-dependent; observer agreement does not establish ground truth.

Project N avoids two critical epistemic traps:

1. **The Facilitated Communication (FC) / RPM Authorship Trap:**
   Facilitated Communication, Rapid Prompting Method (RPM), and Spelling to Communicate (S2C) all failed blinded message-passing tests because the facilitator or observer unknowingly authored the message. If an AI system is trained solely on a caregiver's interpretation (`parent_tag`) and then outputs that same interpretation back to the caregiver, it creates a closed confirmation loop that manufactures false certainty. The child is excluded as an active author.
2. **The Truth Criterion (Actionable Resolution & Child Authorship):**
   Project N establishes an objective, falsifiable ground truth:

   - **Primary Truth Criterion:** The child's direct response via Augmentative and Alternative Communication (AAC) or explicit physical choices outranks all adult interpretations.
   - **Secondary Truth Criterion:** Documenting whether an offered support (e.g., offering water, deep pressure, or a quiet break) successfully resolved the observed distress episode within an observed temporal window.

### 2.2 Failure of Commercial Foundation Encoders & Unconstrained LLMs
Standard foundation models impose severe neurotypical inductive biases:

- **Phonemic Discretization (The Whisper Failure):** ASR models are trained to map acoustic energy into discrete phonemic and lexical tokens. Whisper's decoder discards non-lexical harmonic resonances, vowel hums, and pitch contours as "untranscribable noise." However, intermediate encoder representations (layers 6–12) retain rich paralinguistic and prosodic information. The failure lives in the phonemic text head and CTC loss, not necessarily in the acoustic encoder layers.
- **Spatial Pooling (The Standard ViT Failure):** Standard vision transformers pool pixels spatially across frames, obliterating 3 Hz–6 Hz hand or finger stims into generic background scenery tokens.
- **The Unaligned Prefix Fallacy (`F-01`):** Projecting continuous sensory vectors directly into a frozen LLM prefix without extensive end-to-end multimodal alignment training (which requires hundreds of thousands of paired examples) yields random vectors from the LLM's perspective. The LLM will generate fluent, confident, but **input-independent** clinical prose. Project N therefore removes the LLM from the primary inference path.

---

## 3. Sensory Extraction, Physics & Feature Determination

Project N deploys a modular, multi-pathway sensory extraction architecture combining acoustics, kinematics, and direct physiology.

```mermaid
graph TD
    subgraph Streams ["Sensory Input Streams"]
        Audio["Acoustic Stream (48 kHz WAV)<br/>Micro-pitch F0, CQT 84 bins, 128 Log-Mel"]
        Video["Kinematic Stream (30 fps 720p)<br/>75 Body/Hand Pose Landmarks, RAFT Optical Flow"]
        Physio["Physiological Stream (Wearable)<br/>EDA Conductance, HRV Vagal Tone, Accelerometry"]
    end

    subgraph FrontEnds ["Sensory Latent Projections"]
        X_a["Acoustic Latent<br/>X_a ∈ ℝ^(T_a × 512)"]
        X_k["Kinematic Latent<br/>X_k ∈ ℝ^(T_k × 512)"]
        X_p["Physiological Latent<br/>X_p ∈ ℝ^(T_p × 128)<br/>(Masked via e_∅ if unmonitored)"]
    end

    subgraph Fusion ["Cross-Modal Binding (Apple Silicon MLX)"]
        Resampler["Multimodal Perceiver Resampler<br/>Audio-Visual Correspondence (CAV-MAE)"]
        Pool["Attention Pooling & L2 Normalization"]
        Metric["128-dim Normalized Metric Vector<br/>z_metric ∈ ℝ^128 (||z||₂ = 1)"]
    end

    Audio --> X_a
    Video --> X_k
    Physio --> X_p

    X_a & X_k & X_p --> Resampler
    Resampler --> Pool
    Pool --> Metric

    style Streams fill:#f8fafc,stroke:#94a3b8,stroke-width:1px
    style Audio fill:#e6f7ff,stroke:#1890ff,stroke-width:2px
    style Video fill:#f6ffed,stroke:#52c41a,stroke-width:2px
    style Physio fill:#fffbe6,stroke:#faad14,stroke-width:2px
    style Fusion fill:#f9f0ff,stroke:#722ed1,stroke-width:2px
    style Metric fill:#fff0f6,stroke:#eb2f96,stroke-width:3px
```

### 3.1 Acoustic Front End (Resolving the Micro-Pitch Limit)
In a 7-year-old child, fundamental phonation frequencies range from $150\text{ Hz}$ to $400\text{ Hz}$. At a $48\text{ kHz}$ sampling rate, a 128-band log-mel filterbank produces bands of approximately $27\text{ Hz}$ width near $300\text{ Hz}$, rendering $\pm 15\text{ Hz}$ micro-pitch shifts sub-bin and unresolvable.

Project N resolves this with a dedicated tripartite acoustic engine:

```mermaid
graph TD
    RawAudio["Raw Audio Input<br/>48 kHz PCM (5.0s window = 240,000 samples)"] --> Branch1["1. Dedicated Voice Quality & Pitch Track<br/>pYIN / Autocorrelation (~10ms hop)<br/>Outputs: F0 (~1 Hz res), Jitter, Shimmer, HNR, CPP"]
    RawAudio --> Branch2["2. Constant-Q Transform (CQT) Filterbank<br/>84 geometrically spaced bins (7 octaves)<br/>Logarithmic resolution across human vocal range"]
    RawAudio --> Branch3["3. Broadband Log-Mel Filterbank<br/>128 bands across 20 Hz – 24,000 Hz<br/>STFT N=2048, hop H=160, periodic Hann window"]

    Branch1 & Branch2 & Branch3 --> Align["Linear Alignment & Temporal Concatenation"]
    Align --> LatentAudio["Acoustic Latent Representation<br/>X_audio ∈ ℝ^(B × 500 × 768)"]

    style RawAudio fill:#f0f5ff,stroke:#2f54eb,stroke-width:2px
    style Branch1 fill:#e6f7ff,stroke:#1890ff,stroke-width:2px
    style Branch2 fill:#f6ffed,stroke:#52c41a,stroke-width:2px
    style Branch3 fill:#fffbe6,stroke:#faad14,stroke-width:2px
    style Align fill:#f9f0ff,stroke:#722ed1,stroke-width:2px
    style LatentAudio fill:#fff0f6,stroke:#eb2f96,stroke-width:3px
```

### 3.2 Kinematic Front End (Pose Over Raw Differencing)
Raw-pixel temporal differencing ($\mathbf{\Delta}_t = \|\mathbf{Z}_t - \mathbf{Z}_{t-1}\|_1$) is highly sensitive to handheld camera shake, auto-exposure adjustments, mains flicker, and room shadows.

Project N establishes a robust kinematic hierarchy:

```mermaid
graph TD
    RawVideo["Raw Video Input<br/>30 fps @ 720p (5.0s window = 150 frames)"] --> PoseStream["1. Body-Relative Pose (MediaPipe Holistic)<br/>• 33 Body landmarks (torso, head, limbs)<br/>• 42 Hand keypoints (21 per hand)<br/>Total: 75 keypoints (x, y, visibility)"]
    RawVideo --> FlowStream["2. Dense Optical Flow (RAFT)<br/>Motion displacement field (u, v)<br/>Spatially pooled to 8×8 grid (128-dim)"]

    PoseStream --> TorsoNorm["Torso-Relative Normalization<br/>Scaled by inter-shoulder distance:<br/>p̃ = (p - p_midhip) / ||p_lshoulder - p_rshoulder||₂<br/>(Invariant to camera shake, zoom, and distance)"]

    TorsoNorm & FlowStream --> TempTrans["Temporal Transformer Encoder"]
    TempTrans --> LatentKinematic["Kinematic Latent Representation<br/>X_kinematic ∈ ℝ^(B × 150 × 512)"]

    style RawVideo fill:#f0f5ff,stroke:#2f54eb,stroke-width:2px
    style PoseStream fill:#e6f7ff,stroke:#1890ff,stroke-width:2px
    style FlowStream fill:#fffbe6,stroke:#faad14,stroke-width:2px
    style TorsoNorm fill:#f6ffed,stroke:#52c41a,stroke-width:2px
    style TempTrans fill:#f9f0ff,stroke:#722ed1,stroke-width:2px
    style LatentKinematic fill:#fff0f6,stroke:#eb2f96,stroke-width:3px
```

### 3.3 Physiological Front End & Graceful Degradation
To ground internal arousal without circular inference from video, Project N integrates wearable telemetry (e.g., Empatica EmbracePlus, or Apple Watch sensor streaming):

```mermaid
graph TD
    WearableInput["Wearable Telemetry Stream<br/>(Optional BLE / Apple Watch)"] --> SensorGate{"Sensors Available?"}

    SensorGate -- "Yes" --> EDA["Electrodermal Activity (EDA @ 4 Hz)<br/>cvxEDA decomposition into tonic SCL and phasic SCR"]
    SensorGate -- "Yes" --> HRV["Cardiorespiratory (PPG @ 100 Hz)<br/>Inter-Beat Intervals (IBI) & HRV RMSSD"]
    SensorGate -- "Yes" --> Acc["3-Axis Accelerometry (@ 50 Hz)<br/>Somatic tremor energy & gross movement"]

    EDA & HRV & Acc --> PhysioLatent["Physiological Latent<br/>X_physio ∈ ℝ^(B × 50 × 64)"]

    SensorGate -- "No" --> NullModality["Missing Modality Gating<br/>Substitutes learned null embedding: e_∅^physio<br/>(Zero performance drop on dual-modal operation)"]

    style WearableInput fill:#f0f5ff,stroke:#2f54eb,stroke-width:2px
    style SensorGate fill:#f9f0ff,stroke:#722ed1,stroke-width:2px
    style EDA fill:#e6f7ff,stroke:#1890ff,stroke-width:2px
    style HRV fill:#f6ffed,stroke:#52c41a,stroke-width:2px
    style Acc fill:#fffbe6,stroke:#faad14,stroke-width:2px
    style PhysioLatent fill:#f6ffed,stroke:#52c41a,stroke-width:2px
    style NullModality fill:#fff1f0,stroke:#f5222d,stroke-width:2px
```

---

## 4. Target Architecture: Retrieval-First, Rendering-Only

Project N inverts traditional multimodal generation. The Large Language Model is removed from the primary classification loop and restricted to schema-constrained rendering.

### 4.1 Metric Space & Prototypical Learning

```mermaid
graph TD
    L1["Layer 1: Measured Observation<br/>• Acoustic latent X_a (F0, CQT, Log-Mel)<br/>• Kinematic latent X_k (Normalized Pose, RAFT Flow)<br/>• Physiological latent X_p (EDA, HRV, Accel)"] --> Resampler["Multimodal Perceiver Resampler (Apple MLX)<br/>Cross-attention audio-visual temporal correspondence"]

    Resampler --> Pooling["Attention Pooling & L2 Normalization"]
    Pooling --> MetricVector["128-dimensional Normalized Metric Vector<br/>z_metric ∈ ℝ^128 (||z||₂ = 1)"]

    MetricVector --> MemoryStore["Episodic Memory Retrieval (ChromaDB / SQLite)<br/>Cosine similarity & Euclidean distance to prototypes c_k"]
    MemoryStore --> MatchedCandidates["Ranked Historical Verified Episodes of Child N<br/>Prior physical resolutions, latency, and context"]

    style L1 fill:#e6f7ff,stroke:#1890ff,stroke-width:2px
    style Resampler fill:#f9f0ff,stroke:#722ed1,stroke-width:2px
    style Pooling fill:#f9f0ff,stroke:#722ed1,stroke-width:2px
    style MetricVector fill:#fff0f6,stroke:#eb2f96,stroke-width:3px
    style MemoryStore fill:#f6ffed,stroke:#52c41a,stroke-width:2px
    style MatchedCandidates fill:#fffbe6,stroke:#faad14,stroke-width:2px
```

The high-dimensional sensory representations are mapped into a compact, 128-dimensional metric space:
$$\mathbf{z} = \text{L2\_Normalize}\left( \text{AttentionPool}(\mathbf{X}_{sensory}) \mathbf{W}_{proj} \right) \in \mathbb{R}^{128}$$

Using a 128-dimensional metric space (rather than 4096 dimensions) prevents geometric collapse when operating with hundreds of labeled historical examples rather than hundreds of thousands. Matching is performed using cosine similarity or Euclidean distance over class prototype centers $\mathbf{c}_k$:
$$\mathbf{c}_k = \frac{1}{|S_k|} \sum_{i \in S_k} \mathbf{z}_i$$

### 4.2 Calibrated Abstention & Epistemic Decision Gating

```mermaid
graph TD
    Query["Query Vector z & Nearest Prototype c_nearest"] --> Gate1{"Calibrated Abstention Gate<br/>d(z, c_nearest) > τ_abstain ?"}

    Gate1 -- "Yes (Novel / Unseen)" --> AbstainCard["Abstention Card<br/>'Unrecognized Pattern. Insufficient historical similarity.'<br/>Recommended Action: Present open AAC board or check environment."]

    Gate1 -- "No (Familiar Episode)" --> Gate2{"Validated Medical Gate<br/>NCCPC-R Distress Score ≥ 6 ?"}

    Gate2 -- "Yes (Acute Distress)" --> MedicalCard["Medical Escalation Card<br/>Warrants review for physical pain (ear, dental, GI reflux).<br/>Behavioral/sensory interpretations suppressed."]

    Gate2 -- "No (Regulated / Stimming)" --> DualExec["Dual Execution Pipeline"]

    subgraph AACPathway ["Child Authorship via AAC Bridge"]
        AAC["AAC Candidate Tile Dispatch<br/>Dispatches options ([Water], [Sensory Break], [Deep Pressure])<br/>directly to Nolan's speech device.<br/><b>Child direct selection is authoritative ground truth.</b>"]
    end

    subgraph CaregiverPathway ["Caregiver Decision Support"]
        LLM["Schema-Constrained LLM (Qwen2.5-14B)<br/>• 100% Frozen Base Weights (W₀)<br/>• Formats L1-L4 into distinct visual cards<br/>• Zero ungrounded generative narratives"]
    end

    DualExec --> AAC
    DualExec --> LLM

    style Query fill:#f0f5ff,stroke:#2f54eb,stroke-width:2px
    style Gate1 fill:#f9f0ff,stroke:#722ed1,stroke-width:2px
    style Gate2 fill:#f9f0ff,stroke:#722ed1,stroke-width:2px
    style AbstainCard fill:#fff7e6,stroke:#fa8c16,stroke-width:2px
    style MedicalCard fill:#fff1f0,stroke:#f5222d,stroke-width:2px
    style DualExec fill:#f6ffed,stroke:#52c41a,stroke-width:2px
    style AACPathway fill:#fffbe6,stroke:#faad14,stroke-width:2px
    style CaregiverPathway fill:#e6f7ff,stroke:#1890ff,stroke-width:2px
```

If the distance between the query vector $\mathbf{z}$ and the nearest historical prototype exceeds a calibrated threshold $\tau_{abstain}$:
$$d(\mathbf{z}, \mathbf{c}_{nearest}) > \tau_{abstain}$$
the system **abstains from classification**. It outputs:
```text
Unrecognized Pattern. Insufficient historical similarity to classify.
Recommended Action: Observe environmental context or present AAC open choice board.
```

---

## 5. Clinical Evidence Ingestion & Grounding Pipeline

Rather than bulk-fine-tuning LoRA on academic text (which causes hallucination and fact drift), Project N grounds its clinical knowledge via an inspectable, versioned Retrieval-Augmented Generation (RAG) store.

```mermaid
graph TD
    Lit["Curated Academic Literature<br/>(FBA, HIPPEA, Interoception, AAC, NCCPC-R, Ayres Sensory Integration)"] --> PDF["PDF Extractor (PyMuPDF)<br/>Preserves section hierarchy, tables & metadata"]
    PDF --> Chunk["Semantic Chunker<br/>~500 tokens / chunk with 50-token sliding overlap"]
    Chunk --> Meta["Metadata Enricher<br/>Extracts author, year, population, study design & evidence grade"]
    Meta --> Embed["Local MLX Embedding Model<br/>nomic-embed-text-v1.5 (768-dim metric space)"]
    Embed --> Chroma["ChromaDB: clinical_evidence Collection<br/>Local HNSW cosine index (100% offline)"]
    Chroma -.-> Retrieval["Runtime Layer 4 (L4) Citation Engine<br/>Matches verified episode outcome to peer-reviewed literature"]

    style Lit fill:#f0f5ff,stroke:#2f54eb,stroke-width:2px
    style PDF fill:#f6ffed,stroke:#52c41a,stroke-width:2px
    style Chunk fill:#f6ffed,stroke:#52c41a,stroke-width:2px
    style Meta fill:#fffbe6,stroke:#faad14,stroke-width:2px
    style Embed fill:#f9f0ff,stroke:#722ed1,stroke-width:2px
    style Chroma fill:#e6f7ff,stroke:#1890ff,stroke-width:3px
    style Retrieval fill:#fff0f6,stroke:#eb2f96,stroke-width:2px
```

### 5.1 Evidence Synthesis Without Hallucination
When an episode matches historical precedents in Layer 2 (e.g., deep proprioceptive pressure resolution), the system queries the `clinical_evidence` collection using the structured outcome. The retrieved excerpt is cited directly in Layer 4 (L4) with its author, publication year, and evidence level. The LLM is strictly forbidden from adding clinical claims beyond what is cited in L4.

---

## 6. Full System Architecture: Server, Storage & Client UI

### 6.1 Server Architecture: Python & FastAPI
The server executes as an asynchronous daemon under Python 3.11+ using FastAPI and Uvicorn:

- **Metal Memory Residency:** MLX tensor arrays reside directly in Apple Silicon Unified RAM. FastAPI endpoints execute inference passes via Python C++ bindings with zero IPC overhead.
- **Server-Sent Events (SSE) Bus:** Real-time event streaming (`/api/v1/events/stream`) pushes live progress to the web dashboard (demuxing $\to$ acoustic $\to$ kinematic $\to$ metric $\to$ retrieval $\to$ AAC routing $\to$ LLM streaming).
- **Two-Key macOS Vault:** Background video ingestion while the Mac screen is locked uses a signed LaunchAgent helper with Apple's Data Protection Keychain (`SecItem` with `kSecAttrAccessibleAfterFirstUnlock`). Touch ID authentication is required to decrypt and view raw video files.

### 6.2 Mobile Companion Architecture (Flutter Android & iOS)
The mobile companion app runs on Flutter, supporting both Android and iOS:

- **Playground & Clinic Offline Mode:** When Child N is away from home Wi-Fi (e.g., at an OT therapy session or outdoor playground), the app records 30–120s clips, prompts for quick optional caregiver notes, and stores them in a local AES-encrypted SQLite queue (`offline_clips_outbox`).
- **Store-and-Forward Background Flushing:** When the phone returns home and detects the Mac helper over local Wi-Fi, the background sync manager flushes queued clips over mutual TLS (mTLS) with SHA-256 chunk verification.

### 6.3 Local Caregiver Dashboard (React + Tailwind SPA)
The local Mac interface is a modern React SPA served directly by the FastAPI backend at `http://127.0.0.1:8080`:

- **Live System Telemetry:** Real-time gauge of active vs. peak Metal unified memory, thermal state, and MLX engine status.
- **Timeline & Episode Browser:** Filterable historical diary of verified episodes.
- **Multimodal Video Inspector:** Synchronized video player with `<canvas>` skeletal overlay, interactive audio pitch ($F_0$) waveform, and the four-layer output card.
- **2D UMAP Lexicon Visualizer:** Interactive WebGL cluster map showing the geometry of Child N's behavioral repertoire.
- **Candidate Model Promotion Gate:** Visual holdout calibration curves and one-click model promotion/rollback.

---

## 7. Continuous Adaptation & Periodic Re-fit Protocol

### 7.1 Periodic Full Re-fit Over Nightly SGD
Rather than running unstable nightly SGD on single batches, Project N executes a periodic **full re-fit**:

- Upon accumulation of $K \ge 10$ new verified episodes, the 128-dimensional metric projection head and prototype cluster centers are re-fit over the entire verified historical dataset.
- On Apple Silicon Metal shaders, re-fitting a 128-dimensional metric space over hundreds of episodes completes in seconds, making catastrophic forgetting structurally impossible.

### 7.2 Gated Model Promotion Pipeline
Before any candidate model is deployed to caregiver-facing inference, it must pass the preregistered evaluation protocol in [`docs/evaluation_protocol.md`](file:///Users/olostan/code/project_n/docs/evaluation_protocol.md):

1. **Holdout Evaluation:** Evaluated on leave-one-day-out (LODO) splits and the locked 50-episode safety holdout set.
2. **Safety Regression Check:** Zero tolerance for missed NCCPC-R distress events.
3. **Caregiver Sign-off:** The caregiver inspects validation metrics on the dashboard and explicitly confirms promotion.
