# Project N: Architectural, Neurobiological & Epistemic Design Document
## Multimodal Communication Support, Epistemic Validity, and Retrieval-First Architecture

---

## 1. Biological and Neurological Foundations

### 1.1 Neurobiology & Communication in Minimally Speaking Individuals
Autism Spectrum Disorder (ASD) represents a heterogeneous spectrum of neurodevelopmental profiles. For non-verbal or minimally speaking children—specifically conceptualized around Child N, a 7-year-old minimally speaking child—expressive spoken language is limited. Childhood apraxia of speech (CAS), atypical oral-motor coordination, or motor-planning challenges frequently co-occur, dissociating cognitive capacity from phonetic articulation.

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

## 2. The "Rosetta Stone" Fallacy, Epistemics & Grounding

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

## 3. Latent Space Geometry & Sensory Extraction Architecture

Project N deploys a modular, multi-pathway sensory extraction architecture combining acoustics, kinematics, and direct physiology.

```text
RAW ACOUSTIC STREAM (48 kHz)
├─ Dedicated Pitch Track ────► F0, Jitter, Shimmer, HNR (~10ms hop) ──────┐
├─ Harmonic Filterbank   ────► Constant-Q Transform (CQT) / ERB Filters   ├─► Acoustic Latent (X_a)
└─ Broadband Texture     ────► 128 Log-Mel Spectrogram Bands              │
                                                                           │
KINEMATIC STREAM (30 fps)                                                  │
├─ Body-Relative Pose   ────► MediaPipe Holistic / BlazePose Landmarks   ├─► Kinematic Latent (X_k)
├─ Dense Optical Flow   ────► RAFT Motion Vector Field                     │
└─ Visual Context       ────► Low-Rate Context / TD-ViT (Ablation Arm)    │
                                                                           │
PHYSIOLOGICAL STREAM (Wearable)                                            │
├─ Electrodermal (EDA)  ────► Tonic SCL & Phasic SCR Conductance           ├─► Physiological Latent (X_p)
├─ Cardiorespiratory    ────► Inter-Beat Intervals & Heart Rate Var (HRV)  │
└─ Accelerometry        ────► 3-Axis Somatic Motion & Tremor Energy        │
                                                                           ▼
                                                Multimodal Temporal Binding (CAV-MAE Style)
                                                                           ▼
                                                Attention-Pooled Metric Vector: z ∈ R^128
```

### 3.1 Acoustic Front End (Resolving the Micro-Pitch Limit)
In a 7-year-old child, fundamental phonation frequencies range from $150\text{ Hz}$ to $400\text{ Hz}$. At a $48\text{ kHz}$ sampling rate, a 128-band log-mel filterbank produces bands of approximately $27\text{ Hz}$ width near $300\text{ Hz}$, rendering $\pm 15\text{ Hz}$ micro-pitch shifts sub-bin and unresolvable.

Project N resolves this with a tripartite acoustic engine:
1. **Dedicated Voice Quality & Pitch Tracking:**
   Computes fundamental frequency ($F_0$), local jitter (pitch perturbation), local shimmer (amplitude perturbation), Harmonics-to-Noise Ratio (HNR), and Cepstral Peak Prominence (CPP) using autocorrelation and periodicity tracking (via pYIN or CREPE) at a fine $\sim 10\text{ ms}$ hop interval. This provides $\sim 1\text{ Hz}$ frequency resolution.
2. **Constant-Q Transform (CQT) / ERB Filterbank:**
   Applies geometrically spaced frequency bins where filter bandwidth is proportional to center frequency ($\Delta f / f = Q$). This ensures high spectral resolution in the low-frequency fundamental and formant regions.
3. **Broadband Log-Mel Filterbank:**
   Maintains a 128-band log-mel spectrogram across $20\text{ Hz}$ to $24,000\text{ Hz}$ for capturing broad spectral envelope and vocal tract resonance.
   - STFT parameters: $N = 2048$ (giving $23.44\text{ Hz}$ linear bin spacing at $48\text{ kHz}$), hop length $H = 160$ ($3.333\text{ ms}$), with a periodic Hann window:
     $$w(n) = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N}\right)\right), \quad n = 0, \dots, N-1$$

### 3.2 Kinematic Front End (Pose Over Raw Differencing)
Raw-pixel temporal differencing ($\mathbf{\Delta}_t = \|\mathbf{Z}_t - \mathbf{Z}_{t-1}\|_1$) is highly sensitive to handheld camera shake, auto-exposure adjustments, mains flicker, and room shadows.

Project N establishes a robust kinematic hierarchy:
1. **Body-Relative Pose & Keypoints (Primary):**
   Tracks 33 body landmarks, 21 hand landmarks per hand, and facial contour landmarks using MediaPipe Holistic or BlazePose. Because landmark coordinates are normalized relative to the torso and head centers, they are fundamentally invariant to camera translation, zoom, and background motion.
2. **Dense Optical Flow (Secondary):**
   Computes motion vector fields via RAFT to capture rapid continuous movements (e.g., clothing flutter, peripheral limb trajectories) independent of luminance shifts.
3. **Frame Rate Specification:** Captured at $30\text{ fps}$ (or sampled at $15\text{ fps}$) at $720\text{p}$, which fully satisfies the Nyquist criterion for $3\text{ Hz}$ to $8\text{ Hz}$ motor stims while avoiding the pose-noise overfitting observed at $60\text{ fps}$.

### 3.3 Physiological Front End (Direct Autonomic Correlation)
To ground internal arousal without circular inference from video, Project N integrates wearable telemetry (e.g., Empatica EmbracePlus, or Apple Watch sensor streaming):
1. **Electrodermal Activity (EDA):** Separates skin conductance into tonic baseline level (SCL) and rapid phasic responses (SCR), indexing sympathetic nervous system arousal.
2. **Heart Rate Variability (HRV):** Extracts Root Mean Square of Successive Differences (RMSSD) and High-Frequency (HF) power bands reflecting vagal/parasympathetic modulation.
3. **3-Axis Accelerometry:** Provides continuous wrist/body motion energy, maintaining context when the child moves outside the camera field of view.

---

## 4. Target Architecture: Retrieval-First, Rendering-Only

Project N inverts traditional multimodal generation. The Large Language Model is removed from the primary classification loop and restricted to schema-constrained rendering.

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                L1: MEASURED OBSERVATION                                │
│   Acoustic metrics, pose trajectories, physiological levels, capture quality score      │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       CROSS-MODAL BINDING & METRIC SPACE (MLX)                         │
│   Multimodal Resampler trained on Audio-Visual Temporal Correspondence (CAV-MAE)       │
│   Attention Pooling + L2 Normalization -> 128-dimensional metric vector (z ∈ R^128)    │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   L2: RETRIEVAL & CALIBRATED MATCHING (ChromaDB / SQLite)              │
│   k-NN / Prototype retrieval over historical confirmed episodes of Child N             │
│   Calibrated multi-label probabilities & prediction sets                               │
│   ABSTENTION: If d(nearest) > tau_abstain, emit "Unrecognized Pattern"                 │
│   RED FLAG: If NCCPC-R distress threshold exceeded, emit "Medical Escalation"          │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
              ┌─────────────────────────────┴─────────────────────────────┐
              ▼                                                           ▼
┌───────────────────────────────────────────┐ ┌──────────────────────────────────────────┐
│          CHILD-AUTHORED AAC BRIDGE        │ │        CAREGIVER INTERFACE RENDERER      │
│                                           │ │                                          │
│   Pre-populates candidate options on      │ │   Schema-Constrained LLM (Qwen2.5-14B)   │
│   speech-generating device or AAC choice  │ │   100% Frozen Base Weights (W0)          │
│   board for Child N to select or reject.  │ │   Formats L1-L4 into four distinct       │
│   Child selection outranks adult labels.  │ │   visual evidence layers. No added facts.│
└───────────────────────────────────────────┘ └──────────────────────────────────────────┘
```

### 4.1 Metric Space & Prototypical Learning
The high-dimensional sensory representations are mapped into a compact, 128-dimensional metric space:
$$\mathbf{z} = \text{L2\_Normalize}\left( \text{AttentionPool}(\mathbf{X}_{sensory}) \mathbf{W}_{proj} \right) \in \mathbb{R}^{128}$$

Using a 128-dimensional metric space (rather than 4096 dimensions) prevents geometric collapse when operating with hundreds of labeled historical examples rather than hundreds of thousands. Matching is performed using cosine similarity or Euclidean distance over class prototype centers $\mathbf{c}_k$:
$$\mathbf{c}_k = \frac{1}{|S_k|} \sum_{i \in S_k} \mathbf{z}_i$$

### 4.2 Calibrated Abstention
If the distance between the query vector $\mathbf{z}$ and the nearest historical prototype exceeds a calibrated threshold $\tau_{abstain}$:
$$d(\mathbf{z}, \mathbf{c}_{nearest}) > \tau_{abstain}$$
the system **abstains from classification**. It outputs:
```text
Unrecognized Pattern. Insufficient historical similarity to classify.
Recommended Action: Observe environmental context or present AAC open choice board.
```

---

## 5. Unsupervised Pretraining & Evaluation Protocol

### 5.1 Pretraining via Audio-Visual Temporal Correspondence
Rather than forcing a 64-token bottleneck to reconstruct thousands of raw image patches under 75% masking (which causes severe underfitting on a single-child dataset), pretraining uses **Audio-Visual Temporal Correspondence** (CAV-MAE / AVC style):
- **Objective:** Contrastive pairing determining whether an acoustic segment and a kinematic segment co-occurred synchronously in time versus asynchronously from different time windows.
- **Advantage:** Requires zero manual annotations, learns individualized sensorimotor binding, and runs efficiently on local Apple Silicon hardware.

### 5.2 Preregistered N-of-1 Evaluation Protocol
To prevent data leakage and benchmark gaming, Project N enforces an N-of-1 evaluation protocol:
1. **Time-Separated Splits:** Data is partitioned by **day** or by independent episode with an enforced temporal buffer. Adjacent 5-second windows from the same recording are never split across train and test sets.
2. **Baselines Required with Every Benchmark:**
   - *Metadata-Only Baseline:* Logistic regression on time of day, meal timing, and caregiver antecedent tags.
   - *Shuffled Labels Baseline:* Empirical null distribution verification.
   - *Nearest Historical Episode Baseline:* Raw feature 1-NN lookup.
   - *No-LLM Baseline:* Direct output of retrieved structured records without narrative formatting.
3. **Reported Metrics:** Per-class precision, recall, Macro-F1, calibration error (ECE), and abstention coverage.

---

## 6. Clinical Grounding, Interoception & The AAC Bridge

### 6.1 Clinical Framework Integration
Project N balances multiple clinical and developmental lenses, recognizing each as an interpretative perspective rather than absolute causal truth:
- **Functional Behavior Assessment (FBA):** Analyzes antecedents and behavioral functions across tangible, escape, attention, and sensory reinforcement categories.
- **Predictive Processing (HIPPEA):** Frames repetitive movements as active strategies to minimize sensory prediction error and restore environmental predictability.
- **Interoception:** Evaluates internal bodily signals (hunger, thirst, fatigue, temperature, digestive discomfort) as primary drivers of behavioral state.
- **Ayres Sensory Integration (ASI):** Referenced as one therapeutic lens among several, documented with study population and evidence quality metadata.

### 6.2 The AAC Bridge: Centering Child Authorship
The most critical defect of prior approaches is excluding the child from the communication loop. Project N integrates directly with Augmentative and Alternative Communication (AAC):
- High-probability possibilities are automatically transferred over local Bluetooth/Wi-Fi to Child N's AAC device as candidate icons on an adaptive choice board.
- When Child N taps an option (e.g., "water", "quiet break", "deep pressure") or explicitly rejects all options, this selection is logged as **the authoritative ground truth**, superseding all adult hypotheses.

### 6.3 Validated Distress & Medical Rule-Out (NCCPC-R)
Somatic distress and pain must never be confused with behavioral or sensory preferences. Project N incorporates the **Non-Communicating Children's Pain Checklist – Revised (NCCPC-R)**:
- Evaluates 27 observable items across 6 subscales: Vocal, Emotional, Facial, Body Language, Protective, and Physiological.
- **Red-Flag Escalation:** If the distress score crosses the validated threshold ($\ge 6$ on observed subscales), the system generates an immediate **Medical Escalation Card**, advising caregivers to conduct a medical review for physical pain (e.g., ear infection, dental pain, gastrointestinal reflux, acute injury). Behavioral and sensory interpretations are suppressed.

---

## 7. Continuous Adaptation, Re-fitting & Model Governance

### 7.1 Periodic Full Re-fit Over Nightly SGD
Running nightly Stochastic Gradient Descent with small batches (e.g., 2 novel examples + 8 historical examples) produces high gradient variance and instability in high-dimensional space.

Project N replaces nightly SGD with a **periodic full re-fit**:
- At scheduled intervals (or upon accumulation of $K$ verified examples), the lightweight 128-dimensional metric projection head and prototype clusters are re-fit over the entire verified episodic history.
- On Apple Silicon, re-fitting a 128-dimensional metric head over hundreds of examples executes in seconds on Metal GPU arrays, making catastrophic forgetting structurally impossible.

### 7.2 Model Promotion Gate
No candidate model is deployed to caregiver-facing inference automatically. Promotion requires passing an explicit validation gate:
1. **Holdout Evaluation:** Candidate weights are evaluated on locked, time-separated test splits.
2. **Safety Regression Check:** Zero increase in false reassurance or missed NCCPC-R distress events.
3. **Caregiver Review:** The caregiver reviews performance metrics and approves model promotion.
4. **Lineage & Rollback:** All prior model weights and embedding spaces are versioned, enabling instant one-click rollback.
