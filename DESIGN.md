# Project N: Architectural & Neurobiological Design Document
## Theoretical Formulations, Sensory Encoder Bypass Philosophy, and Continuous Adaptation Architecture

---

## 1. Biological and Neurological Foundations

### 1.1 Neurobiology of Level 3 Non-Verbal Autism Spectrum Disorder
Level 3 Autism Spectrum Disorder (ASD), as delineated by the Diagnostic and Statistical Manual of Mental Disorders (DSM-5), represents individuals requiring "very substantial support." For non-verbal or minimally speaking children—exemplified by Nolan, a 7-year-old Level 3 autistic child—expressive language is characterized by an absence of functional spoken words, severe apraxia of speech (CAS), or an inability to coordinate the complex oromotor musculature required for phonemic articulation.

Crucially, the absence of verbal speech does not indicate the absence of language, cognition, or communicative intent. The neurological landscape of Level 3 ASD involves:
- **Atypical Synaptic Pruning:** Hyper-connectivity within local sensory cortical microcircuits (hyper-reactivity) paired with hypo-connectivity across long-range associative pathways (e.g., fronto-striatal and fronto-temporal tracts).
- **Altered Cerebellar Circuitry:** Morphological variations in Purkinje cells within the cerebellum impair predictive motor sequencing, dynamic sensorimotor feedforward control, and temporal synchrony.
- **Autonomic Nervous System (ANS) Dysregulation:** Baseline vagal tone is frequently suppressed, resulting in sympathetic nervous system hyper-arousal. The child constantly navigates fluctuating states of physiological equilibrium, moving rapidly between baseline regulation, sensory seeking, hyper-arousal, and impending sensory meltdown.

### 1.2 Sensory Processing Differences & Regulatory Stimming
Sensory processing in Level 3 ASD diverges sharply from neurotypical profiles across vestibular, proprioceptive, auditory, and tactile axes:
1. **Proprioceptive & Vestibular Seeking:** The brain receives attenuated feedback from muscle spindles and joint receptors. To orient the body in physical space and ground neurological equilibrium, the child exhibits stereotypic motor behaviors ("stims")—such as rapid wrist rotation (3 Hz to 6 Hz), repetitive finger-flicking within the peripheral visual field, vertical jumping, or rhythmic head movements.
2. **Sensory Overload & Auditory Hyper-Reactivity:** The auditory thalamocortical pathway fails to habituate to repetitive background sensory stimuli (aberrant sensory gating). A fluorescent light hum, HVAC cycling, or sudden high-frequency auditory spikes register not as ambient background noise, but as acute somatic distress.
3. **Self-Regulatory Mechanics:** Motor stimming and continuous tonal vocalizations are not purposeless pathology. They serve as active physiological closed-loop compensation mechanisms designed to downregulate sympathetic hyper-arousal or upregulate under-stimulated sensory pathways.

### 1.3 The Communicative Substrate of Idiosyncratic Vocalizations
In the absence of phonemic speech, acoustic communication operates via raw paralinguistic and somatic sound generation:
- **Continuous Tonal Hums:** Sustained phonations where fundamental frequency ($F_0$), harmonic spacing, and jitter/shimmer index internal homeostatic states. A stable, low-variance $F_0$ typically reflects self-soothing or deep proprioceptive focus.
- **Micro-Pitch and Formant Dynamics:** Shifts in glottal pulse velocity and subglottal pressure induce subtle pitch variations ($\pm 15$ to $50\text{ Hz}$) that standard speech parsers dismiss as pitch instability. In an individual child, these micro-pitch trajectories reliably delineate discomfort, cognitive curiosity, or boundary protests.
- **Guttural Resonance & Non-Laryngeal Phonation:** Pharyngeal friction, clicks, glottal stops, and explosive aspirates convey urgency and affective valence. Because these vocalizations bypass traditional articulatory targets (lips, tongue-tip, alveolar ridge), their information density is concentrated in the spectral envelope, harmonic decay rates, and temporal envelopes.

---

## 2. The "Rosetta Stone" Fallacy & Neurotypical Inductive Bias

### 2.1 The Failure of Commercial Foundation Encoders
Standard multimodal foundation models (e.g., Whisper, Audio Spectrogram Transformer / AST, CLIP, SigLIP, LLaVA, Qwen-VL) rely on strong inductive biases optimized for neurotypical communication:

```text
Standard Foundation Pipeline (Information Annihilation):
Raw Expressive Vocalization ──► [Whisper ASR: Discrete Phonemes] ──► "[Silence]" / "[Unintelligible]"
High-Frequency 4Hz Stimming ──► [Standard ViT: Spatial Pooling]  ──► "Child sitting on carpet"
```

1. **Acoustic Phonemic Discretization (The Whisper Failure):**
   Automatic Speech Recognition (ASR) systems are trained on thousands of hours of spoken human languages where the loss function penalizes acoustic variance that does not map to discrete linguistic tokens (phonemes, subwords). Whisper's encoder discards:
   - Non-lexical harmonic resonances.
   - Sustained vowel-like hums without consonant transitions.
   - Pitch contour variations in unvoiced or glottal sounds.
   The resulting transcription is either a blank token, an erroneous phonetic hallucination, or a generic label like `"[Music]"` or `"[Laughter]"`.
2. **Visual Spatial & Temporal Pooling (The Standard ViT Failure):**
   Commercial Vision Transformers downsample spatial patches ($16 \times 16$ or $14 \times 14$) and pool across temporal frames (e.g., 1 frame per second). A rapid, 4 Hz wrist-flick or finger-tremor occurring over 250 milliseconds spans only 1 or 2 downsampled frames and is spatially blended into the static background pixels of the room. The ViT outputs semantic tokens representing the static scene ("living room", "child sitting"), completely obliterating the communicative motor signal.

### 2.2 The Fallacy of Text-Only Intermediate Captioning
A common failure mode in multimodal translation is two-stage pipeline design: Stage 1 creates textual descriptions ("Child is humming at 180Hz and rotating left hand"), and Stage 2 prompts an LLM with these strings. 

This intermediary text bottleneck introduces catastrophic semantic loss:
- A textual description cannot convey the phase coherence, spectral centroid drift, or rhythmic micro-variations of a stim.
- Textual descriptors force continuous, non-linear neurodivergent expressions into arbitrary categorical buckets created by neurotypical observers.
- Direct latent sensory-to-LLM conditioning is essential. The LLM must attend directly to continuous sensory latents rather than discrete lexical approximations.

---

## 3. Latent Space Geometry & Sensory Encoder Bypass

Project N eliminates neurotypical encoders from the primary sensory loop. Instead, it deploys two specialized, raw sensory extraction engines:

```text
RAW ACOUSTIC PATHWAY:
Audio (48kHz WAV) ──► STFT (n_fft=2048, H=160) ──► 128 Mel Bands ──► AaPE Embedding ──► X_audio ∈ R^(B x 2048 x 768)

RAW KINEMATIC PATHWAY:
Video (60fps Frames) ──► Temporal-L1 Diff Mask (Δt) ──► Spatial Patch ──► TD-ViT ──────► X_kinematic ∈ R^(B x 1500 x 1024)
```

### 3.1 ND-AST: Neurodivergent Audio Spectrogram Transformer

To preserve micro-pitch modulations, sub-harmonic overtones, and continuous acoustic resonance, the acoustic pathway operates over high-resolution time-frequency representations:

#### 3.1.1 Time-Frequency Transform
Given a raw discrete audio signal $s \in \mathbb{R}^T$ sampled at $f_s = 44,100\text{ Hz}$ (or $48,000\text{ Hz}$):
1. **Short-Time Fourier Transform (STFT):**
   $$S(m, k) = \sum_{n=0}^{N-1} s(n + mH) \cdot w(n) e^{-j \frac{2\pi}{N} k n}$$
   where $N = n_{fft} = 2048$ (providing frequency resolution $\Delta f \approx 21.5\text{ Hz}$), $H = 160$ (hop length yielding temporal resolution $\Delta t \approx 3.63\text{ ms}$), and $w(n)$ is a Hann window function:
   $$w(n) = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right)$$
2. **Mel-Frequency Filterbank:**
   The complex spectrogram is converted to power spectral density and projected across $F = 128$ triangular Mel filters covering $20\text{ Hz}$ to $22,050\text{ Hz}$:
   $$\mathbf{M}(m, f) = \ln\left( \sum_{k} |S(m, k)|^2 \cdot \Phi_{mel}(f, k) + \epsilon \right)$$
   where $\Phi_{mel}$ represents the Mel weighting matrix and $\epsilon = 10^{-6}$ prevents numerical instability.

#### 3.1.2 Aliasing-aware Patch Embedding (AaPE)
Standard AST systems use large 2D patch strides ($16 \times 16$) that induce frequency aliasing across narrow harmonic bands. In ND-AST, the patch embedding operates with a small stride in frequency and an overlapping temporal window:
- Patch size: $P_t = 16$ time steps ($\approx 58\text{ ms}$), $P_f = 8$ frequency bins ($\approx 172\text{ Hz}$).
- Stride: $S_t = 8$ time steps ($50\%$ temporal overlap), $S_f = 4$ frequency bins ($50\%$ spectral overlap).
- Projection: Each flattened patch $\mathbf{p} \in \mathbb{R}^{P_t \cdot P_f}$ is projected via a 2D convolution into embedding dimension $d_a = 768$.
- Sinusoidal 2D Positional Embeddings $\mathbf{E}_{pos}^{(a)}$ are added, producing the final acoustic sensory sequence:
  $$\mathbf{X}_{audio} \in \mathbb{R}^{B \times L_a \times 768} \quad \text{where } L_a = 2048$$

### 3.2 TD-ViT: Temporal Difference Vision Transformer

Standard vision backbones are dominated by static spatial tokens (background furniture, lighting, room boundaries). To extract Nolan's motor stimming without interference from static scenery, TD-ViT applies temporal differential masking.

#### 3.2.1 Temporal-L1 Difference Masking
Given consecutive decoded video frames $\mathbf{Z}_t, \mathbf{Z}_{t-1} \in \mathbb{R}^{H \times W \times C}$ captured at $60\text{ fps}$:
1. **Pixel-Level Differential:**
   $$\mathbf{\Delta}_t(i, j) = \frac{1}{C} \sum_{c=1}^C \left| \mathbf{Z}_t(i, j, c) - \mathbf{Z}_{t-1}(i, j, c) \right|$$
2. **Patch Motion Energy Calculation:**
   The frame is partitioned into non-overlapping patches $p_{k, t}$ of dimension $16 \times 16$. The kinetic energy $\mathcal{E}(p_{k, t})$ of patch $k$ is the spatial integral over its differential mask:
   $$\mathcal{E}(p_{k, t}) = \frac{1}{256} \sum_{(i,j) \in p_{k,t}} \mathbf{\Delta}_t(i, j)$$
3. **Threshold Filtering:**
   A learned or calibrated kinetic threshold $\tau$ separates active movement from static background:
   $$\mathcal{M}(p_{k, t}) = \begin{cases} 
   p_{k, t} & \text{if } \mathcal{E}(p_{k, t}) \ge \tau \\ 
   \mathbf{0} & \text{if } \mathcal{E}(p_{k, t}) < \tau 
   \end{cases}$$
   Patches with energy below $\tau$ are dynamically dropped from the computation graph, reducing sequence length while retaining the precise trajectory of hands, face, fingers, and torso.

#### 3.2.2 Kinematic Feature Projection
The surviving motion-active patches are linearly projected into $d_v = 1024$ and augmented with 3D Spatio-Temporal Positional Encodings (temporal index + spatial $x, y$ coordinates). After processing through 12 Transformer encoder layers with dynamic attention masking, the resulting kinematic representation is:
$$\mathbf{X}_{kinematic} \in \mathbb{R}^{B \times L_v \times 1024} \quad \text{where } L_v = 1500$$

---

## 4. The Dimensional Bottleneck: Perceiver Resampler

### 4.1 Token Saturation & KV Cache Explosion
Concatenating the raw acoustic and visual token streams yields:
$$L_{total} = L_a + L_v = 2048 + 1500 = 3548 \text{ tokens per 5-second window}$$

Feeding 3,548 continuous tokens into a 14B parameter LLM for every inference window causes:
1. **Memory Exhaustion:** Dynamic Key-Value (KV) cache allocation scales quadratically with sequence length, instantly breaching our $36.0\text{ GB}$ VRAM limit.
2. **Attention Dilution:** The attention weights across 3,500+ unconstrained sensory tokens become diffuse, degrading the LLM's capacity to synthesize coherent clinical deductions.
3. **Inference Latency:** Autoregressive decoding over thousands of sensory prefix tokens slows generation below interactive speeds on local hardware.

```text
[X_audio (2048)] ──┐
                   ├─► Concatenated Sequence (3548 tokens) ─┐
[X_kinematic (1500)]┘                                       │
                                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PERCEIVER RESAMPLER (ND PROJECTOR)                           │
│                                                                                 │
│   Learnable Latent Queries: Q_latent ∈ R^(64 x 4096)                            │
│   Cross-Attention: Queries attend to Concatenated Sensory Tokens (3548 x 4096)  │
│   FFN Block + Residual Connections (LayerNorm, GELU)                            │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
               Compressed Sensory Prefix: Z_sensory ∈ R^(B x 64 x 4096)
                                        │
                                        ▼
               Injected Directly into LLM Prefix Attention Space
```

### 4.2 Perceiver Resampler Mathematical Architecture
To create an information bottleneck, Project N implements a custom MLX Perceiver Resampler (`NDProjector`).

Let $\mathbf{X}_{sensory} \in \mathbb{R}^{B \times 3548 \times 4096}$ be the concatenated, dimension-aligned acoustic and kinematic sensory sequence:
$$\mathbf{X}_{sensory} = \left[ \mathbf{W}_a \mathbf{X}_{audio} \;\|\; \mathbf{W}_v \mathbf{X}_{kinematic} \right]$$
where $\mathbf{W}_a \in \mathbb{R}^{768 \times 4096}$ and $\mathbf{W}_v \in \mathbb{R}^{1024 \times 4096}$ are linear projection matrices.

1. **Latent Query Matrix:** We instantiate $M = 64$ learnable query embeddings $\mathbf{Q}_{latent} \in \mathbb{R}^{M \times 4096}$, initialized from a truncated normal distribution $\mathcal{N}(0, 0.02)$.
2. **Cross-Attention Mechanism:**
   $$\mathbf{Q} = \mathbf{Q}_{latent} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}_{sensory} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}_{sensory} \mathbf{W}_V$$
   where $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{4096 \times 4096}$.
   The scaled dot-product cross-attention computes:
   $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$
   accelerated via Metal Performance Shaders (`mx.fast.scaled_dot_product_attention`).
3. **Feed-Forward Block & Layer Normalization:**
   $$\mathbf{Z}^{(1)} = \text{LayerNorm}\left( \mathbf{Q}_{latent} + \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) \right)$$
   $$\mathbf{Z}_{sensory} = \text{LayerNorm}\left( \mathbf{Z}^{(1)} + \text{FFN}(\mathbf{Z}^{(1)}) \right)$$
   where $\text{FFN}(\mathbf{u}) = \text{GELU}(\mathbf{u} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$ with intermediate dimension $d_{ff} = 16384$.

The output $\mathbf{Z}_{sensory} \in \mathbb{R}^{B \times 64 \times 4096}$ compresses the sensory stream by a factor of $55.4\times$ while capturing the essential multi-modal correlations.

---

## 5. The 48-Hour Unsupervised Baseline Initialization Protocol

Before any caregiver annotations or intent labels are introduced, the system must establish an unsupervised baseline representation of Nolan's motor and acoustic repertoire.

```text
50 Hours of Unannotated Baseline Video ──► Patch Segmentation & Feature Projection
                                                    │
                                                    ▼
                                    75% Uniform Masking Strategy
                                    ├─ Masked Patches ──► Latent Null Token
                                    └─ Visible 25%    ──► ND-AST & TD-ViT Encoders
                                                    │
                                                    ▼
                                    Perceiver Resampler Compression (64 Latents)
                                                    │
                                                    ▼
                                    Lightweight MLX MAE Decoder
                                                    │
                                                    ▼
                       MSE Reconstruction: L_MAE = ||X_recon - X_target||2
```

### 5.1 The 50-Hour Raw Intake Protocol
The family captures approximately 50 hours of raw, unconstrained daily activity across 7 days:
- Baseline home environment (playroom, kitchen, outdoor backyard).
- Multiple lighting conditions, room acoustics, and emotional states (calm, tired, hungry, engaged).
- Zero requirement for caregiver tagging or interpretation during this phase.

### 5.2 Masked Autoencoding (MAE) Formulation
We apply a high masking ratio ($75\%$) across both time-frequency acoustic patches and kinematic motion patches:
1. **Random Patch Masking:**
   Let $\mathcal{P} = \{p_1, p_2, \dots, p_N\}$ be the full set of sensory patches. A uniform random permutation partitions $\mathcal{P}$ into:
   - Visible patches $\mathcal{P}_{vis}$ ($25\%$).
   - Masked patches $\mathcal{P}_{mask}$ ($75\%$).
2. **Encoder Encoding:** Only $\mathcal{P}_{vis}$ tokens pass through the ND-AST and TD-ViT transformer blocks.
3. **Latent Infilling:** The output representations of $\mathcal{P}_{vis}$ are concatenated with learnable mask tokens $\mathbf{e}_{mask} \in \mathbb{R}^{4096}$ at the masked positions, restoring the full sequence layout.
4. **Resampler Bottlenecking:** The combined sequence is processed through the Perceiver Resampler to produce the 64 latent representations $\mathbf{Z}_{sensory}$.
5. **Reconstruction Objective:** A lightweight 4-layer Transformer decoder projects the 64 latents back to the original sensory patch space to reconstruct the masked acoustic spectrogram bins and kinematic differential pixels:
   $$\mathcal{L}_{MAE} = \frac{1}{|\mathcal{P}_{mask}|} \sum_{i \in \mathcal{P}_{mask}} \left\| \hat{p}_i - p_i \right\|_2^2$$

### 5.3 Outcome of Baseline Initialization
This 48-hour unsupervised training process (executed across two overnight sessions on the M5 Pro) yields:
- An acoustic encoder tuned to Nolan's individual vocal resonances, formants, and harmonic structures.
- A kinematic encoder tuned to Nolan's physical motor velocity, stim trajectories, and spatial range.
- A 64-token Perceiver Resampler that condenses his baseline behaviors into an organized behavioral topology.

---

## 6. Clinical Grounding Strategy

A generic LLM prompted with continuous sensory tokens tends to hallucinate neurotypical conversational interpretations (e.g., assuming a loud vocalization indicates willful defiance or standard verbal conversation). Project N enforces clinical grounding through a two-tier mechanism:

```text
                               ┌────────────────────────────────────────────────────────┐
                               │                    CLINICAL LENS                       │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                    ┌─────────────────────────────────────┴─────────────────────────────────────┐
                    ▼                                                                           ▼
┌───────────────────────────────────────────────┐               ┌───────────────────────────────────────────────┐
│        TIER 1: OFFLINE DOMAIN LoRA            │               │      TIER 2: INFERENCE CLINICAL RAG           │
│                                               │               │                                               │
│   Target: Qwen2.5-14B Attention & MLPs        │               │   Engine: ChromaDB Persistent Vector Store    │
│   Rank r=64, Alpha=128, FP16 Metal Tensors    │               │   Embedder: nomic-embed-text-v1.5 (Local)     │
│   Ingests:                                    │               │   Ingests:                                    │
│   ├─ Ayres Sensory Integration (ASI) Theory   │               │   ├─ Antecedent-Behavior-Consequence (ABC) logs│
│   ├─ DIR/Floortime Developmental Frameworks   │               │   ├─ Speech-Language Pathology (SLP) evaluations│
│   └─ Functional Behavior Assessment (FBA)     │               │   └─ Nolan's Individualized Education Program │
│                                               │               │                                               │
│   Role: Restructures LLM cognitive priors to  │               │   Role: Injects immediate, relevant clinical  │
│   reason through sensory homeostasis.         │               │   precedents into the active prompt window.   │
└───────────────────────────────────────────────┘               └───────────────────────────────────────────────┘
```

### 6.1 Tier 1: Domain-Specific Clinical LoRA
We fine-tune low-rank adapter matrices ($\mathbf{A}, \mathbf{B}$) on the base LLM weights ($\mathbf{W}_0$):
$$\mathbf{W} = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B} \mathbf{A}$$
where $\mathbf{W}_0 \in \mathbb{R}^{d_{in} \times d_{out}}$ is frozen in 4-bit quantization, $\mathbf{B} \in \mathbb{R}^{d_{in} \times r}$, $\mathbf{A} \in \mathbb{R}^{r \times d_{out}}$, rank $r = 64$, and scaling factor $\alpha = 128$.

The LoRA training corpus consists of curated clinical literature:
- Classical and modern texts on Ayres Sensory Integration (ASI).
- Functional Behavior Assessment (FBA) protocols detailing antecedent-behavior-consequence patterns.
- Pediatric occupational therapy literature on proprioceptive, vestibular, and sensory regulation.
- Clinical case studies on Childhood Apraxia of Speech and Level 3 non-verbal communication.

This permanently shifts the model's inductive reasoning: the LLM treats high-energy vocalizations not as conversational turns, but as potential indicators of sensory overload, vestibular seeking, or autonomic dysregulation.

### 6.2 Tier 2: Dynamic Clinical RAG (ChromaDB)
At inference time, the model queries a local ChromaDB instance running on the M5 Pro:
- Collection `clinical_literature`: Stores chunked, semantically indexed excerpts from Nolan's occupational therapy (OT) evaluations, speech-language pathology (SLP) assessments, and individualized functional behavioral goals.
- Embedding Model: `nomic-embed-text-v1.5` executing locally, projecting text queries into a 768-dimensional metric space.
- Retrieved Context: The top-$k$ ($k=3$) relevant clinical chunks are injected into the LLM system prompt alongside the sensory prefix tokens, ensuring every translation is grounded in Nolan's clinical history.

---

## 7. Continuous Adaptation & Preventing Catastrophic Forgetting

A non-verbal child's communicative repertoire evolves continuously. As Nolan develops new motor stims or adapts his vocalizations, the system must learn novel associations without overwriting historical understandings.

```text
                              CAREGIVER INTERACTION (FLUTTER APP)
                                               │
                                               ▼
                              Daily Video Clip + Caregiver Tag:
                          "Deep proprioceptive squeeze requested"
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                      NIGHTLY CONTRASTIVE TRIPLET LEARNING                               │
│                                                                                         │
│   Batch Composition: 80% Historical Replay Vectors  +  20% Novel Daily Vectors          │
│                                                                                         │
│   Triplet Selection:                                                                    │
│   ├─ Anchor (a): Today's 64-token Sensory Latent Vector (Z_sensory)                     │
│   ├─ Positive (p): Historical Vector with matching intent ("proprioceptive seeking")    │
│   └─ Negative (n): Historical Vector with divergent intent ("auditory overload")        │
│                                                                                         │
│   Loss: L_triplet = max( d(a, p) - d(a, n) + α, 0 ) where d(u,v) = 1 - cos(u, v)        │
│                                                                                         │
│   Gradient Descent: Updates ONLY Resampler Latents and LoRA Adapter Matrices            │
│   Base 14B Weights (W0): 100% Frozen in Metal Unified Memory                            │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 7.1 Parent-in-the-Loop Micro-Annotation
Caregivers interact with a lightweight Flutter companion app:
1. **Instant Clip Review:** The app alerts parents when a distinctive, unclassified sensory pattern is observed.
2. **One-Tap Micro-Tagging:** The caregiver provides context via quick-select buttons or a brief voice note:
   - Primary Affect: `[Regulated | Seeking | Overwhelmed | In Pain]`
   - Functional Intent: `[Water/Drink | Deep Pressure | Sensory Break | Protest | Connection]`
   - Environmental Antecedent: `[Post-school fatigue | Loud sibling | Transitioning rooms]`
3. **Ingestion:** Metadata and raw audio-video segments are transmitted over local Wi-Fi to the Mac M5 Pro FastAPI daemon and queued for nightly consolidation.

### 7.2 Contrastive Triplet Loss Formulation
During the nightly optimization cycle, the projection layers and LoRA adapters are tuned using Contrastive Triplet Loss:
$$\mathcal{L}_{triplet} = \max\left( d(\mathbf{a}, \mathbf{p}) - d(\mathbf{a}, \mathbf{n}) + \alpha_{margin}, \; 0 \right)$$
where:
- $\mathbf{a} = \text{mean\_pool}(\mathbf{Z}_{anchor}) \in \mathbb{R}^{4096}$ is the anchor sensory latent embedding from the daily recording.
- $\mathbf{p} = \text{mean\_pool}(\mathbf{Z}_{positive}) \in \mathbb{R}^{4096}$ is a positive sample from the historical database matching the caregiver-verified intent label.
- $\mathbf{n} = \text{mean\_pool}(\mathbf{Z}_{negative}) \in \mathbb{R}^{4096}$ is a hard-negative sample from the historical database possessing a distinct intent label but similar acoustic or visual characteristics.
- $\alpha_{margin} = 0.25$ is the separation margin.
- $d(\mathbf{u}, \mathbf{v})$ is the Cosine Distance metric:
  $$d(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

### 7.3 RAG-Backed Replay Buffer (The 80/20 Invariant)
To prevent catastrophic forgetting (overwriting previously mastered intents while adapting to new daily data), the nightly training loop enforces a strict batch composition:
- **$80\%$ Historical Replay Exemplars:** Randomly and hard-mined samples drawn from the ChromaDB collection `nd_communicative_intents`.
- **$20\%$ Novel Daily Samples:** The fresh clips and caregiver annotations gathered during the preceding 24-hour cycle.

This 4:1 historical-to-novel ratio anchors the gradient trajectory, ensuring that updates refine existing cluster boundaries without distorting the established latent topology.
