# Project N — Review Refinements

**Status:** working document. Merged output of two independent scientific/engineering reviews of the
specification set (`README.md`, `DESIGN.md`, `SPECS.md`, `INVARIANTS.md`, `AGENTS.md`) at commit `ebb0710`.
**Date:** 2026-09-08
**Repository state at review:** documentation only; no code yet.

This is a research and architecture review. It is not an assessment of any child, and it is not medical,
legal, or regulatory advice.

## How to use this document

- **Findings** (`F-nn`) are defects or risks. Each states the claim, the evidence, and the fix.
- **Actions** (`P0`/`P1`/`P2`) are checkboxed and reference the findings they close.
- **Numbers** in §4 were recomputed from the formulas in the specification documents against the
  published model configuration. Where two independent reviews recomputed the same quantity and agreed,
  it is marked `[confirmed x2]`.
- Nothing here should be promoted into `INVARIANTS.md` until it has earned that status empirically. See §11.

---

## 1. Executive summary

Five commitments in the project are worth preserving unchanged: respect for non-speaking communication,
individualized rather than population-level modeling, local-first handling of exceptionally sensitive data,
multimodal sensing, and caregiver participation.

Three problems block the current design.

1. **No objective connects the sensory latents to language.** The three specified losses — MAE, clinical
   LoRA, nightly triplet — do not touch each other. The nightly triplet loss is computed upstream of the
   LLM, so it cannot produce a LoRA gradient at all. See `F-01`.
2. **The acoustic front end cannot resolve the phenomenon it was built for.** The stated signal is
   ±15–50 Hz micro-pitch; the specified 128-band mel filterbank has ~27 Hz bands in the relevant range.
   See `F-03`.
3. **There is no definition of success.** No baseline, no evaluation protocol, no abstention state, and
   labels that are the caregiver's own interpretation being fed back to the caregiver as apparent
   confirmation. See `F-11`, `F-12`, `F-13`.

The appropriate near-term product is a **private, caregiver- and child-controlled insight and
communication-support tool**, not an intent translator. It should surface as much as the evidence supports:
recurring signal patterns, comparable caregiver-verified episodes, relevant personal context, source-linked
research perspectives, calibrated possibilities, and AAC- or observation-based ways to confirm them. It must
not present a diagnosis, or an unverified inference of pain, autonomic state, or intent, as fact. It must not
silently change its deployed model.

### 1.1 Product contract

Project N is a parent-owned private assistant. Its question is:

> Given this clip, this child's history, and a curated local evidence library — what is worth noticing,
> what has helped in similar situations, what are the reasonable possibilities, and how can the parent
> and child confirm or reject them?

Every statement the system makes must be traceable to exactly one of four layers, and the interface must
keep them visually distinct rather than letting fluent prose blur them:

| Layer | Source | Example |
| :-- | :-- | :-- |
| L1 | Measured features in the current clip | "Sustained vocalization, 4.2 s, F0 mean 268 Hz, low variance" |
| L2 | Comparable episodes in this child's own history | "3 prior caregiver-confirmed episodes; 2 resolved with deep pressure" |
| L3 | Explicit caregiver context and recorded outcome | "Tagged post-school; water offered; accepted" |
| L4 | A cited, versioned source from the evidence library | "Per [source, 2019, p.4] — scope: n=32, ages 4–12" |

No output may combine layers without labelling them, and no output may assert anything that is not in one of them.

---

## 2. What is well founded

### 2.1 Retain as stated

| Premise | Assessment |
| :-- | :-- |
| Non-speaking does not imply absence of language, cognition, agency, or communicative intent | Sound. Consensus position. |
| Vocalizations, movement, sensory context, and antecedents can carry useful individual information | Plausible and testable as an N-of-1 measurement hypothesis. |
| Stimming can be regulatory, and sometimes social or communicative | Supported in broad terms. Functions are plural, context-dependent, person-specific. |
| A Perceiver-style resampler is a valid sequence-compression family | Sound architecture family, but not at a fixed 64 tokens and not without alignment training. |
| Self-supervision and replay are reasonable directions for low-label personalized learning | Reasonable research direction; not a demonstrated solution here. |
| Local-first processing is privacy-positive | Correct posture. Necessary but not sufficient — see §8. |

### 2.2 The performance envelope — what "working" looks like

These are the quantitative anchors the specification currently lacks. They validate the premise, set a
realistic target, and define the baselines that must be beaten.

| Result | Source | Implication for Project N |
| :-- | :-- | :-- |
| 7,077 labeled non-verbal vocalizations, 8 minimally speaking individuals, labeled in real time by a close family member | ReCANVo (Johnson et al., 2023) | Almost exactly this project's data design. The premise is validated, not speculative. |
| Speaker-dependent 5-way function classification: UAR **0.75 / 0.53 / 0.79** for three individuals | Narain et al., 2022 | **This is the realistic target.** Achieved with conventional spectral features plus SVM/random forest — not a 14B LLM. |
| Aggression forecast 1 min ahead from 3 min of wearable physiology: AUROC **0.71** population vs **0.84** person-dependent, n=20, 85% minimally verbal | Goodwin et al., 2023 | Quantifies the personalization gain. Also the strongest argument for adding a physiological channel (`F-07`). |
| Pose-derived features on the public stimming benchmark: 97.5% (LSTM) / 98.75% (GRU) | SSBD frame-rate ablation | Pose is robust and cheap; see `F-05`. |
| Accuracy **peaks at 15-frame sampling**, not every frame (LSTM 90.0% → 97.5%), at ~93% less compute | SSBD frame-rate ablation | 60 fps is not merely unnecessary, it is counterproductive; see `F-06`. |
| Aided AAC: improved vocal output in **87%** of participants, no study reporting decreased speech | NDBI + aided AAC meta-analysis | AAC does not suppress speech. Removes the main objection to `P0-4`. |
| Population autism-vs-control voice features: Cohen's d 0.4–0.5, **61–64%** discriminatory accuracy | Fusaroli et al., 2017 | Cited for contrast: *population* prosody markers are weak. This is a different question from within-child decoding, and the distinction must be kept straight in the docs. |

**Read 2.2 together:** person-specific models substantially outperform population models, and the honest
ceiling for this task is roughly 0.75 UAR on a handful of classes — not a fluent narrative of a child's
inner state.

---

## 3. Findings register

| ID | Severity | Area | Summary |
| :-- | :-- | :-- | :-- |
| `F-01` | **Blocking** | Training graph | Triplet loss is upstream of the LLM; `∂L/∂(LoRA) = 0`. No objective aligns latents to language. |
| `F-02` | **Blocking** | Pretraining | MAE-through-a-64-token-bottleneck is a far harder task than MAE and will underfit. |
| `F-03` | **Blocking** | Acoustics | Mel filterbank cannot resolve the stated ±15 Hz micro-pitch signal. |
| `F-04` | **Blocking** | Model interface | `d_model=4096` is incompatible with Qwen2.5-14B (`hidden_size=5120`). |
| `F-05` | High | Vision | Raw-pixel differencing responds mostly to camera motion, exposure, and flicker. |
| `F-06` | Medium | Capture | 60 fps / 1080p is unnecessary and empirically counterproductive. |
| `F-07` | High | Sensing | The design reasons about autonomic arousal but has no channel that measures it. |
| `F-08` | High | Continual learning | 2 labels/night is ~1 gradient step/night; nightly SGD is the wrong shape for this data. |
| `F-09` | High | Retrieval | Embedding drift invalidates historical nearest-neighbour geometry. |
| `F-10` | Medium | Metric learning | Mean-pooling 64 latents discards the structure the resampler exists to build. |
| `F-11` | **Blocking** | Validity | Labels are the caregiver's interpretation; the system learns the caregiver, not the child. |
| `F-12` | High | Validity | Clip-level cross-validation will leak; no baselines are defined. |
| `F-13` | High | Safety | "In Pain" is in the tag enum with no medical rule-out path. |
| `F-14` | High | Safety | Fluent generation with no abstention or calibration path. |
| `F-15` | Medium | Clinical framing | Group-level neuroscience is presented as individual causal fact. |
| `F-16` | Medium | Clinical framing | The clinical prior is one generation behind and omits AAC entirely. |
| `F-17` | Medium | Grounding | Bulk LoRA training on undifferentiated literature is the wrong mechanism for facts. |
| `F-18` | Medium | Governance | Offline execution does not address consent, third parties, retention, or output claims. |
| `F-19` | Medium | Process | Invariants freeze untested hyperparameters, preventing the search the project needs. |
| `F-20` | Low | Correctness | Two documented API idioms do not exist in MLX; one fails silently. |

### F-01 — No objective connects the sensory latents to language `[Blocking]`

*Refs: `DESIGN.md` §5, §7; `SPECS.md` §4, §5*

Trace the three specified objectives:

```
MAE (DESIGN §5)       pixels/mel -> encoders -> resampler -> decoder -> pixels
                      Trains encoders + resampler. Never touches the LLM.

LoRA (DESIGN §6.1)    clinical text -> LLM -> next-token loss
                      Trains LoRA on text. Never touches the sensory path.

Triplet (DESIGN §7)   Z_sensory -> mean_pool -> cosine margin
                      mean_pool(Z_sensory) is the resampler OUTPUT, upstream of
                      the LLM. Therefore d(L_triplet)/d(LoRA) = 0, exactly.
```

`DESIGN.md` §7 states that the nightly step "updates ONLY Resampler Latents and LoRA Adapter Matrices."
This is not achievable: the LoRA sits downstream of where the loss is computed, so no gradient reaches it.

The deeper problem is that **no objective is a captioning objective**. Flamingo's resampler works inside a
system trained end-to-end for vision-language alignment; LLaVA uses multimodal instruction tuning on the
order of 600K pairs for its alignment stage. Project N provides zero paired examples. A ~209M-parameter
projector emitting into a frozen 14B model's prefix, never trained against any language target, produces
64 vectors that are noise from the LLM's point of view. The model will generate fluent, specific,
confident, and **input-independent** clinical prose.

**Fix.** Take the LLM out of the inference path (see §5). If direct multimodal conditioning is still wanted
later, run it as an explicit research arm with a trained connector and matched labeled examples — it is not
a free consequence of prefix concatenation.

### F-02 — The MAE task as specified will underfit `[Blocking]`

*Refs: `DESIGN.md` §5.2*

In standard MAE the decoder receives **all visible tokens** plus mask tokens; the bottleneck is the masking
ratio, not the token count. `DESIGN.md` §5.2 routes the sequence through the 64-latent resampler and
reconstructs from there — asking a 4-layer decoder to rebuild roughly 2,660 masked patches from 64 vectors.
That is a ~40:1 reconstruction ratio stacked on top of 75% masking. Expect underfitting, or collapse to
reconstructing the room's mean appearance.

**Fix.** Split the objectives. Pretrain each encoder with a standard per-modality MAE (visible tokens
straight to the decoder). Train the resampler on a task it suits: **audio-visual temporal correspondence** —
does this vocalization co-occur with this movement? — in the CAV-MAE / AVC style. It needs no labels, it is
the cross-modal binding the architecture depends on, and it is well matched to single-subject data.

### F-03 — The acoustic front end cannot resolve micro-pitch `[Blocking]`

*Refs: `DESIGN.md` §1.3, §3.1; `SPECS.md` §1.2; `INVARIANTS.md` §4*

`DESIGN.md` §1.3 states the communicative signal lives in micro-pitch trajectories of ±15 to 50 Hz.
`DESIGN.md` §3.1 specifies a 128-band log-mel filterbank over 20 Hz – 22,050 Hz. Computing mel band widths
in a 7-year-old's phonation range:

```
Mel band width, 128 bands over 20-22050 Hz
  @ 150 Hz   22.9 Hz          @ 400 Hz   29.2 Hz
  @ 250 Hz   25.5 Hz          @ 800 Hz   40.2 Hz
  @ 300 Hz   26.9 Hz

Signal the design intends to capture ....  +/- 15 Hz
Smallest resolvable step ................  ~27 Hz  (one band)
```

A 15 Hz shift is **sub-bin** — it does not survive the filterbank. The raw STFT does not rescue it either:
`n_fft=2048` gives 21.5 Hz linear resolution, also coarser than the effect.

This front end is the one Whisper uses. The differences specified here are hop length and band count,
neither of which recovers F0 precision. **`INVARIANTS.md` §4 bans the encoder while adopting its lossy step.**

Two secondary errors in the same section: "P_f = 8 frequency bins (≈172 Hz)" multiplies 8 by the *FFT* bin
width, conflating linear FFT bins with mel bands; and mel bands are non-uniformly spaced, so no single Hz
figure describes 8 of them.

**Fix.** Add an explicit pitch and voice-quality channel — F0, jitter, shimmer, HNR, spectral tilt, CPP —
at ~10 ms hop, computed by an autocorrelation or neural pitch tracker (pYIN, CREPE, or Praat-equivalent).
These resolve F0 to ~1 Hz because they estimate *periodicity* rather than reading a bin. Add a CQT or
gammatone/ERB filterbank for harmonic structure, giving log-frequency resolution where the harmonics are.
Keep the mel branch for broadband texture.

**Note for `INVARIANTS.md`:** a pitch tracker is not ASR. Nothing in Invariant 4's rationale forbids it.
State this explicitly so the invariant does not block its own goal.

### F-04 — `d_model=4096` is incompatible with the named base model `[Blocking]`

*Refs: `README.md` §2, §3; `DESIGN.md` §4; `SPECS.md` §1, §4*

Qwen2.5-14B-Instruct has `hidden_size=5120`, 48 layers, 40 attention heads, 8 KV heads, and
`intermediate_size=13824`. A `(B, 64, 4096)` prefix cannot be concatenated with `(B, L, 5120)` token
embeddings. Every `4096` in the design — `d_model`, all four `4096²` projections, the `64 × 4096` latents,
the FFN width — is wrong.

Also: `README.md` offers "Qwen2.5-Omni / Qwen2.5-14B-Instruct" as interchangeable. They are materially
different systems; Omni ships its own audio encoder, which contradicts the custom-encoder thesis outright,
and it is not a 14B dense model. The 9.0 GB 4-bit budget describes only the 14B.

**Fix.** Pick one model, pin it as an explicit versioned configuration, and either select a model with a
verified 4096 width or make the connector emit 5120-wide vectors and re-derive every downstream number.

### F-05 — Raw-pixel differencing will mostly detect the camera `[High]`

*Refs: `DESIGN.md` §3.2; `SPECS.md` §1.2*

Temporal-L1 differencing on raw pixels responds to any change: handheld camera motion, auto-exposure and
auto-white-balance steps, mains flicker, rolling shutter, shadows, a sibling walking past, curtains. In a
family home with a phone camera these will dominate a 3–8 Hz wrist movement, and the threshold τ cannot
separate them because it discriminates on magnitude while camera shake *is* high-magnitude.

Three further problems:

- The hard indicator `1(E >= tau)` has zero gradient almost everywhere, so "a learned τ" is not learnable as written.
- Thresholding yields a **variable** number of surviving patches, while `SPECS.md` fixes `L_v = 1500`.
  That is 2.6% of the 58,800 candidate patches — exactly 5 patches per frame — which cannot hold across clips.
  A fixed count requires deterministic top-k plus padding and an attention mask, or a fixed pooled grid.
- Differencing discards static but relevant context: the object being requested, a communication partner, posture.

**Fix — hybrid, not a ban.** Stabilize frames. Make a pose/keypoint track (MediaPipe Holistic, BlazePose,
or an MMPose model) the primary kinematic representation — body-relative landmarks are inherently robust to
camera motion, and pose-based models reach 97–99% on the public stimming benchmark. Add optical flow (RAFT)
and retain a low-rate context/object channel. Keep temporal-difference-only as an ablation arm, not the
mandated path. Do not treat "stims" as a normative target to detect or reduce.

### F-06 — 60 fps is unnecessary and counterproductive `[Medium]`

*Refs: `README.md` §2; `SPECS.md` §1.1*

Nyquist for an 8 Hz stim is 16 fps; 30 fps gives comfortable margin. The published frame-rate ablation is
more pointed: accuracy **peaked at 15-frame sampling intervals** (LSTM 90.0% → 97.5%, GRU 92.5% → 98.75%),
with the authors attributing every-frame degradation to overfitting on pose noise, at ~93% less compute.

**Fix.** Capture at 30 fps / 720p and treat frame rate as a tuned hyperparameter, not a fixed capture spec.
This roughly quarters ingestion, storage, and encoder cost, buying back headroom for the pitch and
physiology channels.

### F-07 — Autonomic arousal is reasoned about but never measured `[High]`

*Refs: `DESIGN.md` §1.1*

The state model rests on sympathetic hyper-arousal and vagal tone. The sensor suite is a camera and a
microphone, so arousal is *inferred* from the same ambiguous surface behavior the system is trying to
disambiguate. That is circular.

Wrist electrodermal activity, heart-rate variability, and accelerometry give a direct, continuous, objective
arousal correlate, and the evidence in this exact population is strong (AUROC 0.84 person-dependent for
1-minute-ahead forecasting). A wearable also works when the camera does not: in the car, at school, in the dark.

Separately, the claim in `DESIGN.md` §1.1 that "baseline vagal tone is frequently suppressed" is overstated —
RSA findings in autism are heterogeneous — and should be hedged or removed.

**Fix.** Add a physiological channel (Empatica EmbracePlus or equivalent; Apple Watch HRV as a
lower-fidelity start) as a third encoder path, subject to the child's tolerance for wearing it. Of every
change in this document, this is the most likely to move real-world accuracy.

### F-08 — Nightly SGD is the wrong shape for this data `[High]`

*Refs: `INVARIANTS.md` §5; `DESIGN.md` §7.3*

`INVARIANTS.md` §5 fixes batches at 8 historical + 2 novel. At 2 novel clips per day that is roughly
**730 labeled examples per year** and about **one optimization step per night**. Meanwhile the triplet loss
operates on 4096-dimensional mean-pooled vectors — more dimensions than there will be examples for years.
Hard-negative mining, which the invariant requires, is known to collapse on small datasets.

The 80/20 replay idea is directionally right, but freezing an exact ratio as an inviolable invariant is
false precision. The continual-learning literature points at buffer diversity, low learning rates,
regularization, and distillation, with the useful mix depending on data distribution and metrics.

Automatic nightly deployment is the more serious problem: a single mistaken or over-confident label can
change caregiver-facing behavior with no validation gate.

**Fix.** Replace nightly SGD-on-a-batch with a nightly **full re-fit** of the small components over all
accumulated data — at this scale that takes seconds to minutes and makes catastrophic forgetting
structurally impossible, because there is no prior state to forget. Reduce the metric space to 64–128
dimensions. Replace triplet-with-hard-mining with supervised contrastive loss or prototypical networks,
both markedly more stable few-shot. Demote 80/20 and 75% to tuned defaults (§11). Gate deployment (§9).

### F-09 — Embedding drift invalidates retrieval `[High]`

*Refs: `SPECS.md` §3.1; `DESIGN.md` §7.3*

Historical vectors in `nd_communicative_intents` were produced by an earlier encoder. Once the encoder
changes, new queries live in a different space, and nearest-neighbour lookups degrade **silently** — no
error, just quietly wrong neighbours. The design compounds this by using one mutable Chroma collection as
training source, retrieval index, and de facto benchmark simultaneously.

**Fix.** Version every embedding with the encoder checkpoint that produced it. Either freeze the retrieval
encoder independently of the predictive head, or re-embed a versioned corpus snapshot whenever the encoder
changes. Keep an immutable raw/feature record (consent permitting) so re-embedding is always possible.
Separate mutable training data from the immutable evaluation set and from the retrieval index.

### F-10 — Mean-pooling discards the resampler's structure `[Medium]`

*Refs: `DESIGN.md` §7.2; `SPECS.md` §4*

The resampler's value is that its 64 latents can specialize — one attending to pitch contour, another to
hand trajectory. `mx.mean(z_sensory, axis=1)` averages that away before the loss sees it, and does so
without L2-normalizing first, so high-norm latents dominate the cosine metric arbitrarily.

**Fix.** Use a learned attention-pool or a dedicated CLS latent, and L2-normalize before computing cosine distance.

### F-11 — The system learns the caregiver, not the child `[Blocking]`

*Refs: `SPECS.md` §3.1; whole design*

Ground truth is `parent_tag`. The model is trained to reproduce a parent's reading of the behavior. Where
that reading is accurate the system amplifies real expertise; where it is wrong, the system will confidently
and fluently reproduce the error back to the parent, who will read it as independent confirmation. That
closed loop manufactures false certainty.

Two literatures bear directly on this:

- The emotion-inference critique establishes that facial, body, and vocal cues are context-dependent, and
  specifically warns against calling agreement with observers "accuracy" (Barrett et al., 2019). That is
  precisely the metric this design would otherwise report.
- Facilitated Communication, Rapid Prompting Method, and Spelling to Communicate all failed on
  **authorship**. ASHA, ASAT, and the National Autism Center hold that none has demonstrated independent
  authorship under blinded testing; a 2017 review of 108 RPM papers found not one message-passing validity
  test. Project N is not FC — the signal is a real sensor recording, not a facilitator's hand — but it is
  exposed to the identical epistemic failure, and the documents currently contain no validity test at all.

**Fix.** Build a message-passing-equivalent from the start. The falsifiable claim is not "we decoded intent"
but **"the offered support resolved the behavior."** Log a prospective prediction, log which support was
offered, log whether the behavior resolved within N minutes. Add a blinded arm where the model predicts on
clips whose tag is withheld, and a second-rater arm where a different familiar adult tags independently —
giving inter-rater reliability on the labels themselves. You cannot exceed it, and you currently do not
know what it is. Where the child can respond via AAC or gesture, that response outranks all adult labels.

### F-12 — Cross-validation will leak; no baselines are defined `[High]`

Two 5-second windows from the same episode are near-duplicates. Splitting at window or clip level leaks and
inflates reported accuracy by a wide, unquantified margin.

Two baselines are required before any sensory result means anything:

- **Metadata only** — time of day, time since last meal, time since school, plus the caregiver's own
  antecedent tag. For "hydration request" and "post-school fatigue" this will be strong. Any sensory model
  must beat it to justify a single line of encoder code.
- **Shuffled labels** — establishing the null under the exact split and class distribution in use.

**Fix.** Split by **day**, or by episode with a temporal buffer. Report per-class precision and recall, not
accuracy — the 8-class enum will be severely imbalanced. See §10 for the full protocol.

### F-13 — "In Pain" with no medical rule-out `[High]`

*Refs: `SPECS.md` §3.1; `DESIGN.md` §7.1*

Distress behavior in non-speaking autistic children is frequently **medical**: constipation, reflux, dental
pain, otitis, migraine, seizure activity, and later menarche. A system that answers "vestibular seeking"
while the child has appendicitis has caused harm, and it will answer confidently, because that is what LLMs do.

The ad-hoc enum is also unnecessary. **NCCPC-R** (Non-Communicating Children's Pain Checklist – Revised) is
a validated 27-item instrument across six subscales — vocal, emotional, facial, body language, protective,
physiological — built for this population, designed for use by parents without training, internal
consistency α ≈ 0.92, with established sensitivity to pain. Several items are directly observable in the
video and audio streams.

**Fix.** Separate `possible pain / acute distress` from ordinary intent labels; it must never be silently
scored as sensory overload. Adopt NCCPC-R as the distress schema in place of `unspecified_discomfort`. Add a
conservative escalation card co-designed with the child's own clinicians: when the distress score or its rate
of change crosses threshold, the output is "this pattern warrants medical review" — with no sensory
interpretation offered. Encode the cost asymmetry explicitly: a missed medical cause is not equivalent to a
missed drink request.

### F-14 — Fluent generation with no abstention or calibration `[High]`

*Refs: `INVARIANTS.md` §4.3; `SPECS.md` §4*

An LLM handed 64 uninterpretable vectors and a clinical system prompt will produce polished, specific,
confident narratives about a child's inner state, with or without signal. There is no calibration step, no
confidence threshold, and no "I do not recognize this" state anywhere in the design. Note that
`confidence_score` in the schema is the *parent's* confidence in their own tag, not the model's.

**Fix.** Make abstention a first-class output and gate generation on retrieval: if nearest-neighbour distance
to any labeled exemplar exceeds threshold, return "unrecognized pattern" and show the nearest matches without
narrating. Use calibrated discriminative probabilities, prediction sets, or a conservative reject option.
Display coverage and observed calibration on held-out data. Never expose a fabricated numeric confidence.
The correct default for an unfamiliar pattern is "I do not know; here is a way to ask or observe."

### F-15 — Group-level neuroscience presented as individual fact `[Medium]`

*Refs: `DESIGN.md` §1.1*

| Claim in `DESIGN.md` §1 | Status | Action |
| :-- | :-- | :-- |
| Absence of speech ≠ absence of language or intent | Sound | Retain. |
| Local hyper-connectivity + long-range hypo-connectivity | **Contested** | Long-range underconnectivity has reasonable support; the local overconnectivity half is not a consistent finding. Hedge. |
| Purkinje cell variation impairing predictive motor sequencing | Reasonable at group level | Reduced Purkinje counts are among the better-replicated post-mortem findings, but the inference to *this child's* cerebellar function from a video clip is unlicensed. Remove the causal step. |
| "Baseline vagal tone is frequently suppressed" | **Overstated** | RSA findings are heterogeneous, and nothing measures this. Remove or hedge; see `F-07`. |
| Aberrant sensory gating / failure to habituate | Reasonable | Retain with hedging. |
| Stimming as active self-regulation, not pathology | Sound | Retain — the best-argued paragraph in the document. |

A Level 3 support designation describes support needs, not a fixed neurobiological subtype. Severe
motor-speech impairment or childhood apraxia may co-occur but cannot be assumed from non-speaking status.

**Fix.** Rewrite as heterogeneous, non-diagnostic possibilities. Remove claims of attenuated proprioceptive
feedback, persistent vagal suppression, or specific circuit failure unless a treating clinician has
independently assessed them — in which case store their report as L3 context, not as model reasoning.

### F-16 — The clinical prior is one generation behind, and omits AAC `[Medium]`

*Refs: `DESIGN.md` §6.1; `INVARIANTS.md` §4*

The LoRA corpus is Ayres Sensory Integration, DIR/Floortime, and Functional Behavior Assessment. Of these,
FBA's functional-analysis logic — escape, attention, tangible, automatic reinforcement — is the empirically
strongest and maps directly onto the `parent_tag` enum. ASI and DIR are the weakest: ASI's evidence is
genuinely mixed, judged largely on individually defined functional goals, and reviews continue to describe
the debate as open. None of this establishes the *neurophysiological explanation* the design attributes to it.

Two more current frameworks explain why stimming occurs more falsifiably than sensory integration does:

- **HIPPEA** (Van de Cruys et al., 2014) — prediction errors assigned inflexibly high precision, with
  stereotyped behavior as a strategy for increasing environmental predictability. This reframes a stim as
  uncertainty reduction, which is testable and antecedent-linked.
- **Interoception** — covers the body-state axis that the sensory-integration literature handles indirectly.

The larger omission: **the documents never mention AAC.** Aided augmentative and alternative communication is
the standard of care for minimally speaking autistic children, with improved vocal output in 87% of
participants and no study reporting decreased speech. A system that decodes a child *for* caregivers while
never touching the channel that lets the child speak for himself has chosen the harder problem and the
weaker outcome.

**Fix.** Rebalance toward functional analysis, predictive-processing accounts, interoception, and the AAC
literature; retain ASI labelled as one lens among several. Store every source with its scope and evidence
quality. Never train the model to treat one framework as settled causal truth. Then add the AAC bridge (`P0-4`).

### F-17 — Bulk LoRA on undifferentiated literature is the wrong mechanism `[Medium]`

*Refs: `DESIGN.md` §6.1*

Training a LoRA on "clinical literature" permanently mixes contradictory, weak, outdated, and possibly
rights-restricted material into weights, with no way to show why a specific statement was made. Retrieval is
strictly better for facts: inspectable, updatable, removable, and source-linked.

**Fix.** Split the two roles cleanly.

- **Retrieval carries facts.** Each source in the local library records author, date, licence/rights status,
  population studied, evidence level, and scope limitations. Synthesis must link to the precise excerpts used.
- **LoRA carries behavior only** — parent-respectful language, evidence-first answer structure, abstention
  discipline, source citation, and a hard prohibition on inventing causal or clinical claims. It is not the
  repository of clinical facts.

### F-18 — Offline is necessary but not sufficient `[Medium]`

Invariant 1 solves data exfiltration. It does not address consent, third parties, retention, key management,
or the regulatory weight of the output's phrasing. See §8 in full.

### F-19 — Invariants freeze untested hyperparameters `[Medium]`

`INVARIANTS.md` presents 80/20, 75% masking, 64 latents, margin 0.25, `r=64`, and 36.0 GB as inviolable
rules that agents may not modify. These are hyperparameters. Freezing them in a document that binds every
contributor prevents exactly the empirical search the project needs. See §11 for the proposed split.

### F-20 — Two documented API idioms do not exist `[Low]`

*Refs: `AGENTS.md` §3.4, §4.3*

- `param.trainable = False` is not MLX. Freezing goes through `nn.Module.freeze()` / `unfreeze()` and
  `trainable_parameters()`. The documented idiom **silently does nothing**, which is the worst available
  failure mode for Invariant 3.
- `mx.metal.get_active_memory()` has moved to top-level `mx.get_active_memory()` / `mx.get_peak_memory()`
  in current MLX. Pin the MLX version in the docs.
- Minor: `DESIGN.md` §4.2 specifies truncated-normal latent init while the `SPECS.md` code uses plain
  `mx.random.normal`; and the Hann window is written with the symmetric `N-1` denominator where STFT wants
  the periodic form.

---

## 4. Numeric corrections

Recomputed from the formulas in the specification against the published model configuration.

| Quantity | Documented | Recomputed | Cause |
| :-- | --: | --: | :-- |
| LLM hidden size | 4096 | **5120** `[confirmed x2]` | Qwen2.5-14B `hidden_size=5120`, 48 layers, 40 heads, 8 KV heads, FFN 13824. Every 4096 is wrong. |
| AaPE audio tokens `L_a` | 2048 | **5766** `[confirmed x2]` | From the document's own patch/stride: `floor((1500-16)/8)+1 = 186` time × `floor((128-8)/4)+1 = 31` freq. Total sequence becomes 7,266, not 3,548 — so the 55.4× compression figure also follows the wrong input. |
| STFT frame count | 1500 | **1488** uncentered / **1501** centered | `1 + floor((240000-2048)/160) = 1488`. The centering/padding convention must be named; 1500 is not self-justifying. |
| STFT bin spacing | 21.5 Hz | **23.4375 Hz** at 48 kHz | 21.5 Hz is the 44.1 kHz value. |
| STFT hop duration | 3.63 ms | **3.333 ms** at 48 kHz | 3.63 ms is the 44.1 kHz value. |
| Mel ceiling | 22,050 Hz | **24,000 Hz** at 48 kHz | 22,050 Hz is the 44.1 kHz Nyquist. |
| Mel band width near F0 | "8 bins ≈ 172 Hz" | **~27 Hz per band** at 300 Hz | 172 Hz = 8 × *FFT* bin width. Conflates linear FFT bins with mel bands. See `F-03`. |
| Video frame differences | 300 | **299** | 300 frames yield 299 consecutive differences. |
| Kinematic tokens `L_v` | 1500 fixed | **variable** | 1500 is 2.6% of 58,800 candidates = 5 patches/frame. Thresholding cannot produce a fixed count. |
| NDProjector parameters | ~68 M | **~209 M** `[confirmed x2]` | Aligns 7.3 M + QKVO 67.1 M + FFN (4096×16384×2) 134.2 M + latents 0.26 M. The FFN alone exceeds the stated total. |
| LoRA parameters (r=64, 7 modules, 48 layers) | ~85 M | **~275 M** `[confirmed x2]` | 5.73 M/layer × 48 at the real hidden size. |
| Total trainable | ≤160 M | **~484 M** | `INVARIANTS.md` §3.3 is violated by its own configuration. AdamW states ≈ 3.9 GB. |
| KV cache @ 8k tokens | 10.0 GB | **~1.6 GB** (1.5 GiB) `[confirmed x2]` | GQA with 8 KV heads: 192 KB/token × 8192. The 10 GB figure matches a *multi-head* assumption (40 heads → 8.05 GB), i.e. GQA was overlooked. |
| KV cache scaling | "quadratic in sequence length" | **linear** | Cache *storage* is linear in cached tokens. Prefill *compute* is where the quadratic dependence sits. |
| Activation memory for backprop | not budgeted | — | If any loss flows through the LLM, storing activations across 48 layers is a first-order cost absent from every table. |
| Clip length | 30–120 s | 5 s | `README.md` §2 captures 30–120 s; every tensor in `SPECS.md` is a 5 s window. No aggregation rule is given for the 6–24 windows per clip. |
| Base model identity | "Omni / 14B" | pick one | Materially different systems. See `F-04`. |

**Pattern, not a list of typos.** The consistent direction is round numbers — 2048, 4096, 64, 80/20,
36.0 GB, 75% — presented as derived quantities. The specification has the *form* of a rigorous design
document without the derivations underneath.

**Action:** before writing model code, rebuild every table in `SPECS.md` §1–2 from a single script that
computes shapes and byte counts from first principles and from the instantiated model, and make that script
the source of truth the tests assert against. Replace theoretical memory rows with **measured** benchmark
envelopes per exact configuration, including batch, capture window, prefill, generation length, cache type,
and concurrent server load.

---

## 5. Target architecture

Invert the current design: **retrieval first, generation last.** The LLM leaves the inference path.

```
                     ┌─────────────────────────────────────────────┐
   clip (30 fps,     │  L1  MEASUREMENT                            │
   48 kHz audio,     │  ├─ acoustic: log-mel + CQT + F0/jitter/    │
   wearable stream)  │  │            shimmer/HNR  (~10 ms hop)     │
        │            │  ├─ kinematic: pose/keypoints + optical      │
        ├───────────►│  │            flow + low-rate context       │
        │            │  ├─ physiology: EDA / HRV / accelerometry    │
        │            │  └─ capture quality gate                    │
        │            └───────────────────┬─────────────────────────┘
        │                                │
        │            ┌───────────────────▼─────────────────────────┐
        │            │  encoders (pretrained, fine-tuned)          │
        │            │  BEATs / AudioMAE · VideoMAE / pose model   │
        │            │  resampler trained on audio-visual          │
        │            │  temporal correspondence   (see F-02)       │
        │            └───────────────────┬─────────────────────────┘
        │                                │  64-128 dim embedding
        │            ┌───────────────────▼─────────────────────────┐
        │            │  L2  RETRIEVAL + CALIBRATED HEAD            │
        │            │  ├─ prototype / k-NN over confirmed episodes│
        │            │  ├─ calibrated multi-label probabilities    │
        │            │  └─ ABSTAIN if d(nearest) > threshold       │
        │            └───────────────────┬─────────────────────────┘
        │                                │  structured result object
   ┌────▼─────────────┐  ┌───────────────▼─────────────────────────┐
   │ L3  caregiver     │  │  L4  evidence library (versioned RAG)  │
   │ context + outcome │  │      cited excerpts, scope, quality    │
   └────┬─────────────┘  └───────────────┬─────────────────────────┘
        │                                │
        └────────────────┬───────────────┘
                         │
              ┌──────────▼──────────────────────────────────────────┐
              │  RENDERING ONLY — schema-constrained LLM            │
              │  May phrase L1-L4 into plain language.              │
              │  May NOT add a label, cause, diagnosis, or          │
              │  intervention absent from the structured result.    │
              └─────────────────────────────────────────────────────┘
```

Why this shape:

- It fits ~730 labels/year instead of needing hundreds of thousands of paired examples, closing `F-01` and `F-02`.
- It is inspectable, which closes `F-14` and satisfies the "review the basis" property discussed in §8.
- It runs on a phone rather than a 48 GB Mac, making the hardware invariants largely moot.
- It matches the published person-specific results (§2.2) rather than an architecture with no precedent at this data scale.

**Encoder policy.** Rewrite Invariant 4 to ban the *phonemic decoding step and text-only intermediates*,
not pretrained acoustic encoders. Layer-wise probing shows Whisper's encoder retains substantial
paralinguistic content, with middle layers (~6–12) encoding prosody, rhythm, vocal effort, and pitch
contour; published comparisons of wav2vec 2.0 against Whisper for non-verbal vocalization classification
found both usable. The phonemic bias lives in the decoder — the text head, the CTC loss, the k-means targets.

The cost of the current ban is severe: training an AST and a 12-layer ViT from scratch on **50 hours of one
child**, where AudioSet-scale SSL sees ~5,800 hours / 2M clips and ImageNet-scale MAE sees 1.3M distinct
images. 10.8M frames drawn from a handful of rooms and one body are enormously redundant and tiny in
effective sample size. Make raw, pretrained, and hybrid encoders **competing experimental arms**.

---

## 6. Data schema

The current ontology confounds four different things: observable form, contextual antecedent, hypothesized
function, and response. Replace the single `parent_tag` with separate record groups.

| Field group | Examples | Rule |
| :-- | :-- | :-- |
| **Observation** | vocal duration, F0 statistics, motion rhythmicity, object/partner present, time window, capture quality | Measured, or explicitly caregiver-observed. No inferred state. |
| **Context** | meal timing, transition, ambient noise, sleep/illness report, location, medication change | Allow unknown and free text. Record observer and source. |
| **Hypothesis** | "may want a drink", "may be overwhelmed", "unclear" | Multiple labels allowed. Carries caregiver confidence and rationale. **Never ground truth by default.** |
| **Action and outcome** | AAC offered, water offered, break offered, child acceptance/refusal, resolution latency | **The primary learning signal.** Supports causal humility. |
| **Child response** | AAC selection, gesture, reach, rejection | Where available, this outranks every adult label. |
| **Distress** | NCCPC-R subscale scores | Separate from intent labels. Never silently mapped to a sensory explanation. See `F-13`. |
| **Provenance** | model / data / schema / encoder version, consent status, retention date, annotator role | Required for reproducibility, audit, withdrawal, and rollback. See `F-09`. |

Every annotation screen must offer: `None of these` · `Not enough information` · `Other` ·
`Child confirmed via AAC/gesture` · `Do not learn from this clip`.

---

## 7. Output contract

Never let the LLM write an unconstrained "translation." Return a structured, non-diagnostic object, rendered
with its layers visibly distinct:

```
Observed (L1)      Repeated right-hand movement, 3.8 Hz; sustained vocalization
                   4.2 s, F0 mean 268 Hz, low variance. Capture quality: adequate.

Similar (L2)       3 prior caregiver-confirmed episodes — 12 Aug, 27 Aug, 2 Sep.
                   2 of 3 resolved after deep pressure was offered.

Context (L3)       Tagged post-school. 40 min since last drink.

Possibilities      Offer AAC choice board · offer a quiet break · offer water.
                   Insufficient evidence to rank a single explanation.

Confidence         Coverage 0.62 at this threshold. Calibration error 0.08 on
                   held-out days. Not a ranked diagnosis.

Confirm            Record child's response, or select "none of these."

Safety             If agreed red flags are present, follow the existing care plan.
```

Language to remove from all documents: "child state diagnosis", "translate ... into intent",
"actionable caregiver support protocols", "clinical reasoning core". Replace with observational,
uncertainty-bearing wording.

---

## 8. Security, privacy, and governance

### 8.1 Always-on encrypted vault

The intended flow — Mac screen-locked but logged in, receives a clip from the paired Flutter app, encrypts
and stores it, begins processing with no parent at the keyboard — is valid, and it determines how key access
must work.

Use a signed **per-user app helper or LaunchAgent**, not a system-wide or root daemon. Apple's TN3137
documents that the modern Data Protection Keychain is available only to processes running in a user login
context and **cannot be used from a launchd daemon**; it recommends `SecItem` with
`kSecUseDataProtectionKeychain=true`. A locked screen is not a logged-out user, so a user-context helper
stays available while the Mac is awake.

```
Flutter phone (paired once) --mTLS + signed request--> Mac per-user helper
                                                        │
                                                        ├─► per-clip random AES-256-GCM key (DEK)
                                                        │      └─► encrypted media / features / metadata
                                                        │
                                                        └─► vault key in Data Protection Keychain
                                                               └─► wraps each clip DEK
```

- Store encrypted media, embeddings, and metadata **outside** the Keychain; store only compact keys inside it.
- Use AES-256-GCM, a fresh nonce per encryption, and authenticated associated data containing clip ID,
  schema version, and ciphertext version.
- Make the vault key **device-only and non-synchronizing**. Do not set `kSecAttrSynchronizable`.
- Give the *processing* key a background-capable access policy (for example "after first unlock") so the
  helper can work while the screen is locked. Make this a consciously selected setting, visible to the parent.
- Require Touch ID or password for a **separate private-review/export key**: opening the timeline, exporting
  raw media, changing retention, pairing a device, or revealing sensitive notes.
- One DEK per clip or short bundle. Deleting its wrapped DEK gives cryptographic erasure; also delete the
  ciphertext and log the event. Any encrypted backup needs an explicit recovery design.

**State the residual risk honestly.** If the system can decrypt and analyze clips while the screen is locked,
the helper must be able to obtain the processing key without Touch ID at that moment. This protects against a
powered-off or stolen disk (with FileVault on) and against ordinary unauthorized app access. It does **not**
protect against malware already running as the same logged-in user. Mitigate with FileVault, a non-admin
account for daily use, hardened runtime and code signing, a minimal helper interface, local firewall rules,
no inbound WAN exposure, and prompt updates.

Do not describe this mode as "Touch-ID-protected processing while locked." It is **automatic encrypted
processing while locked**.

### 8.2 Transport

Pair each phone on the Mac while a parent is present, then issue a device-specific client credential.
Require mutual TLS, a per-request signature and nonce to prevent replay, rate and size limits, and a local
allowlist. **Never treat mDNS discovery or a LAN IP address as authentication.** The Flutter app must store
its credential in the iOS Keychain or Android Keystore, not in application preferences.

### 8.3 Consent, assent, and retention

- Continuous household recording captures siblings, visitors, and support staff who have not consented.
  Audio recording of third parties triggers all-party-consent statutes in several US states and GDPR
  obligations in the EU.
- The child cannot consent. Implement an explicit **dissent protocol** that respects behavioral refusal: if
  he covers or turns from the camera, recording stops. Engage the disability-rights and autistic
  self-advocacy critique of surveilling autistic children in `docs/`, rather than leaving it implicit.
- Enforce camera-free zones (bathroom, bedroom) in the capture app, not by policy.
- Automatic deletion of untagged footage after a fixed window, rather than indefinite retention.
- Face-blurring of non-subjects.
- Keep only synthetic or fixture media in source control.
- **De-identify the committed documents.** `DESIGN.md` and `INVARIANTS.md` name a real minor child and
  describe his clinical presentation, in a repository whose `README.md` points at a public GitHub URL. Even
  with media gitignored, the prose is identifiable pediatric health information. Use a pseudonym and keep
  the linkage offline.

### 8.4 Regulatory posture

A private, unmarketed, family-built tool is a different analysis from a marketed clinical product — but the
**claims** move the analysis. "Child state diagnosis," clinical reasoning, and treatment protocols are the
phrases that change it. FDA's current CDS guidance addresses software intended for patients and caregivers as
well as clinicians, and the non-device CDS exclusion under §520(o)(1)(E) is written for software intended for
health care professionals. Separately, the guidance's "independently review the basis of the recommendation"
criterion is one an opaque 64-token prefix cannot satisfy — which points the same direction as §5 and §7:
**show the evidence, not the conclusion.**

HIPAA applies when the operator is a covered entity or business associate, not automatically to a
family-owned tool; other US state, education, and consumer-privacy rules, and non-US rules, may apply.
Make no compliance claims without jurisdiction-specific counsel. Obtain specialist regulatory and privacy
advice before clinical deployment, before sharing beyond the family, and before any marketing. Remove
"diagnosis" from every document now.

---

## 9. Model lifecycle

No autonomous nightly deployment. Replace it with a gated promotion pipeline.

1. Keep a **frozen production model**. It changes only by explicit promotion.
2. Train a **candidate** only after a minimum number of independently confirmed, sufficiently diverse examples.
3. Evaluate against **time-separated and context-stratified holdouts** plus a locked "never train" safety set.
4. Require predeclared improvement in calibration, coverage, and per-class error, **with no safety regression**.
5. Caregiver review — and clinician review where the child's care team is involved — then promote or discard.
6. Retain rollback, data and model lineage, and a change log.

Version all embeddings; re-embed a versioned corpus snapshot before changing retrieval, or freeze the
retrieval encoder independently (`F-09`). Never use a mutable Chroma collection as training source,
retrieval index, and benchmark simultaneously.

---

## 10. Evaluation protocol

Write this before the first encoder. Preregister it in `docs/` so results cannot be selected after the fact.

**Design.** Preregistered single-case / N-of-1 study with the child, caregivers, and — where appropriate —
AAC/SLP, OT, and medical partners, plus an ethics and privacy review.

**Splits.** Time-separated and context-separated. Leave-one-day-out, or leave-one-episode-out with a temporal
buffer. **Never** split adjacent clips from the same event across train and test. Collect independent,
delayed confirmation where feasible so the same caregiver's expectation is not both label and outcome.

**Predeclare:**

- [ ] Target use, exclusion cases, red-flag policy, and explicit non-uses.
- [ ] Label definitions and an adjudication rule for disagreement.
- [ ] **Primary benefit:** increased successful child-confirmed communication, or measured caregiver
      usefulness — *not* inferred-emotion accuracy, and not reduction of stimming.
- [ ] Precision and recall **per possibility label**, stratified by context, capture condition, and time.
- [ ] Calibration error, abstention rate, prediction-set coverage, and drift.
- [ ] Critical errors: missed pain or urgent-care escalation, false reassurance, harmful or culturally
      invalid framing.
- [ ] Inter-rater reliability on the labels themselves — you cannot exceed it.
- [ ] Baselines, all reported alongside every model number:
      caregiver notes alone · nearest prior episode · **structured context-only model** ·
      shuffled labels · raw-features model · pretrained/hybrid model · no-LLM output.
- [ ] A stopping rule and a rollback rule.

**Abstention conditions.** Poor video or audio quality, camera motion, a new setting, illness, new
medication, and out-of-distribution patterns all trigger abstention rather than a guess.

---

## 11. Invariants: true invariants vs tunable defaults

Split `INVARIANTS.md` in two. A component becomes an invariant only after it earns that status empirically
for this child, this task, and this deployment.

### 11.1 True invariants — keep and enforce

- **Privacy and offline execution.** No cloud AI APIs, no outbound WAN traffic, no telemetry.
- **Encryption and key hierarchy** as specified in §8.1, including non-synchronizing device-only vault key.
- **Base LLM weights frozen.** No de-quantization of `W0` for fine-tuning.
- **No autonomous model deployment.** Promotion requires the §9 gate.
- **Abstention is always available**, and is the default for unfamiliar patterns.
- **No diagnosis, and no unverified inference of pain, autonomic state, or intent presented as fact.**
- **Red-flag escalation** takes precedence over any sensory interpretation.
- **Every output traceable to L1–L4**, with layers visibly distinct.
- **Provenance recorded** for every stored record: model, data, schema, encoder, consent, retention.
- **Immutable evaluation and safety sets**, separate from training data and the retrieval index.
- **A measured memory ceiling** — measured on the real configuration, not asserted.

### 11.2 Tunable defaults — move to a config document

Replay ratio (currently 80/20) · masking ratio (75%) · latent count (64) · LoRA rank and alpha
(r=64, α=128) · triplet/contrastive margin (0.25) · batch size (10) · sequence lengths (`L_a`, `L_v`) ·
capture frame rate and resolution · mel band count · hop length · nearest-neighbour abstention threshold ·
context window · memory ceiling value.

Each of these needs an ablation and a recorded result, not a rule.

---

## 12. Action ladder

### P0 — before any caregiver-facing use

- [ ] **P0-1 Invert the architecture** to retrieval-first, generation-last (§5). LLM leaves the inference
      path; it renders a structured result under a schema and may not add labels, causes, or interventions.
      *Closes `F-01`, `F-02`, `F-14`.*
- [ ] **P0-2 Write and preregister the evaluation protocol** (§10) before the first encoder.
      *Closes `F-11`, `F-12`.*
- [ ] **P0-3 Fix the acoustic front end and the invariant that caused it.** Add F0/jitter/shimmer/HNR at
      ~10 ms hop plus CQT or ERB. Rewrite Invariant 4 to ban phonemic decoding and text-only intermediates,
      not pretrained encoders, and state that pitch tracking is permitted. *Closes `F-03`; amends `F-16`.*
- [ ] **P0-4 Add the AAC bridge and the medical rule-out.** Route predictions into a speech-generating device
      or AAC board as **pre-populated candidate requests the child selects from**, rather than conclusions
      delivered to adults — making him the author. Adopt NCCPC-R for distress, with threshold crossings
      escalating to "seek medical review." *Closes `F-13`; addresses `F-11`, `F-16`.*
- [ ] **P0-5 Correct the model interface.** Pin one base model; fix `d_model` to 5120 or select a verified
      4096-wide model and re-derive every dependent number. *Closes `F-04`.*
- [ ] **P0-6 Remove deterministic clinical language** from `DESIGN.md` and `README.md`: neurobiology as
      individual fact, F0-to-affect mappings, stimming-to-intent mappings, "diagnosis", "translation",
      "protocols". *Closes `F-15`; addresses `F-18`.*
- [ ] **P0-7 Implement the two-key privacy posture** (§8.1): automatic encrypted processing while locked,
      Touch-ID-gated private review and export, with the residual-risk tradeoff documented in plain words.
      *Closes part of `F-18`.*
- [ ] **P0-8 Add safety machinery:** abstention, confirmation, red-flag policy, human oversight, model
      promotion gate, rollback, incident path. *Closes `F-14`; supports `F-08`.*

### P1 — before training

- [ ] **P1-1 Add the physiological channel** (EDA / HRV / accelerometry). Highest expected accuracy gain of
      any change here. *Closes `F-07`.*
- [ ] **P1-2 Replace nightly SGD with nightly re-fit**; 64–128-dim metric space, attention-pooled and
      L2-normalized, supervised-contrastive or prototypical loss. *Closes `F-08`, `F-10`.*
- [ ] **P1-3 Make pose the primary kinematic path**, drop to 30 fps / 720p, soft-gate the motion threshold,
      let sequence length vary with a mask. Keep temporal-difference as an ablation arm. *Closes `F-05`, `F-06`.*
- [ ] **P1-4 Version embeddings and corpus snapshots**; separate mutable training data from the immutable
      evaluation set and the retrieval index. *Closes `F-09`.*
- [ ] **P1-5 Regenerate every number in `SPECS.md` from one script** driven by the instantiated model, with
      executable shape tests. Resolve the 44.1/48 kHz split, name the STFT centering convention, reconcile
      the 5 s window against 30–120 s clips. *Closes §4; unblocks `AGENTS.md` §4.2.*
- [ ] **P1-6 Replace theoretical memory rows with measured envelopes** per exact configuration — batch,
      capture window, prefill, generation length, cache type, concurrent load. Count trainable parameters
      from the instantiated model. *Closes the remainder of §4.*
- [ ] **P1-7 Fix the MLX idioms** in `AGENTS.md`: `Module.freeze()`, top-level `mx.get_active_memory()`.
      Pin the MLX version. *Closes `F-20`.*

### P2 — scientific quality

- [ ] **P2-1 Restructure the evidence library.** Per-source author, date, rights, evidence level, population,
      and scope limitations; synthesis links to exact excerpts. Stop bulk-training a LoRA on undifferentiated
      literature; reserve LoRA for behavioral and formatting objectives only. *Closes `F-17`.*
- [ ] **P2-2 Rebalance the clinical corpus** toward functional analysis, predictive-processing accounts,
      interoception, and AAC; hedge the connectivity and vagal-tone claims. *Closes `F-16`.*
- [ ] **P2-3 Split `INVARIANTS.md`** into true invariants and tunable defaults (§11). *Closes `F-19`.*
- [ ] **P2-4 Add a source-backed claims table** to the project docs; label every unsupported design
      hypothesis as a hypothesis.
- [ ] **P2-5 Add an ablation protocol and a results registry.** Raw vs pretrained vs hybrid encoders;
      16/32/64/128 latents; replay composition; masking ratio.
- [ ] **P2-6 Governance and de-identification pass** (§8.3): pseudonymize the child, write the data-governance
      note, enforce camera-free zones and dissent handling in the app.

---

## 13. Bottom line

The humane premise is strong, and the personalization bet is validated by published person-specific results.
The current design solves the wrong bottleneck: it spends its complexity budget on a ~209M-parameter projector
into a frozen 14B LLM that no loss function ever teaches to read it, while the actual bottleneck is ~730
labels a year and no definition of success.

Trade the LLM prefix for a calibrated retrieval system. Spend the freed effort on measurement — pitch,
physiology, pose — and on a real evaluation protocol. Route the output through AAC so the child is the author
rather than the subject. The system should become **less certain and more useful**: observe carefully, combine
the clip with personal history and a curated evidence library, show the basis for each possibility, enable the
child to confirm or reject it, say plainly when the evidence is inadequate, and validate every learned
association prospectively.

---

## 14. References

**Population and task**

- Johnson, Narain, Quatieri, Maes, Picard. *ReCANVo: A database of real-world communicative and affective nonverbal vocalizations.* Scientific Data 10:523 (2023). https://www.nature.com/articles/s41597-023-02405-7
- Narain et al. *Modeling Real-World Affective and Communicative Nonverbal Vocalizations From Minimally Speaking Individuals.* IEEE Trans. Affective Computing (2022). https://ieeexplore.ieee.org/document/9898895/
- Goodwin et al. *Wearable Biosensing to Predict Imminent Aggressive Behavior in Psychiatric Inpatient Youths With Autism.* JAMA Network Open (2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10739066/
- NIDCD. *Workshop on minimally verbal / non-speaking autistic people* (2023). https://www.nidcd.nih.gov/workshops/2023/summary

**Epistemics and validity**

- Barrett, Adolphs, Marsella, Martinez, Pollak. *Emotional Expressions Reconsidered.* Psychological Science in the Public Interest (2019). https://journals.sagepub.com/doi/10.1177/1529100619832930
- National Autism Center. *Position statement on S2C, RPM, and FC.* https://nationalautismcenter.org/national-autism-center-releases-position-statement-on-spelling-to-communicate-rapid-prompting-method-and-facilitated-communication/
- *Safeguarding the communication rights of minimally- or non-speaking people* (2025). https://www.tandfonline.com/doi/full/10.1080/23297018.2025.2544116
- Ghassemi et al. *Practical guidance on artificial intelligence for health-care data / uncertainty and abstention.* npj Digital Medicine (2020). https://www.nature.com/articles/s41746-020-00367-3
- NIST. *AI Risk Management Framework Core.* https://airc.nist.gov/airmf-resources/airmf/5-sec-core/

**Clinical frameworks**

- Fusaroli et al. *Is voice a marker for Autism spectrum disorder? A systematic review and meta-analysis.* Autism Research 10(3):384–407 (2017). d = 0.4–0.5, ~61–64% discriminatory accuracy. https://onlinelibrary.wiley.com/doi/10.1002/aur.1678
- Van de Cruys et al. *Precise Minds in Uncertain Worlds: Predictive Coding in Autism.* Psychological Review (2014). https://sandervandecruys.be/pdf/2014-VandeCruysetal-PsychRev-Precise_minds.pdf
- Schaaf et al. (2018) https://pubmed.ncbi.nlm.nih.gov/29280711/ and Acuña et al. (2025) https://pubmed.ncbi.nlm.nih.gov/40193295/ — Ayres Sensory Integration systematic reviews.
- Schoen et al. *A systematic review of Ayres Sensory Integration intervention for children with autism.* Autism Research (2019). https://onlinelibrary.wiley.com/doi/full/10.1002/aur.2046
- Vasa et al. *The disrupted connectivity hypothesis of autism.* (2017). https://pubmed.ncbi.nlm.nih.gov/28083565/
- *Is functional brain connectivity atypical in autism? A systematic review of EEG and MEG studies.* PLOS One (2017). https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0175870
- Breau et al. *Non-Communicating Children's Pain Checklist – Revised (NCCPC-R).* https://www.community-networks.ca/wp-content/uploads/2015/07/PainChklst_BreauNCCPC-R2004.pdf
- *The Effect of NDBI and Aided AAC on Language Development of Children on the Autism Spectrum with Minimal Speech.* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12208088/
- ASHA. *AAC in early intervention.* https://www.asha.org/practice/early-intervention-provider-support/augmentative-and-alternative-communication-in-early-intervention/

**Models and systems**

- Qwen2.5-14B-Instruct configuration. https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/main/config.json
- Alayrac et al. *Flamingo: a Visual Language Model for Few-Shot Learning.* https://arxiv.org/abs/2204.14198
- Liu et al. *Visual Instruction Tuning (LLaVA).* https://arxiv.org/abs/2304.08485
- Chen et al. *BEATs: Audio Pre-Training with Acoustic Tokenizers.* https://arxiv.org/abs/2212.09058
- Rajagopalan et al. *Detecting self-stimulatory behaviours for autism diagnosis* (SSBD). https://ieeexplore.ieee.org/document/7025294/
- *Evaluating the Effect of Frame Rate in Sequence-Based Classification of Autism-Related Self-Stimulatory Hand Idiosyncrasies.* https://arxiv.org/html/2607.07957v1
- MLX neural-network API. https://ml-explore.github.io/mlx/build/html/python/nn.html
- MLX-LM (cache and prefill control). https://github.com/ml-explore/mlx-lm

**Platform and regulatory**

- Apple. *TN3137: On Mac keychain APIs and implementations.* https://developer.apple.com/documentation/Technotes/tn3137-on-mac-keychains
- Apple. *Restricting keychain item accessibility.* https://developer.apple.com/documentation/security/restricting-keychain-item-accessibility
- FDA. *Clinical Decision Support Software* guidance. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- HHS. *HIPAA Security Rule.* https://www.hhs.gov/hipaa/for-professionals/security/laws-regulations/index.html
