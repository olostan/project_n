# Project N scientific, clinical, and systems review

> [!NOTE]
> **Historical Audit Provenance**  
> This document preserves the original architectural, clinical, and systems critique delivered on 2026-09-08 that motivated the architectural overhaul of Project N. The actionable recommendations from this audit were incorporated into [`REVIEW_REFINEMENTS.md`](REVIEW_REFINEMENTS.md), and current normative technical specifications reside in [`SPECS.md`](SPECS.md), [`DESIGN.md`](DESIGN.md), and [`INVARIANTS.md`](INVARIANTS.md).

**Date:** 2026-09-08  
**Scope:** All Markdown documents present in the repository at review start (`AGENTS.md`, `DESIGN.md`, `INVARIANTS.md`, `README.md`, and `SPECS.md`), plus the clarified intended use: a private, parent-owned system for understanding one child better, running locally on an always-on Mac and accepting clips from a paired Flutter app. This is a research and architecture review, not an assessment of any child and not medical, legal, or regulatory advice.

## Executive answer

Project N starts from several good commitments: respect for non-speaking communication, individualized rather than population-level modeling, local-first handling of exceptionally sensitive data, multimodal sensing, and caregiver participation. These are worth preserving. NIDCD specifically identifies participatory research, heterogeneity, personalized designs, AAC, and carefully developed AI as priorities for minimally verbal/non-speaking autistic people [NIDCD workshop](https://www.nidcd.nih.gov/workshops/2023/summary).

The private, personal setting makes the project more appropriate and more feasible than a product for external clinicians. It does **not**, however, make deterministic claims about an individual child's internal state scientifically valid. The current design is not sound as a literal “intent translator” or “clinical reasoning” engine: it makes specific, often deterministic claims from behavior and sound without sufficient evidence; it hard-codes unvalidated ML choices as invariants; and it has blocking mathematical and model-interface inconsistencies.

The appropriate near-term product is a **private, caregiver- and child-controlled insight and communication-support tool**. It should aim to provide as much useful insight as the evidence can support: recurring signal patterns, comparable caregiver-verified episodes, relevant personal context, source-linked research perspectives, calibrated possible explanations, and AAC/observation-based ways to confirm them. It must not present a diagnosis or an unverified inference of pain, autonomic state, or intent as fact, and it must not silently alter its deployed model. Only after prospective individual validation meets predeclared utility and safety criteria should capabilities expand.

### Agreed product contract

Project N is a parent-owned private assistant, not a clinician-facing service. Its core question is: **“Given this clip, this child's history, and a curated local evidence library, what is worth noticing, what has helped in similar situations, what are reasonable possibilities, and how can the parent and child confirm or reject them?”**

This is intentionally broader than a narrow classifier. The system may synthesize a rich explanation, but every insight must be traceable to one of four layers: (1) measured features in the current clip, (2) comparable episodes in the child's own history, (3) explicit caregiver context and outcomes, or (4) a cited, versioned source from the evidence library. The interface must distinguish those layers rather than letting fluent language blur them together.

## What is well founded, and what is only a hypothesis

| Project premise | Assessment | What to retain or change |
|---|---|---|
| Non-speaking does not mean no language, cognition, agency, or communicative intent. | Sound ethical and clinical orientation. Communication ability is heterogeneous; AAC and assistive technologies are established routes to communication. | Retain; make the child an active design participant where possible, and treat AAC as the primary communication channel, not a backup. |
| Vocalizations, movement, sensory context, and antecedents can contain useful individual information. | Plausible and testable. It does not establish that any feature has a stable one-to-one meaning. | Retain as an N-of-1 measurement hypothesis with prospective validation. |
| Stimming can be regulatory and sometimes social/communicative. | Supported in broad terms, but functions are plural, context-dependent, and person-specific. | Replace every “this means X” assertion with “this pattern may be associated with X for this child, pending confirmation.” |
| Standard ASR and image classifiers may be poor feature extractors for this task. | Plausible, but the claimed total failure is unproved. Modern encoders can retain useful non-lexical and temporal information. | Make raw, pretrained, and hybrid encoders competing experimental arms; do not ban them by ideology. |
| A Perceiver-style resampler can reduce a long sensory sequence. | Sound architecture family. Flamingo shows a resampler can compress visual features to 64 tokens, but in a model trained for vision-language alignment. | Keep as a candidate, not a fixed 64-token invariant; ablate 16/32/64/128 tokens and train an aligned connector. |
| Self-supervision and replay can help in low-label personalized learning. | Reasonable research direction, not a demonstrated solution here. | Use frozen snapshots, held-out evaluation, and gated deployment; remove the fixed 80/20 claim. |
| Offline/local processing is privacy-positive. | Strong design choice, but “local” alone is not a complete security program. | Keep, while adding encryption, key management, access control, retention/deletion, audit, and LAN security. |

## Clinical and neurobiological assessment

### 1. Do not convert group-level neuroscience into an individual causal story

`DESIGN.md` describes atypical pruning, a local-hyper/long-range-hypoconnectivity profile, Purkinje-cell changes, and suppressed vagal tone as the neurological landscape of a Level 3 non-speaking child. These are **research hypotheses and group-level associations**, not diagnostic facts about a particular child. Connectivity research itself emphasizes inconsistent findings and clinical/age heterogeneity [Vasa et al., *Disrupted Connectivity Hypothesis*](https://pubmed.ncbi.nlm.nih.gov/28083565/). Cerebellar involvement is biologically plausible, but it does not license an inference from a video clip to cerebellar function [Wang et al., *The cerebellum, sensitive periods, and autism*](https://pubmed.ncbi.nlm.nih.gov/25102558/).

Likewise, a Level 3 support designation says something about support needs, not a fixed neurobiological subtype. Severe motor-speech impairment or childhood apraxia may co-occur, but cannot be assumed from non-speaking status. Rewrite these passages as heterogeneous, non-diagnostic possibilities and remove claims of attenuated proprioceptive feedback, persistent vagal suppression, or specific circuit failure unless a treating clinician has independently assessed them and the product only stores their report as context.

### 2. Respect stimming without making it an internal-state decoder

The documentation is right to reject the view that all repetitive movement is purposeless pathology. Reviews find regulatory, attentional, and sometimes communicative functions, but also a sparse and heterogeneous evidence base [RRBI scoping review](https://doi.org/10.1016/j.rasd.2024.102458). A motion pattern can be enjoyable, regulatory, expressive, habitual, responsive to context, or several of these simultaneously. It can also be unrelated to the preselected intent taxonomy.

The hard problem is **identifiability**: identical observable behavior can arise from different needs, while a single need can be expressed many ways. The same concern applies to facial, body, and vocal cues. The broad emotion-inference literature finds these cues context dependent and warns against calling agreement with observers “accuracy” [Barrett et al.](https://journals.sagepub.com/doi/10.1177/1529100619832930). This does not mean sensing is worthless; it means the system must report measurements and hypotheses, not hidden mental states.

Replace “child state diagnosis,” “translate … into intent,” and “actionable caregiver support protocols” with:

> “Observed pattern and context summary; similar prior episodes; caregiver/child-confirmable possibilities; uncertainty; and safe next steps.”

Every screen should offer `None of these`, `Not enough information`, `Other`, `Child confirmed via AAC/gesture`, and `Do not learn from this clip`.

### 3. The acoustic theory overclaims

There is evidence that some acoustic/prosodic characteristics differ in some autistic cohorts, but effect sizes are modest, results are inconsistent, and this literature concerns mostly speech—not a validated mapping from idiosyncratic non-speech vocalization to a child's hunger, discomfort, curiosity, or pain. A meta-analysis found only roughly 61–64% discriminatory accuracy for population autism-vs-control voice features [Fusaroli et al.](https://doi.org/10.1002/aur.1678), and a newer systematic review and meta-analysis again reports inconsistent prosody/voice-quality findings [Song, Kuang, & Chen, 2026](https://pubmed.ncbi.nlm.nih.gov/42287519/).

Therefore the following claims should be removed unless supported by child-specific prospective data: stable low-variance F0 means self-soothing; ±15–50 Hz trajectories reliably distinguish named intentions; and spectral or harmonic properties encode a particular affective valence. Preserve F0, intensity, spectral shape, duration, periodicity, and rhythm as measurable features. Treat their interpretation as an empirical, individual, uncertainty-bearing prediction task.

### 4. Clinical frameworks are not interchangeable priors

ABC/FBA can support a disciplined hypothesis process, but proper functional assessment combines multiple information sources, relevant medical considerations, direct observation, data review, and confirmation of a hypothesis rather than an automatic label [ABA practice guidance](https://pa.performcare.org/content/dam/amerihealth-caritas/performcare-pa/pdf/providers/quality-improvement/cpg/applied-behavior-analysis-practice-guidelines-treatment-of-autism-spectrum-disorder.pdf.coredownload.inline.pdf). A model cannot replace that assessment.

Ayres Sensory Integration has some evidence when it is faithfully delivered and judged on individualized functional goals, but its evidence is not a proof of the project’s proposed neurophysiological explanation or of automated recommendation. Reviews describe promising functional/developmental benefits and continued research needs [Schaaf et al.](https://pubmed.ncbi.nlm.nih.gov/29280711/); newer synthesis still notes ongoing debate [Acuña et al.](https://pubmed.ncbi.nlm.nih.gov/40193295/). Store framework-specific sources with their scope and evidence quality. Never train the LLM to treat one framework as settled causal truth.

### 5. AAC must be central, not downstream

The system should preferentially help the child communicate *now*: ensure their existing AAC is reachable, offer a personally co-designed visual/gesture/AAC confirmation interface, and record the child’s response when available. NIDCD's workshop calls for participatory research, heterogeneity-aware personalization, AAC innovation, and meaningful outcomes defined with affected people and families [NIDCD](https://www.nidcd.nih.gov/workshops/2023/summary). The outcome is increased autonomous, successful communication—not apparent model cleverness or reduction of stimming.

## Blocking scientific and engineering defects

### 1. The stated tensor shapes cannot all be true

For a 5-second, 48 kHz clip, `n_fft=2048` has a bin spacing of **23.4375 Hz**, not 21.5 Hz; 21.5 Hz is the approximate 44.1 kHz value. Frame count must name the STFT centering/padding convention: without centering it is `1 + floor((240000 - 2048)/160) = 1488`; common centred implementations yield a different count (typically 1501). `1500` is not self-justifying.

With the specified no-padding 2-D convolution over a `1500 × 128` mel image, kernel `16 × 8`, and stride `8 × 4`, the token grid is `186 × 31 = 5766`—not 2048. If padding, cropping, pooling, or a different patch layout produces 2048, document it explicitly and test it. Also, mel bands are non-uniformly spaced; “8 bins ≈ 172 Hz” is not valid over a 20 Hz–22.05 kHz mel scale. A 128-band mel transform deliberately smooths frequency detail; keep a parallel higher-resolution linear/CQT or raw-waveform branch if fine harmonic discrimination is a stated requirement.

The video path has equivalent inconsistency: 300 frames yield 299 ordinary frame differences, and thresholding produces a *variable* number of active patches. It cannot guarantee 1500 tokens without a deterministic top-k selection plus padding/mask (or a fixed pooled grid). A fixed pixel difference is also fragile to camera shake, automatic exposure, flicker, shadows, and caregiver movement. It removes static but potentially relevant context, including the object being requested, a communication partner, and posture.

**Recommended visual representation:** use a hybrid, not a ban. Stabilize frames; track person/regions of interest; use optical flow and pose/keypoint trajectories where consent and reliability permit; retain low-rate context/object features; fuse those with temporal video features. Compare that against temporal-difference-only input in an ablation study. Do not identify or evaluate “stims” as a normative target.

### 2. Qwen2.5-14B is not a 4096-wide model

The named base model has `hidden_size=5120`, 48 layers, 40 attention heads, 8 KV heads, and intermediate size 13,824 according to its published configuration [Qwen config](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/main/config.json). A `(B,64,4096)` prefix cannot be concatenated with its `(B,L,5120)` token embeddings. The project must either:

1. select a language model with a verified 4096 hidden size and update every model/memory claim; or
2. make the connector output 5120-wide vectors and retrain/validate it against that exact model.

The second option increases the already high trainable and activation cost. Model identity must be an explicit versioned configuration, not the interchangeable phrase “Qwen2.5-Omni / Qwen2.5-14B-Instruct.” They are materially different systems.

### 3. A prefix alone does not create sensory-language grounding

A Perceiver resampler is a valid compression component, but the claim that 64 arbitrary sensory vectors can be simply prepended to a frozen text LLM and yield clinically meaningful language is unsupported. Flamingo's resampler works in a system trained to align vision and language and uses learned cross-attention layers within a frozen LM [Flamingo](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/tackling-multiple-tasks-with-a-single-visual-language-model/flamingo.pdf). LLaVA likewise uses end-to-end multimodal instruction tuning to connect an image encoder and an LLM [LLaVA](https://arxiv.org/abs/2304.08485).

Training a literature LoRA does not align raw child-specific audio/video latents to language. It can instead make fluent, unjustified explanations more likely. First train an explicit **non-generative** prediction/retrieval head on child-confirmed outcomes. Only use an LLM after that to render a fixed, structured result into plain language; enforce a schema and forbid it from adding labels, causes, or treatments absent from the model output and retrieved, versioned source.

### 4. The stated memory explanation and parameter budgets are inaccurate

KV-cache *storage* grows roughly linearly with cached tokens (attention prefill compute is the part with quadratic dependence), not quadratically. For the stated Qwen2.5-14B configuration and FP16 K/V, the theoretical cache is about 1.5 GiB at 8192 tokens for batch 1: `8192 × 48 × 2 × 8 × 128 × 2 bytes`. A 3548-token prefix alone is about 0.65 GiB. Actual peak will be higher because of weights, activations, allocator behavior, batch size, and prefill, but “10 GB KV for 8k” should be measured rather than asserted. MLX-LM provides fixed-size/rotating caches, prompt caching, and configurable prefill steps specifically to trade memory for latency [MLX-LM](https://github.com/ml-explore/mlx-lm).

The 160M total trainable-parameter claim also cannot follow from the shown projector. With 4096-wide Q/K/V/O projections and a 4096→16384→4096 FFN, the projector alone is approximately 202M weights before layer norms/biases. Applying rank-64 LoRA to all listed Qwen 14B attention and MLP linear layers is also far above the stated 85M (roughly 275M under the published dimensions). The exact count depends on target names and orientation, but the present 8 GB optimizer-state budget is unsubstantiated. Count parameters from the instantiated model and benchmark peak unified memory under realistic batch, sequence, prefill, and concurrent-server load before locking a hardware invariant.

### 5. The continuous-learning design is unsafe as written

Replay is a useful family of techniques, but no evidence establishes that exactly 80% historic / 20% daily data prevents forgetting in this task, nor that it is always optimal. Continual-learning literature uses replay, regularization, distillation, and selective memory; the useful mix depends on data distribution, buffer policy, and metrics [continual-learning review](https://doi.org/10.1016/J.NEUNET.2019.01.012). An exact ratio is a tunable experiment parameter, not a biological or safety invariant.

Automatic nightly updates create a second and more serious problem: a bad or mistakenly confirmed label can change future caregiver-facing behavior with no validation gate. Mean-pooled embeddings used for historical retrieval will also drift as the encoder changes, invalidating nearest-neighbor geometry. Version all embeddings, preserve an immutable raw/feature record with consent, and either use a frozen retrieval encoder or re-embed a versioned corpus before changing retrieval. Do not use a mutable Chroma collection as both the benchmark and the training source.

Recommended lifecycle:

1. keep a frozen production model;
2. train a candidate only after a minimum number of independently confirmed, sufficiently diverse examples;
3. evaluate it against time-separated, context-stratified holdouts and a locked “never train” safety set;
4. require predefined improvement in calibration, coverage, and per-class error without safety regression;
5. obtain caregiver/clinician review; then promote or discard the candidate;
6. retain rollback, data/model lineage, and a change log.

## Safer data, output, and learning design

### Data schema

The current ontology confounds at least four different things: observable form, contextual antecedent, hypothesized function/need, and response. Replace the single `parent_tag` with separate records:

| Field group | Examples | Rule |
|---|---|---|
| Observation | vocal duration, motion rhythmicity, object/partner present, time window | Measured or explicitly caregiver-observed; no inferred diagnosis. |
| Context | meal timing, transition, noise, sleep/illness report, location | Allow unknown and free text; record observer/source. |
| Hypothesis | “may want drink”, “may be overwhelmed”, “unclear” | Multiple labels allowed; caregiver confidence and rationale; never ground truth by default. |
| Action and outcome | offered AAC, water offered, break offered, child acceptance/refusal, later resolution | This is the most useful learning signal. It supports causal humility. |
| Provenance | model/data/schema version, consent status, retention date, annotator role | Required for reproducibility, audit, withdrawal, and safe rollback. |

Separate `possible pain/acute distress` from ordinary intent labels. It should never be silently scored as “sensory overload.” Define a conservative escalation card designed with the child's clinicians and caregivers (for example: seek urgent medical advice for an agreed set of red flags), with no model diagnosis.

### Output contract

Do not let the LLM write an unconstrained “translation.” Return a structured, non-diagnostic object such as:

```text
Observed: repeated right-hand movement and sustained vocalization; clip quality adequate.
Similar episodes: 3 prior caregiver-confirmed episodes (dates and summaries shown).
Possibilities: offer AAC choice/communication board; offer a quiet break; check usual routine cue.
Confidence/coverage: insufficient evidence to rank a single explanation.
Confirm: child/caregiver response, or “none of these.”
Safety: if agreed red flags are present, follow the child's existing care plan.
```

The correct default for an unfamiliar pattern is “I do not know; here is a way to ask or observe,” not a plausible clinical narrative. Medical ML literature specifically identifies abstention and uncertainty communication as necessary safeguards [Ghassemi et al.](https://www.nature.com/articles/s41746-020-00367-3). Use calibrated discriminative probabilities, prediction sets, or a conservative reject option; display both coverage and observed calibration on held-out data. Do not expose a fabricated numeric “confidence” that has not been calibrated.

### Evaluation before any real-world claims

Run a preregistered single-case / N-of-1 study with the child, caregivers, AAC/SLP and OT/medical partners as appropriate, and an ethics/privacy review. Use time-separated and context-separated splits; never split adjacent clips from the same event across train and test. Collect independent, delayed confirmation when feasible so the same caregiver's expectation is not both label and outcome.

Predeclare:

- target use, exclusion cases, red-flag policy, and explicit non-uses;
- label definitions and adjudication for disagreement;
- primary benefit: increased successful child-confirmed communication or caregiver usefulness, not inferred emotion accuracy;
- accuracy/precision/recall by possibility label, context, capture condition, and time;
- calibration error, abstention rate, prediction-set coverage, and drift;
- critical errors: missed pain/urgent-care escalation, false reassurance, and harmful/culturally invalid framing;
- comparison baselines: caregiver notes alone, nearest prior episode, structured context-only model, raw-features model, pretrained/hybrid model, and no-LLM output;
- a stopping rule and rollback rule.

Treat poor video/audio quality, camera motion, new settings, illness, new medications, and out-of-distribution patterns as abstention conditions. NIST's AI RMF calls for defined human oversight, representative human-subject evaluation, deployment-condition testing, production monitoring, documented limits of generalizability, and safe failure [NIST AI RMF Core](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/).

## Privacy, security, and regulatory posture

Offline inference meaningfully reduces transfer risk, and a parent-owned local vault is the right default. A 60 fps child video stream plus health/education records remains exceptionally sensitive, however. “Encrypted directories” is insufficient; the system needs encryption at rest and in transit over LAN, paired-device authentication, least-privilege processes, recovery and deletion semantics, audit logging, secure backups, dependency updates, and a local-network threat model. Use test fixtures/synthetic media only in source control.

### Always-on encrypted Mac vault

The desired operational flow is valid: the Mac may be screen-locked but still logged in, receive a clip from the paired Flutter app, encrypt/store it, and begin processing without a parent standing at the computer. That requirement changes how Keychain access must work.

Use a signed **per-user app helper or LaunchAgent**, rather than a system-wide/root daemon. Apple documents that the modern Data Protection Keychain is available in a logged-in user context and recommends the `SecItem` API with `kSecUseDataProtectionKeychain=true`; it is not available to a system `launchd` daemon [Apple TN3137](https://developer.apple.com/documentation/Technotes/tn3137-on-mac-keychains). The screen being locked is not the same as the user logging out, so this user-context helper can remain available while the Mac is awake.

Recommended key hierarchy:

```text
Flutter phone (paired once) -- mTLS + request signature --> Mac per-user helper
                                                            |
                                                            +--> per-clip random AES-256-GCM key (DEK)
                                                            |         |
                                                            |         +--> encrypted media / derived features / metadata
                                                            |
                                                            +--> vault key stored in Data Protection Keychain
                                                                        |
                                                                        +--> wraps each clip DEK
```

- Store encrypted clip data, embeddings, and metadata outside Keychain; store only compact cryptographic keys in Keychain. Use an authenticated encryption mode such as AES-256-GCM, a fresh nonce for every encryption, and authenticated associated data containing clip ID, schema version, and ciphertext version.
- Make the vault key **device-only and non-synchronizing**. Do not set `kSecAttrSynchronizable`; Apple notes that `kSecUseDataProtectionKeychain=true` provides modern keychain behavior without iCloud synchronization [Apple Keychain item guidance](https://developer.apple.com/documentation/security/ksecusedataprotectionkeychain).
- Configure a background-capable key-access policy such as “after first unlock” for this *processing key*. This allows the helper to process uploads while the screen is locked. It should be a consciously selected setting, visible to the parent.
- Require Touch ID/password for a separate **private-review/export key** or for opening the timeline, exporting raw media, changing retention settings, pairing a device, and revealing highly sensitive notes. Apple supports per-item access controls and user-presence/biometric gating [Apple access controls](https://developer.apple.com/documentation/security/restricting-keychain-item-accessibility).
- Use one DEK per clip (or per short recording bundle), wrapped by the vault key. Removing its wrapped DEK implements cryptographic erasure on deletion; also delete the ciphertext and log the event. An optional encrypted backup must have an explicit recovery design.

There is an irreducible tradeoff: if the system can decrypt and analyze raw clips while the screen is locked, the background helper must be able to obtain the processing key without Touch ID at that moment. This protects against a powered-off/stolen disk (with FileVault enabled) and ordinary unauthorized app access, but it cannot protect against malware already executing with the same logged-in user's authority. Reduce that residual risk with FileVault, a non-admin macOS account for daily use, hardened runtime/code signing, a minimal helper interface, local firewall rules, no inbound WAN exposure, and prompt security updates. Do not misrepresent this mode as “Touch-ID-protected processing while locked”; it is **automatic encrypted processing while locked**.

For upload security, pair each phone on the Mac while a parent is present, then issue a device-specific client credential. Require mutual TLS, a per-request signature/nonce to prevent replay, rate/size limits, and a local allowlist. Never rely on mDNS discovery or a LAN IP address as authentication. The Flutter app should store its client credential with iOS Keychain or Android Keystore, not in application preferences.

HIPAA applies when the developer/operator is a covered entity or business associate, not automatically to every family-owned app; other US state, education, consumer-privacy, and non-US rules may apply. Do not make compliance claims without jurisdiction-specific counsel. When the product stores electronic PHI in a regulated setting, HHS identifies administrative, technical, and physical safeguards under the Security Rule [HHS](https://www.hhs.gov/hipaa/for-professionals/security/laws-regulations/index.html).

The product language matters even for personal use. A private tool is not the same thing as a marketed clinical product, but claims of “child state diagnosis,” clinical reasoning, or treatment protocols can change that analysis. FDA's current CDS guidance explicitly addresses software intended for patients or caregivers as well as clinicians [FDA CDS guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software). Obtain specialist regulatory and privacy advice before clinical deployment, sharing beyond the family, or marketing. Maintain a risk file, model/data change-control plan, usability testing, and an incident pathway from the start.

## Recommended phased architecture

### Phase A — foundations and participatory design (do this before model building)

1. Co-design with the child where possible and the family; optionally involve the child's AAC/SLP, OT, pediatric contact, and an autistic advisor. Define benefit, unacceptable outputs, preferred language, consent/assent, data retention, and escalation policy.
2. Rebrand the product and requirements around communication support and observation. Remove claims of diagnosing autonomic status, pain, intent, or neural mechanisms.
3. Implement the always-on encrypted Mac vault, paired-phone upload, schema provenance, capture-quality checks, consent/withdrawal/delete flows, and an observation-first review UI.
4. Establish a fixed dataset ledger and a locked test/safety set before model training.

### Phase B — useful without generative AI

1. Ship a local clip diary, timeline, caregiver notes, and child-confirmable AAC-choice workflow.
2. Add deterministic signal summaries (duration, rhythm, acoustic levels, motion/pose trajectories, capture quality) and transparent retrieval of similar *caregiver-verified episodes*.
3. Evaluate whether this alone improves response time, caregiver confidence, and child-confirmed communication. It is a very strong baseline.

### Phase C — conservative personalized prediction

1. Use a small, versioned multimodal encoder; compare raw spectral, pretrained acoustic/video, and hybrid features through ablations.
2. Train a calibrated multi-label retrieval/classification head on action/outcome-confirmed events. Include `uncertain/other` as a first-class result.
3. Gated candidate updates only; no autonomous nightly deployment. Tune replay composition and regularization empirically against retained benchmarks.

### Phase D — bounded language assistance

1. Use a curated, local, versioned research/document library for retrieval. Each source needs its author, date, licence/rights status, population, evidence level, and scope/limitations. The synthesis must link the parent to the precise excerpts used.
2. Do **not** use “all public papers and documentation” as an undifferentiated LoRA corpus. It would permanently mix contradictory, weak, outdated, and potentially rights-restricted material into weights, without showing why a specific statement was made. Retrieval is better for facts because it is inspectable, updateable, removable, and source-linked.
3. A carefully curated LoRA can still be useful for stable behavior: parent-respectful language, a structured evidence-first answer format, abstention, source citation, and a prohibition on invented causal or clinical claims. It should not be the primary repository of clinical facts.
4. Keep the LLM outside the predictive inference path. It may synthesize a rich report from measured features, personal-history retrieval, and cited literature, but it cannot add a diagnosis, causal story, intent label, or intervention beyond the structured evidence.
5. If direct multimodal language conditioning is still desired, run it as a research experiment with an explicitly aligned connector and matched labeled examples; it is not an inexpensive consequence of prefix concatenation.

## Required document corrections before implementation

**P0 — must change before any caregiver-facing use**

- Remove deterministic neurobiology, F0, vocalization, and stimming-to-intent claims from `DESIGN.md` and `README.md`.
- Replace clinical diagnosis/translation/protocol wording with observational, uncertainty-bearing support wording.
- Remove the ASR, vision-backbone, text-description, and 80/20 bans from `INVARIANTS.md`; replace them with measurable retention, privacy, calibration, safety, and evaluation requirements.
- Correct Qwen2.5-14B dimension to 5120 or select and document a different compatible model.
- Add abstention, confirmation, red-flag, human-oversight, model-promotion, rollback, and incident-response requirements.
- Add the two-key privacy posture: automatic encrypted processing while screen-locked, and Touch-ID-gated private review/export. Document the explicit residual-risk tradeoff.

**P1 — correct before training**

- Resolve the STFT, patch-grid, mel-band, video-difference, and variable-token contradictions above in `SPECS.md`; create executable shape tests from the actual preprocessors.
- Replace theoretical memory rows with measured benchmark envelopes for each exact model/configuration. Include batch, capture window, prefill, generation length, cache type, and concurrent workload.
- Count trainable parameters from the instantiated projector/LoRA targets; reduce targets or use a much smaller model/adapter if the measured budget demands it.
- Version embeddings and corpus snapshots; separate mutable training data from immutable evaluation and retrieval indexes.

**P2 — improve scientific quality**

- Add source provenance, date, rights, evidence level, population, and framework limitations to research RAG documents. Do not bulk-train a LoRA on undifferentiated public literature; use it only for carefully defined behavioral/formatting objectives.
- Add a source-backed scientific claims table to the project docs; label unsupported design hypotheses as hypotheses.
- Add an ablation/evaluation protocol and a results registry. A component becomes an invariant only after it earns that status empirically for the stated child, task, and deployment setting.

## Bottom line

The humane premise is strong: take non-speaking communication seriously, avoid forcing experience through a text-only lens, keep the family in control, and allow the parent to receive deep, evidence-linked insight from each clip. The project should become **less certain and more useful**: observe carefully, combine the clip with personal history and a curated research library, explain the evidence for each possibility, ask/enable communication, say when evidence is inadequate, and validate every learned association prospectively. The current sensory-to-language system is a promising research direction; the right parent-facing promise is rich, transparent decision support—not a claim to know the child's internal state.

## Claim-to-source ledger

| Claim used in this review | Source and access note |
|---|---|
| Participatory, heterogeneous, AAC- and personalization-oriented research priorities for minimally verbal/non-speaking autistic people | [NIDCD workshop summary, 2023](https://www.nidcd.nih.gov/workshops/2023/summary); primary US research-agency workshop summary. |
| Observed facial/body/vocal cues do not justify direct emotion detection claims; context and observer agreement matter | [Barrett et al., 2019](https://journals.sagepub.com/doi/10.1177/1529100619832930); peer-reviewed critical review. |
| Connectivity findings in autism are inconsistent and moderated by heterogeneity | [Vasa et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28083565/); peer-reviewed review. |
| RRB/stimming functions are plural and regulation is one, not exclusive, function | [Hinton et al., 2024](https://doi.org/10.1016/j.rasd.2024.102458); scoping review. |
| Prosodic population findings are modest/inconsistent, not an individual intent decoder | [Fusaroli et al., 2017](https://doi.org/10.1002/aur.1678) and [Song, Kuang, & Chen, 2026](https://pubmed.ncbi.nlm.nih.gov/42287519/); systematic reviews/meta-analyses. |
| ASI evidence is targeted to individually defined functional goals and remains qualified | [Schaaf et al., 2018](https://pubmed.ncbi.nlm.nih.gov/29280711/) and [Acuña et al., 2025](https://pubmed.ncbi.nlm.nih.gov/40193295/); systematic reviews. |
| Qwen2.5-14B dimensions | [Official Qwen configuration](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/main/config.json); first-party model config. |
| Perceiver compression and need for trained cross-attention integration | [Flamingo](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/tackling-multiple-tasks-with-a-single-visual-language-model/flamingo.pdf) and [LLaVA](https://arxiv.org/abs/2304.08485); primary model papers. |
| MLX cache/prefill control options | [MLX-LM documentation](https://github.com/ml-explore/mlx-lm); first-party project documentation. |
| Uncertainty/abstention is a safety requirement in clinical ML | [Ghassemi et al., 2020](https://www.nature.com/articles/s41746-020-00367-3); peer-reviewed clinical ML review. |
| Human oversight, deployment-valid evaluation, safety monitoring and privacy/bias assessment | [NIST AI RMF Core](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/); US national standard/guidance. |
| Modern macOS Keychain choice, per-user Data Protection Keychain behavior, and why a system daemon is unsuitable | [Apple TN3137: On Mac keychain APIs and implementations](https://developer.apple.com/documentation/Technotes/tn3137-on-mac-keychains); first-party platform guidance. |
| Per-item Keychain access controls, user presence, and biometric gating | [Apple: Restricting keychain item accessibility](https://developer.apple.com/documentation/security/restricting-keychain-item-accessibility); first-party platform guidance. |
| Caregiver-facing CDS/device boundary requires assessment | [FDA CDS guidance, 2026](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software); current US regulator guidance. |
| ePHI security safeguards in regulated settings | [HHS HIPAA Security Rule summary](https://www.hhs.gov/hipaa/for-professionals/security/laws-regulations/index.html); current US government guidance. |
