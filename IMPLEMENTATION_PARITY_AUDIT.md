# Project N — Implementation Parity Audit

**Audit target:** commit `7328907` *"feat(core): implement Phase 0, 1, and 2 offline backend and triage pipeline"* plus uncommitted working tree
**Scope:** Python backend only — `extraction/`, `models/`, `rag/`, `server/`, `storage/`, `training/`, `tests/`, CI configuration
**Explicitly out of scope:** `app/` (Flutter companion), `ui/` (React dashboard) — deferred by project owner
**Audit axis:** *functional parity between what the documentation declares and what the code actually does*
**Verdict:** **REJECTED for Phase 0/1/2 sign-off.** Substantial, well-structured work — but the system does not yet do what the documentation says it does, and in several places it emits fabricated values through interfaces documented as measurements.

---

## 0. How to read this report

This audit does **not** evaluate code style, architecture taste, or test count. Those are good. It evaluates one thing:

> When `docs/` says the system measures, computes, retrieves, encrypts, or verifies X — does the code measure, compute, retrieve, encrypt, or verify X?

Where the answer is no, the finding is recorded regardless of how reasonable the placeholder was as an intermediate step. This standard is deliberately strict, because Project N's entire ethical premise — stated in `INVARIANTS.md` Invariant 6 ("must never present an inferred state as fact") and `docs/evaluation_protocol.md` §1.1 ("observer agreement does not establish ground truth") — is *epistemic honesty about what is and is not known*. A system that silently substitutes zeros, random weights, or literature-borrowed thresholds for measurements it did not make violates that premise in software, no matter how carefully the documents state it in prose.

### Definition of Done applied throughout

A feature is counted as **implemented** only if all of the following hold:

1. **No synthetic substitution.** No code path replaces absent real input with zeros, constants, neutral poses, or generated signals and then proceeds as if the result were measured.
2. **No fabricated numerics.** No value reaching an API response, an SSE payload, a database row, or a vector store record originates from an untrained model, a hardcoded literal, or a placeholder string.
3. **No silent degradation.** When a real input or dependency is unavailable, the system must raise, abstain, or set an explicit quality/availability flag that propagates to the caller — never continue quietly.
4. **Declared thresholds are derived, not borrowed.** Any numeric threshold used in a safety path must be computed from Child N's own data or carry an in-code citation to the source that validates it *for the quantity actually being measured*.
5. **Doc and code agree.** Field names, vocabularies, layer definitions, and endpoint contracts match `docs/SPECS.md` / `INVARIANTS.md` exactly.

---

## 1. Executive summary

| Area | Status |
| :--- | :--- |
| Psychometric instruments (`models/nccpc.py`) | **Implemented** — exemplary |
| Facts / Human-in-the-Loop confirmation gate | **Implemented** |
| Vault envelope encryption (media) | **Implemented** |
| SSE bus mechanics | **Implemented** (event *types* incomplete) |
| Windowing, STFT, CQT, optical flow | **Implemented** (unnormalized; see §4) |
| Pitch / voice-quality descriptors | **Partial + mislabeled** — not the clinical measures they are named after |
| Pose / kinematic substrate | **Not implemented** — MediaPipe never called |
| Metric projection head | **Not implemented as a model** — random untrained weights |
| Episodic retrieval loop | **Not implemented** — no write path |
| Four-layer output contract (L1–L4) | **Partial** — L3 and L4 absent, layers renumbered |
| Signal-quality abstention gates | **Not implemented** |
| Physiological channel | **Not implemented** |
| Relational DB encryption (SQLCipher) | **Not implemented** |
| Transport security (mTLS, HMAC, nonce) | **Not implemented** |
| Model promotion / rollback / lineage | **Not implemented** |
| Training, splits, calibration, baselines B1–B4 | **Not implemented** (`training/` is empty) |
| Retention purge / privacy zones | **Not implemented** |
| CI verification of the implementation | **Never executed** — and cannot pass as configured |

**Counts:** 36 tests pass; 92% line coverage. Both numbers are real and creditable. Neither detects any P0 finding below, because the tests exercise the same synthetic paths the production code falls back to.

---

## 2. P0 — Fabricated data reaching documented interfaces

These are the blocking findings. Each one causes the system to emit a value through an interface that `docs/SPECS.md` describes as a measurement or a model output, when it is neither.

### P0-1 — `/analyze` never reads the clip it was given

**Location:** [`server/routes/clips.py:141-154`](server/routes/clips.py)

```python
def _run_analysis() -> None:
    # Default to synthetic baseline window if raw video isn't unpacked
    import numpy as np
    audio_synth = np.zeros(240000, dtype=np.float32)
    frames_synth = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in range(150)]
    analysis.analyze_sensory_clip(clip_id=clip_id, audio_pcm=audio_synth, ...)
```

`POST /api/v1/clips/{id}/analyze` accepts a `clip_id`, ignores it as a data source, and runs the entire pipeline over 5 seconds of digital silence and 150 black frames. The vault write path (`finalize`) and the analysis read path are not connected — there is no demux, no decrypt, no `VaultManager.decrypt_clip` call anywhere in `server/`.

The consequences compound:

- The pipeline returns an L1 "measured observation" block derived from silence.
- `AnalysisService` then **writes this to SQLite as a real episode** ([`server/services/analysis_service.py:252-266`](server/services/analysis_service.py)), with `captured_at` hardcoded to the literal string `"2026-09-14T00:00:00Z"`.
- An `analysis_completed` SSE event is published announcing a completed analysis.

There is no flag anywhere in the response, the DB row, or the event indicating the input was synthetic.

**Required:** implement the real path — retrieve ciphertext, unwrap DEK, decrypt, demux audio to 48 kHz float32 PCM and video to RGB frames at 30 fps, then analyze. If a clip cannot be decoded, the endpoint must fail loudly and must not create an episode row. Delete the synthetic fallback entirely; do not make it conditional.

### P0-2 — The metric embedding is the output of an untrained, randomly initialized network

**Locations:** [`models/projection.py`](models/projection.py), [`models/clip_encoder.py:57`](models/clip_encoder.py), [`server/deps.py:62-71`](server/deps.py)

There is no checkpoint anywhere in the repository. Repository-wide, there are **zero** occurrences of `load_weights`, `save_weights`, `model.freeze()`, `mx.random.seed`, or `np.random.seed`. `MetricProjectionHead` and `ClipSequenceAttentionPool` are constructed with fresh `nn.Linear` layers and explicit `mx.random.normal(...) * 0.02` parameters at process start.

Therefore:

- `output["metric_embedding"]` returned to API clients is a random projection of the input, not a learned behavioral embedding.
- The weights differ on **every server restart**, so any embedding persisted in one process is incomparable to any embedding produced in another. This silently voids `INVARIANTS.md` Invariant 9 (lineage / no silent embedding drift) before the lineage machinery is even written.
- `encoder_version` is the hardcoded string `"v1.0.0"` in four places ([`analysis_service.py:243,256`](server/services/analysis_service.py), [`matcher.py:27`](models/matcher.py), [`vector_store.py:61`](rag/vector_store.py)), asserting a version identity that corresponds to no artifact.

**Required:** (a) a training entrypoint in `training/` that produces a checkpoint; (b) checkpoint loading at service construction, failing hard if no checkpoint is present rather than falling back to random init; (c) `encoder_version_id` derived from the loaded checkpoint's content hash, never a literal; (d) deterministic seeding per `ENGINEERING_STANDARDS.md` §4.2. Until (a)–(c) exist, `metric_embedding` must not be returned to clients or written to any store.

### P0-3 — MediaPipe is never called; real video silently yields a zero pose matrix

**Location:** [`extraction/pose_tracker.py:117-127`](extraction/pose_tracker.py)

```python
# In real video, MediaPipe Holistic would process frames.
for t in range(t_len):
    if isinstance(frames[t], np.ndarray) and frames[t].shape == (TOTAL_LANDMARK_COORDS,):
        mat[t] = normalize_torso_landmarks(frames[t])
    elif isinstance(frames[t], np.ndarray) and frames[t].ndim >= 2:
        # Placeholder for direct camera frames: fallback normalized neutral pose
        mat[t] = np.zeros(TOTAL_LANDMARK_COORDS, dtype=np.float32)
```

`mediapipe` is a declared dependency in `pyproject.toml` and is never imported. The only branch that produces real landmark data requires the caller to have *already* supplied a 258-element landmark vector. Actual camera frames — the documented input — take the `ndim >= 2` branch and receive all zeros, with no warning, no log, and no quality flag.

This directly violates `INVARIANTS.md` Invariant 4: *"Pixel differencing alone must not be relied upon due to camera motion vulnerability. Pose/keypoint landmarks (e.g., MediaPipe Holistic, BlazePose) form the primary motion substrate."* With pose zeroed, the kinematic latent is optical flow alone — exactly the prohibited configuration.

It also structurally disables a safety path: `detect_acute_guarding` operates on the zeroed matrix, so `acute_guarding_detected` is permanently `False` for real video, and that flag is one of three inputs to the INV-7 distress screener.

**Required:** real MediaPipe Holistic integration producing 33 pose landmarks × 4 + 42 hand keypoints × 3. When landmark detection fails or mean confidence falls below the documented 0.40 threshold, the window must be marked low-quality and the clip must abstain (see P0-5) — never zero-filled.

### P0-4 — The episodic retrieval loop has no write path

**Locations:** [`rag/vector_store.py:44`](rag/vector_store.py), [`server/services/analysis_service.py:186`](server/services/analysis_service.py)

`VectorStore.add_episode_embedding` is called from exactly one place in the repository: `tests/test_storage.py:148`. Nothing in `server/` ever writes an embedding. There is no endpoint that marks an episode *confirmed* (the state `INVARIANTS.md` Invariant 9 calls "verified records"), and `EpisodeRepository.update_outcome` — which records the caregiver-observed outcome — is never called by any route.

`EpisodicPrototypeMatcher` requires ≥20 confirmed episodes before leaving the `collecting` state. Since nothing can ever be written, the matcher abstains permanently. The four-state lifecycle is well-designed and currently unreachable beyond state 1.

Compounding this: [`analysis_service.py:219`](server/services/analysis_service.py) reads `match_result.get("matches", [])`, but `EpisodicPrototypeMatcher.match` returns the key `"candidates"` ([`matcher.py:133`](models/matcher.py)). Even with a populated store, L2 would always be empty. No test catches this because no test exercises the service against a populated collection.

**Required:** a caregiver confirmation endpoint (`POST /api/v1/episodes/{id}/outcome` or equivalent) that records the observed action, outcome state, settling latency, and child response, and *on confirmation only* writes the embedding plus that metadata to ChromaDB tagged with the real encoder checkpoint id. Fix the `matches`/`candidates` key mismatch and add an integration test that drives the matcher through all four states via the service, not via a mock.

### P0-5 — Mandatory signal-quality abstention gates do not exist

**Source:** `docs/evaluation_protocol.md` §4.4 — *"The system **must** abstain with 100% probability if capture quality breaches deterministic bounds: Acoustic SNR < 12 dB or digital clipping > 5% of samples; Pose landmark mean detection confidence < 0.40; Excessive camera motion blur (optical flow velocity > 85 px/s)."*

A repository-wide search for `snr`, `clipping`, `0.40`, `85`, `signal_quality`, and `motion_blur` in `extraction/`, `models/`, `server/`, `storage/`, and `rag/` returns **no matches**. None of the three gates is implemented. There is no SNR estimator, no clipping detector, no landmark-confidence propagation (MediaPipe's `visibility` channel is preserved in the contract but never populated — see P0-3), and no flow-velocity ceiling.

This is the mechanism that prevents the system from offering confident-sounding output on unusable data. It is the load-bearing safety feature of the abstention design and it is absent.

**Required:** implement all three gates as a per-window `SignalQualityReport` computed in `extraction/`, aggregated per clip, surfaced in L1, and wired as a hard precondition that forces abstention before any retrieval or screening runs.

### P0-6 — `GET /episodes/{id}/media` returns a hardcoded byte string

**Location:** [`server/routes/episodes.py:60-64`](server/routes/episodes.py)

```python
# In production, retrieves ciphertext and unwrap DEK. For testing/mock:
media_bytes = b"mock_decrypted_mp4_video_bytes"
return Response(content=media_bytes, media_type="video/mp4")
```

The endpoint is documented in `docs/SPECS.md` §5.1 as *"Decrypted binary video stream (`video/mp4`)"* and returns a 30-byte literal with an MP4 content type. `VaultManager` is injected as `_vault` and unused.

**Required:** real vault retrieval, DEK unwrap, decryption, and a streaming response with HTTP range support. A `404` when the ciphertext is missing.

### P0-7 — Pairing returns a mock certificate against a hardcoded PIN

**Location:** [`server/routes/auth.py:44-50`](server/routes/auth.py), [`server/deps.py:23`](server/deps.py)

```python
if req.pairing_pin != "123456":
    ...
"client_cert_pem": "-----BEGIN CERTIFICATE-----\nMIIC...MOCK_CERT...\n-----END CERTIFICATE-----",
```

The submitted `csr_pem` is accepted and discarded. The PIN is a compile-time constant. The returned "certificate" is a placeholder string. `expires_at` is the hardcoded literal `"2027-09-14T00:00:00Z"` and is never enforced. `_valid_tokens` is seeded with two permanent development tokens, `"mock_local_paired_token"` and `"dev_token"`, which are valid in production builds.

`docs/SPECS.md` §5.1 specifies: mTLS with client certificates issued at pairing, `X-Timestamp` within a 300-second window, `X-Signature` as HMAC-SHA256 over `<timestamp>.<method>.<path>.<body_sha256>`, and a rolling nonce cache for replay rejection. None of the four exists. `X-Chunk-SHA256` on the chunk upload endpoint is also unimplemented.

**Required:** real CSR signing against a local CA, a one-time randomly generated pairing PIN surfaced to the user at setup, enforced token expiry, removal of the static development tokens, and the documented HMAC/timestamp/nonce middleware. If mTLS termination is deferred, say so explicitly in `INVARIANTS.md` rather than leaving the document asserting a control that does not exist.

---

## 3. P1 — Declared features with no implementation

Each item below is described in `docs/` as part of the delivered architecture. None has any code.

| # | Feature | Documented in | Present |
| :-- | :--- | :--- | :--- |
| P1-1 | Relational DB encryption (SQLCipher) | `INVARIANTS.md` §1, `SPECS.md` §5.2 | No — plain `sqlite3`, and pinned to `":memory:"` at [`deps.py:27-30`](server/deps.py), so no persistence at all |
| P1-2 | macOS Data Protection Keychain provider (`SecItem`, `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly`) | `INVARIANTS.md` §1 | No — only `MockKeychainProvider`, which is also the production default at [`vault.py:51`](storage/vault.py) |
| P1-3 | `POST /api/v1/models/promote` | `SPECS.md` §5.1 | No |
| P1-4 | `POST /api/v1/models/rollback` | `SPECS.md` §5.1 | No |
| P1-5 | Model lineage / checkpoint registry writes | `INVARIANTS.md` Inv. 9 | No — `model_checkpoints` table is created and never read or written by any code |
| P1-6 | Physiological channel (EDA/HRV, 50 steps × 64 dims) | `SPECS.md` §2.3, `INVARIANTS.md` Inv. 4 | No extractor; `x_physio=None` is hardcoded at [`analysis_service.py:152`](server/services/analysis_service.py) |
| P1-7 | L3 (caregiver context) output layer | `INVARIANTS.md` Inv. 6, `SPECS.md` §4.1 | No |
| P1-8 | L4 (versioned, cited evidence library) output layer | `INVARIANTS.md` Inv. 6, `SPECS.md` §4.1 | No — `evidence_coll` is created in `VectorStore` and never populated or queried |
| P1-9 | Parent View / Therapist View rendering | `README.md`, `INVARIANTS.md` Inv. 8 | No — `generate_render` is accepted at [`clips.py`](server/routes/clips.py) via `AnalyzeRequest` and silently ignored |
| P1-10 | `system_telemetry` and `four_layer_card` SSE events | `SPECS.md` §5.3 | No — only `processing_progress` and `analysis_completed` are published |
| P1-11 | `distress` query filter on `GET /episodes` | `SPECS.md` §5.1 | No |
| P1-12 | Forward-chaining temporal splitter | `evaluation_protocol.md` §2.2, `ENGINEERING_STANDARDS.md` §4.2 | No — `training/` is empty |
| P1-13 | Baselines B1–B4 | `evaluation_protocol.md` §3 | No |
| P1-14 | Platt / isotonic calibration of `τ_abstain`, ECE computation | `evaluation_protocol.md` §4.3 | No — `is_calibrated=False` is hardcoded at [`analysis_service.py:60`](server/services/analysis_service.py) |
| P1-15 | Locked safety holdout set and promotion gate | `evaluation_protocol.md` §5 | No |
| P1-16 | Automated rollback on 3 consecutive negative caregiver ratings | `evaluation_protocol.md` §5.2 | No — caregiver ratings are not captured at all |
| P1-17 | 90-day untagged-footage retention purge | `INVARIANTS.md` Inv. 1 | No |
| P1-18 | Train-partition-only feature standardization | `ENGINEERING_STANDARDS.md` §4.2 | No — see §4 |
| P1-19 | Gradient clipping ‖g‖₂ ≤ 1.0, deterministic seeds | `ENGINEERING_STANDARDS.md` §4.2 | No |
| P1-20 | LaunchAgent daemon packaging | `INVARIANTS.md` Inv. 1 | No |

P1-17 and the assent/dissent enforcement in `INVARIANTS.md` Invariant 1 are partly client-side and may legitimately land with the Flutter work; the **retention purge** is server-side and is not.

---

## 4. P1 — Scientific parity: measurements that are not what they are named

This section is separated because these defects are not missing code — they are code that runs, produces plausible numbers, and labels them with clinically meaningful names they do not carry. In a tool that a therapist will read, this is the most dangerous failure class in the repository.

### P1-21 — `cpp_db` is not Cepstral Peak Prominence, and its threshold is borrowed

**Location:** [`extraction/audio_pitch.py:109-115`](extraction/audio_pitch.py)

```python
q_region = cepstrum[q_min:q_max]
peak = float(np.max(q_region))
mean_c = float(np.mean(q_region))
cpp_arr[i] = max(0.0, float(20.0 * (peak - mean_c)))
```

Standard CPP is the difference, in decibels, between the cepstral peak and a **linear regression baseline** fitted across the quefrency range. This implementation uses the arithmetic mean as the baseline and multiplies a raw cepstral amplitude by 20. The result is not in dB and is not comparable to any published CPP value.

The severity comes from what consumes it. [`models/anomaly.py`](models/anomaly.py) and [`models/baseline_profiler.py`](models/baseline_profiler.py) apply `cpp_lower_limit_db = 4.0` — a cutoff whose meaning derives from the clinical CPP literature — to this non-CPP quantity. The default fallback `cpp_mean = 8.5` is likewise a literature-shaped constant applied to an incommensurate scale. This is a fabricated numeric standard inside the INV-7 medical triage path, which `evaluation_protocol.md` §4.5 governs under **zero tolerance**.

**Required:** implement CPP with a regression baseline and dB scaling, validated against a reference implementation on a shared fixture; **or** rename the field to something honest (`cepstral_peak_contrast_raw`) and derive its threshold purely from Child N's own distribution with no literature-anchored default.

### P1-22 — `jitter` and `shimmer` are not the perturbation measures

**Location:** [`extraction/audio_pitch.py:123-131`](extraction/audio_pitch.py)

`jitter_local` is computed as `|ΔF0| / max(F0, 50)` between consecutive **10 ms frames**. Clinical jitter is cycle-to-cycle *period* perturbation within a sustained voiced segment. Two consequences: the measure is frame-rate dependent rather than a property of the voice, and at every voicing boundary F0 steps from 0 to a few hundred Hz, producing a spurious maximum-magnitude spike. `shimmer_local` is `|ΔRMS_dB| / 80.0` — the divisor is an unexplained constant.

`hnr` uses the Boersma autocorrelation form and is defensible; `jitter_rap` and `shimmer_apq3` are 3-point median filters of the above, which is not what RAP and APQ3 denote.

**Required:** implement cycle-synchronous perturbation over voiced segments only, gated on a minimum voiced-run length, or rename all four fields and remove the clinical acronyms from `SPECS.md` §2.1, the docstrings, and the Therapist View contract.

### P1-23 — F0 tracking is documented as pYIN and implemented as integer-lag autocorrelation

**Location:** [`extraction/audio_pitch.py:73-79`](extraction/audio_pitch.py); `ENGINEERING_STANDARDS.md` §3 lists `audio_pitch.py # Autocorrelation / pYIN F0 tracking`

`np.argmax` over the lag search region returns an integer lag with no parabolic interpolation. At the documented 600 Hz ceiling, `min_lag = 80` samples, so adjacent lags map to 600.0 Hz and 592.6 Hz — a quantization floor of roughly 7 Hz that worsens toward the top of the range. There is no octave-error correction and no voiced/unvoiced probabilistic tracking.

`README.md` states the system exists to track *"subtle pitch inflections in his throat."* A 7 Hz quantization floor is not compatible with that claim.

**Required:** pYIN (available in the already-declared `librosa` dependency) or, at minimum, parabolic peak interpolation plus octave-error correction, with an accuracy test against synthetic glides of known F0.

### P1-24 — No feature normalization anywhere; the metric space is scale-dominated

`ENGINEERING_STANDARDS.md` §4.2 mandates z-score standardization fitted strictly on the training partition. Nothing implements it. Raw features enter `nn.Linear` directly with wildly heterogeneous scales: spectral centroid ~10³ Hz, F0 ~10² Hz, log-mel ~10⁰–10¹, CQT magnitude unlogged while mel is logged, jitter ~10⁻². `AttentionPool` scores the same unscaled vectors.

Whatever the projection head is eventually trained to do, its input geometry is currently dominated by spectral centroid magnitude. This is a correctness precondition for every retrieval claim in `evaluation_protocol.md` §4.2.

**Required:** a persisted, versioned `FeatureStandardizer` fitted on the training partition only, applied identically at inference, with its parameters bound to the encoder checkpoint id.

### P1-25 — Latent widths are padding, presented as capacity

[`audio_pipeline.py:52-54`](extraction/audio_pipeline.py) builds a `(500, 512)` matrix of which only columns 0–227 carry signal; [`kinematic_pipeline.py:52-53`](extraction/kinematic_pipeline.py) fills 386 of 512. There is no learned acoustic or kinematic encoder — `ACOUSTIC_LATENT_D = 512` describes a zero-padded concatenation, not a latent representation.

This is an acceptable staging decision, but `docs/SPECS.md` §2.1–2.2 and `docs/DESIGN.md` §3 read as though encoders exist.

**Required:** either implement the encoders or amend `SPECS.md` and `contracts.py` to state that these are concatenated feature widths with explicit padding, and record the padded range.

### P1-26 — Clip-level aggregation misreports L1 measurements

**Location:** [`analysis_service.py:171-174`](server/services/analysis_service.py)

`motion_rhythm_hz` is averaged across windows with `np.mean`. A clip with 3.0 Hz flapping in one window and no detected rhythm (`0.0`) in another is reported to the caregiver and therapist as `1.5 Hz observed`. The same applies to `cpp_db` averaged across silent windows, and to `f0_mean_hz` averaged across windows of differing voiced duration.

L1 is defined in `INVARIANTS.md` Invariant 6 as **Measured Observation**. A number that corresponds to no moment in the recording does not belong in that layer.

**Required:** report per-window distributions (median plus IQR, voiced-duration-weighted where appropriate) or the dominant-window value with its support, never an unweighted mean across windows including non-detections.

### P1-27 — Uncited magic constants in the safety path

- [`pose_tracker.py:94`](extraction/pose_tracker.py): guarding triggers on `np.any(lw_diff < -0.4)`. Uncited, uncalibrated, undocumented, and currently unreachable (P0-3).
- [`kinematic_pipeline.py:63`](extraction/kinematic_pipeline.py): rhythmicity requires `np.max(spec[stim_band]) > 0.1` — a scale-dependent magnitude threshold on an unnormalized FFT.
- [`pose_tracker.py:50-56`](extraction/pose_tracker.py): `shoulder_dist` has only an `1e-8` epsilon and no minimum-magnitude guard, so a near-degenerate or partially occluded pose divides coordinates by ~1e-4 and injects extreme values into the latent.
- [`baseline_profiler.py`](models/baseline_profiler.py): the fallbacks `f0_upper = max(μ + 2.5σ, 400.0)` and `cpp_lower = max(μ − 2.0σ, 3.0)` impose literature-shaped floors on a personal baseline. If Child N's true distribution is narrower, the floor silently overrides the personal calibration the whole design exists to produce.

**Required:** every constant in a safety path carries either a derivation from Child N's data or an in-code citation, and every clamp that can override a personal baseline is removed or justified in writing.

---

## 5. P2 — Contract drift between documents and code

These are cheap to fix and each one currently makes a documented guarantee false.

### P2-1 — The four output layers are renumbered, and two are missing

`INVARIANTS.md` Invariant 6 and `docs/SPECS.md:454` define:

| Layer | Documented meaning |
| :-- | :--- |
| L1 | Measured Observation |
| L2 | Comparable History |
| L3 | Context & Antecedents |
| L4 | Evidence Library (cited, versioned literature) |

[`analysis_service.py:192-250`](server/services/analysis_service.py) emits `layer1_sensory`, `layer2_hypotheses`, `layer3_dyadic`, `layer4_safety`. L3 (caregiver context) and L4 (cited evidence) — precisely the two layers that make output *traceable* rather than merely *generated* — do not exist, and the safety block has been promoted into L4's slot. The SSE payload keys also differ from the documented `L1_measured` / `L2_historical` / `L3_context` / `L4_evidence`.

This is a direct Invariant 6 violation, not a naming preference. Restore the documented four layers, move safety into its own sibling block, and add `tests/test_output_schema.py` (already named in `INVARIANTS.md` §3 as the INV-6 verifier) asserting all four are present and non-empty.

### P2-2 — `declined` vs `rejected` — the drift test cannot see the drift

`CAREGIVER_DECISIONS` in [`models/contracts.py:143`](models/contracts.py) contains `"declined"`. The SQLite `CHECK` constraint at [`storage/db_schema.py:71`](storage/db_schema.py) and `docs/SPECS.md:331` both say `'rejected'`. Writing the contract-legal value raises `IntegrityError` at runtime.

`tests/test_shapes.py:171-175` asserts the vocabulary against `contracts.py` only, so the test designed to prevent exactly this drift passes while the two sources disagree. **Extend `test_shapes.py` to assert every controlled vocabulary against the live `CHECK` constraints parsed from `db_schema.py`**, not just against the Python constants.

### P2-3 — `matches` vs `candidates`

Covered in P0-4. Listed here as a contract item: the matcher's return schema is undocumented, so nothing detects the mismatch. Define it as a typed dataclass or `TypedDict` in `models/contracts.py`.

### P2-4 — Layering and async standards

- `ENGINEERING_STANDARDS.md` §5.2 mandates `async def` for all I/O and `asyncio.to_thread` for Metal/CPU-heavy work. `AnalysisService.analyze_sensory_clip` is fully synchronous and most routes are `def`. FastAPI's threadpool makes this functional but non-compliant, and it will not hold under concurrent clip processing.
- `AGENTS.md` §3.2 declares `mx.fast.scaled_dot_product_attention` **mandatory** for attention blocks, and the §6 review checklist asserts it is used. Repository-wide occurrences: **zero**. There is no multi-head attention block at all — `AttentionPool` is single-vector additive pooling. Either implement the attention architecture `docs/SPECS.md` §3.1 describes, or correct both documents.
- `AGENTS.md` has duplicated section numbering (§4 contains subsections "3.1–3.5"; §7 contains "6.1–6.3"). This is the file agents are instructed to obey literally; fix the numbering.

---

## 6. P1 — Process and verification parity

### P1-28 — The implementation has never been through CI, and CI cannot pass as configured

`gh run list` shows the most recent **CI Quality Gate** run at `f2acfa6` — the standards commit immediately *preceding* the implementation. Commit `7328907` and everything after it are unpushed and unverified.

Worse, the workflow cannot succeed as written. [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on `ubuntu-latest` only, while `mlx` ships **macOS-arm64 wheels exclusively** (`uv.lock:2176-2178`, with `mlx-metal` marked `sys_platform == 'darwin'`). `uv sync --extra dev` will fail at install; `mypy`, `pytest`, and `tests/test_shapes.py` all import `mlx.core` at module scope.

`ENGINEERING_STANDARDS.md` §8.3 already specifies *"Linux & macOS runners."* Add a `macos-14` job, push, and let the gate that has already been written actually run against the code.

### P1-29 — 7 of 10 invariant verifiers named in `INVARIANTS.md` §3 do not exist

| Invariant | Declared verifier | Present |
| :-- | :--- | :-- |
| INV-1 Offline sandbox | `tests/test_offline_sandbox.py` | No |
| INV-2 Memory ceiling | `tests/verify_memory_ceiling.py` | No |
| INV-3 Weight freezing | `tests/verify_frozen_weights.py` | No |
| INV-4 Sensory bypass | `tests/test_sensory_pipeline.py` | No |
| INV-5 Model promotion | `python -m training.evaluate_candidate --strict` | No |
| INV-6 Output traceability | `tests/test_output_schema.py` | No |
| INV-7 Distress screener | `tests/test_distress_screener.py` | No |
| INV-8 Insight delivery | `tests/test_insight_delivery.py` | No |
| INV-9 Version lineage | `python -m rag.verify_lineage` | No |
| INV-10 MLX purity | `grep` scan | Yes (in CI and pre-commit) |

The table is headed *"To be activated upon Phase implementation"*, which is honest. But Phases 0–2 are now claimed complete, and **INV-1, INV-4, INV-6, and INV-7 all govern behaviour delivered in this commit**. Those four verifiers are now due.

### P1-30 — No coverage gate

`ENGINEERING_STANDARDS.md` §8.2 specifies 90% branch coverage for `models/` and 80% line coverage for `extraction/` and `server/`. `pytest-cov` is not a declared dependency and no gate enforces any threshold. Measured coverage is currently 92% line overall — good, and worth locking in before it decays. Note that the standard specifies **branch** coverage for `models/`; that is not currently measured at all.

### P1-31 — The forbidden-import scan is a substring grep

`grep -rnE "(openai|anthropic|vertexai|google\.generativeai|import torch)"` flags the word "anthropic" in a comment and misses `importlib.import_module("torch")`, `from torch import ...`, and any aliased import. For INV-10, which `README.md` presents as a headline guarantee, use an AST-based import scan.

### P1-32 — Roadmap state is stale

`README.md`'s roadmap shows Phases 0, 1, and 2 as unchecked while commit `7328907` claims all three. Given this audit, leaving them unchecked is currently the accurate state — but the roadmap should say *in progress* with a link to this document rather than being silently out of step with the commit log.

---

## 7. What is genuinely correct — do not regress it

Stated so the rework does not damage the parts that already meet the bar:

- **[`models/nccpc.py`](models/nccpc.py)** — exact Breau et al. (2002) item sets for both instruments, strict integer validation that correctly rejects `bool` despite `isinstance(True, int)`, NA tracking, subscale decomposition, and an `is_indeterminate` flag on high missingness. This is the reference standard for the rest of the repository.
- **[`storage/vault.py`](storage/vault.py)** — correct AES-256-GCM envelope encryption: fresh per-clip DEK, `clip_id` as AAD on both media and DEK wrap, nonce prefixing, and zero-overwrite erasure.
- **Facts module** (`routes/facts.py`, `storage/facts_repo.py`) — the Human-in-the-Loop confirmation gate is fail-closed by default and fully implemented. No placeholders.
- **`EpisodicPrototypeMatcher`'s four-state lifecycle** and **`AcuteDistressAnomalyDetector`'s cold-start fail-closed gate** — both correctly refuse to operate before calibration, with honest user-facing language ("does not assess or exclude pain"). The designs are right; they need real inputs.
- **Positional-encoding magnitude decoupling** in [`clip_encoder.py`](models/clip_encoder.py) — a subtle correctness issue, identified and handled properly, with the reasoning documented in both code and spec.
- **`docs/evaluation_protocol.md`** — prespecified baselines, forward-chaining splits, a locked safety set, and the explicit rejection of "accuracy vs. adult observer tags" as a success criterion. This is more rigorous than most published work in the area. Preserve it exactly.
- **Retention of `docs/report-source.md`** — keeping the critical external audit in-tree rather than absorbing it quietly.

---

## 8. Required remediation, in dependency order

**Gate A — stop emitting fabricated values (blocking, do first)**

1. Delete every synthetic and mock fallback in `server/`: `clips.py:141-154`, `episodes.py:60-64`, `auth.py:44-50`, `deps.py:23`. Replace with real implementations or explicit `501 Not Implemented`. No silent substitution survives this gate.
2. Remove `metric_embedding` from API responses and block all vector-store writes until a real checkpoint exists (P0-2).
3. Land `tests/test_offline_sandbox.py` (INV-1) and `tests/test_output_schema.py` (INV-6).
4. Add a `macos-14` CI job, push, and get a green run on the implementation commit.

**Gate B — make the sensory path real**

5. MediaPipe Holistic integration with confidence propagation (P0-3).
6. Vault → decrypt → demux → analyze, wired end to end (P0-1).
7. All three signal-quality abstention gates (P0-5).
8. Feature standardizer, fitted train-only, versioned with the checkpoint (P1-24).
9. Fix or honestly rename CPP, jitter, shimmer, RAP, APQ3; implement pYIN or interpolated F0 (P1-21, P1-22, P1-23).
10. Fix L1 aggregation to stop reporting means across non-detections (P1-26).

**Gate C — close the learning loop**

11. `training/`: forward-chaining splitter, deterministic seeds, gradient clipping, training entrypoint, checkpoint serialization.
12. Checkpoint loading with hard failure when absent; `encoder_version_id` derived from checkpoint content hash.
13. Caregiver outcome/confirmation endpoint; embedding write path; fix `matches`/`candidates` (P0-4).
14. `τ_abstain` calibration and ECE computation on forward-chaining splits (P1-14).

**Gate D — restore the documented contract**

15. L1–L4 per Invariant 6, including the cited evidence library backed by `evidence_coll` (P2-1, P1-8).
16. Parent View / Therapist View rendering; honor or remove `generate_render` (P1-9).
17. `system_telemetry` and `four_layer_card` SSE events (P1-10).
18. Reconcile `declined`/`rejected`; extend `test_shapes.py` to assert against live DB `CHECK` constraints (P2-2).

**Gate E — security and lifecycle parity**

19. SQLCipher; real macOS Keychain provider; remove `MockKeychainProvider` as production default (P1-1, P1-2).
20. Real CSR signing, generated PIN, enforced expiry, HMAC/timestamp/nonce middleware (P0-7).
21. Promote/rollback endpoints, checkpoint registry writes, lineage verifier (P1-3, P1-4, P1-5, INV-9).
22. 90-day retention purge (P1-17).
23. Baselines B1–B4 and the locked safety holdout gate (P1-13, P1-15).

**Standing rule for all gates**

> Any documented capability that will not be implemented in the commit that claims its phase must be removed from `INVARIANTS.md` / `docs/SPECS.md` or explicitly relabeled as *planned*, in the same commit. A document asserting a control that does not exist is treated in this audit as equivalent to a fabricated value — it makes a false claim to the clinicians and caregivers the documents are written for.

---

## 9. Acceptance criteria for re-audit

Phase 0/1/2 will be re-audited when all of the following hold:

- [ ] Repository-wide search for `mock`, `placeholder`, `synth`, `dummy`, `for now`, `in production` in `extraction/`, `models/`, `rag/`, `server/`, `storage/` returns matches only in test fixtures and in the intentionally-named `MockKeychainProvider` used *exclusively* by tests.
- [ ] No hardcoded timestamp, version string, PIN, token, certificate, or media payload in non-test code.
- [ ] Every value in an API response, SSE payload, DB row, or vector record traces to a measurement, a loaded checkpoint, or caregiver input.
- [ ] `POST /clips/{id}/analyze` fails cleanly on an undecodable clip and creates no episode row.
- [ ] A clip with pose confidence < 0.40, SNR < 12 dB, clipping > 5%, or flow > 85 px/s abstains, with the triggering gate named in the response.
- [ ] `test_shapes.py` asserts controlled vocabularies against the live SQLite `CHECK` constraints.
- [ ] `test_output_schema.py` asserts L1–L4 as documented in Invariant 6.
- [ ] CI is green on `macos-14` against the implementation commit, with a coverage gate at the `ENGINEERING_STANDARDS.md` §8.2 thresholds including branch coverage for `models/`.
- [ ] INV-1, INV-4, INV-6, INV-7 verifiers exist and run in CI.
- [ ] An end-to-end integration test drives a real (synthetic-but-decoded) media file from upload through vault, decrypt, extraction, projection, screening, retrieval, caregiver confirmation, and embedding persistence — with the matcher observed transitioning out of `collecting`.

---

## 10. Closing judgment

The documentation set is the strongest part of this project and is genuinely ahead of comparable published work — the invariant/tunable split, the prespecified evaluation protocol, and the refusal to treat observer agreement as ground truth are all correct and rare. The code is well-factored, well-typed, cleanly layered, and readable.

The failure is one of *sequencing and honesty at the boundary*: the enforcement machinery was written before the implementation and has never been pointed at it, and where the implementation could not yet reach the specification, it filled the gap with silent substitutes instead of explicit gaps. The result is a system that currently analyzes silence, projects it through random weights, compares it against an empty store, and reports the outcome through an interface documented as measurement.

This is recoverable and mostly mechanical. It is also far cheaper to fix now than after Phases 3–5 are built on top of it. But it should not be signed off as Phase 0/1/2 complete, and the roadmap should not be checked, until Gates A through C are closed.

One closing note on standard of care. In most projects a zero-fill fallback is a pragmatic stub. Here, the documents promise a non-verbal child's caregivers and clinicians that the system will never present an inference as a fact and will say "I don't know" when it does not know. Code that substitutes silence for a recording, random weights for a trained model, and literature thresholds for a personal baseline — without saying so — breaks that promise more completely than any missing feature does. Every item in §2 and §4 should be read in that light.

---

*Audit performed against commit `7328907` and the uncommitted working tree. All findings verified by direct file inspection, repository-wide search, execution of the test suite (36 passed), a coverage measurement pass, and inspection of GitHub Actions run history.*
