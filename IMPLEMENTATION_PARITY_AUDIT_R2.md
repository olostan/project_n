# Project N — Implementation Parity Audit, Round 2

**Audit target:** commit `9818e05`, covering Gates A–E (`897dec1`, `efa9c03`, `17d3f93`, `2ea88df`, `9818e05`)
**Previous audit:** [`IMPLEMENTATION_PARITY_AUDIT.md`](IMPLEMENTATION_PARITY_AUDIT.md) against `7328907`
**Scope:** Python backend only — `extraction/`, `models/`, `rag/`, `server/`, `storage/`, `training/`, `tests/`, CI
**Out of scope:** `app/` (Flutter, now committed at `7ce31df`), `ui/` (React dashboard)
**Audit axis:** functional parity between what the documentation declares and what the code does
**Suite state:** 57 tests pass locally (was 36)

---

## Verdict

**Substantial, genuine progress. Still not approvable for Phase 0/1/2 sign-off.**

Round 1 raised 7 P0 findings. **Four are fully closed, two are partially closed, one is unchanged.** That is real engineering, and the sensory path in particular is now honest in a way it was not before: the pipeline decodes actual media, runs actual MediaPipe, and refuses to emit an embedding it cannot stand behind.

However, this round surfaces **three new P0 findings that did not exist before**, two of which were *introduced by the remediation itself*:

1. A **complete, empirically verified authentication bypass** — the new pairing flow publishes its own PIN over an unauthenticated endpoint.
2. The new training entrypoint **writes fabricated evaluation metrics into the model promotion registry** — the exact defect class the last audit was written to eliminate, now relocated into the gate that governs model deployment.
3. The new INV-6 verifier **asserts the drifted output schema**, converting a known documentation violation into a CI-enforced contract.

The first two are more serious than anything in Round 1, because Round 1's fabrications were inert placeholders; these are live in the security boundary and the promotion gate. Findings 1 and 2 must be fixed before any further feature work.

---

## 1. Scorecard against Round 1

### Closed — verified

| ID | Finding | Evidence |
| :-- | :--- | :--- |
| **P0-1** | `/analyze` ran on synthetic zeros | [`clips.py:120-176`](server/routes/clips.py) now checks `vault.has_clip` → 404, calls `vault.retrieve_clip` + `demux_clip_bytes`, and publishes `processing_failed` **without creating an episode row** on decode failure. [`extraction/demux.py`](extraction/demux.py) is real ffmpeg + OpenCV decoding. Clean fix. |
| **P0-3** | MediaPipe never called | [`pose_tracker.py:102-194`](extraction/pose_tracker.py) instantiates `mp.solutions.holistic.Holistic`, extracts real pose + hand landmarks, derives per-frame confidence from `landmark.visibility`, and records `0.0` on detection failure so the gate can see it. The `shoulder_dist < 1e-3` guard from P1-27 is also in. |
| **P0-4** | No retrieval write path | [`episodes.py:90-150`](server/routes/episodes.py) adds `POST /{id}/outcome` (+ `/confirm` alias) writing full outcome fields and indexing to ChromaDB — **only when `get_clip_embedding` returns non-`None`**, i.e. only under a trained checkpoint. Correctly fail-closed. `matches`/`candidates` mismatch resolved at [`analysis_service.py:226`](server/services/analysis_service.py). |
| **P0-6** | `/media` returned a byte literal | [`episodes.py`](server/routes/episodes.py) now performs real vault retrieval with a 404 on missing ciphertext. |
| **P2-2** | `declined` vs `rejected` | [`contracts.py:143`](models/contracts.py) reconciled to `"rejected"`; [`test_shapes.py:260-291`](tests/test_shapes.py) now **parses the live `CHECK` constraints via regex and asserts parity** — exactly the fix requested. |
| **P1-31** | Substring grep for forbidden imports | [`tests/test_imports.py`](tests/test_imports.py) is a real AST visitor covering `torch`, `torchvision`, `torchaudio`, and the cloud SDKs. Wired into CI and pre-commit. |
| **P1-27** *(partial)* | Degenerate `shoulder_dist` | Guarded. The remaining magic constants in this finding are still unaddressed (see §4). |

### Partially closed

| ID | Finding | What landed | What did not |
| :-- | :--- | :--- | :--- |
| **P0-2** | Embeddings from an untrained network | Excellent gating: [`analysis_service.py:284-296`](server/services/analysis_service.py) emits `metric_embedding: None` and `encoder_version: "uncalibrated_v0"` unless `projection_head.is_trained`. Checkpoint save/load and hashing exist. | `get_analysis_service()` ([`deps.py:105-116`](server/deps.py)) still constructs `MetricProjectionHead()` with **no checkpoint load attempt**, so the server runs permanently uncalibrated. No trained checkpoint is produced from real data (see P0-9). |
| **P0-5** | Signal-quality gates absent | [`extraction/signal_quality.py`](extraction/signal_quality.py) implements all four gates with thresholds matching the protocol, surfaces them in L1, and emits a `signal_quality_abstained` SSE event. | **Abstention semantics are fail-open** — see P0-10. And `compute_audio_snr_db` does not compute SNR — see P1-33. |
| **P0-7** | Mock cert, hardcoded PIN, static tokens | Real X.509 CA in [`server/ca.py`](server/ca.py), genuine CSR signing, ephemeral rotating PIN, token expiry registry, static dev tokens removed. Solid work. | PIN is publicly readable (**P0-8**). No mTLS transport enforcement, no `X-Signature`/`X-Timestamp`/nonce cache, no `X-Chunk-SHA256`. Issued certs are never verified on any subsequent request, so the CA is currently decorative. |
| **P1-21** | CPP not CPP | Now uses a true linear-regression baseline over quefrency ([`audio_pitch.py:117-127`](extraction/audio_pitch.py)) — a real improvement. | Still `20.0 * max(prominence)` on raw log-power cepstral units, so the output is still not dB on the standard scale, and the `4.0` threshold in [`anomaly.py`](models/anomaly.py) remains literature-borrowed. No validation against a reference implementation, which was the stated acceptance criterion. |
| **P1-22** | Jitter/shimmer not perturbation measures | Now computed over contiguous voiced segments of ≥3 frames using period and amplitude ratios ([`audio_pitch.py:129-152`](extraction/audio_pitch.py)) — the correct *formula* shape. | Still sampled at the **10 ms frame hop**, not cycle-synchronously. At 250 Hz one glottal cycle is 4 ms, so consecutive "periods" are 2–3 cycles apart. This is not cycle-to-cycle perturbation. `jitter_rap` / `shimmer_apq3` remain 3-point median filters, which is not RAP/APQ3. |
| **P1-23** | F0 quantization | Parabolic interpolation added ([`audio_pitch.py:79-89`](extraction/audio_pitch.py)) — removes the ~7 Hz floor, which was the substantive defect. | Still autocorrelation, not pYIN; no octave-error correction. `ENGINEERING_STANDARDS.md` §3 still advertises "Autocorrelation / pYIN". |
| **P1-28** | CI never ran; ubuntu can't host MLX | `macos-14` job `apple-silicon-verification` added. | **Nothing has been pushed.** `origin/main` is still at `f2acfa6`; all seven implementation commits are local and unverified. `mlx>=0.22.0` remains an unconditional dependency with no platform marker, so the `ubuntu-latest` `quality-gate` job will still fail at `uv sync`. `ffmpeg` is now a hard runtime dependency and is installed in neither job. |
| **P1-29** | Missing invariant verifiers | Six new verifiers written: `test_offline_sandbox.py`, `test_sensory_pipeline.py`, `test_output_schema.py`, `test_distress_screener.py`, `verify_memory_ceiling.py`, `verify_frozen_weights.py`, `verify_lineage.py`. | `test_insight_delivery.py` (INV-8) still absent. **`verify_frozen_weights.py` and `verify_lineage.py` never execute** — they do not match pytest's `test_*` collection pattern and are invoked by neither CI nor pre-commit. `INVARIANTS.md:135` still documents INV-9 as `python -m rag.verify_lineage`, but the file lives at `tests/verify_lineage.py` and `rag/` contains only `vector_store.py`. And `test_output_schema.py` asserts the wrong contract (**P0-10**). |

### Not addressed

| ID | Finding | Status |
| :-- | :--- | :--- |
| **P2-1** | L1–L4 renumbered; L3 context and L4 evidence absent | Unchanged, and now **enforced** by the new verifier. `INVARIANTS.md:68-71` and `SPECS.md:454` are untouched; only 2 lines of `SPECS.md` changed across all five gate commits. |
| **P1-24** | No feature standardization | Unchanged. Repository-wide search for any standardizer/z-score fit returns nothing. Gate B item 8 not delivered. |
| **P1-26** | L1 aggregation misreports | **Improved but incomplete.** [`analysis_service.py:182-186`](server/services/analysis_service.py) now filters zeros before averaging CPP and rhythm — good. But it still reports an unweighted **mean** across heterogeneous windows rather than a distribution, so a 3 Hz window and a 5 Hz window still report "4.0 Hz observed". |
| **P1-1 / P1-2** | SQLCipher; real Keychain | Unchanged. No `sqlcipher` anywhere; `get_db()` still `init_db(":memory:")`; `MockKeychainProvider` still the production default at [`deps.py:88`](server/deps.py). |
| **P1-3 / P1-4 / P1-5** | promote / rollback / lineage writes | Endpoints still absent. `model_checkpoints` is now *written* by the trainer — with fabricated values (**P0-9**). |
| **P1-6** | Physiological channel | Unchanged; `x_physio=None` hardcoded. |
| **P1-8 / P1-9 / P1-10** | Evidence library, Parent/Therapist rendering, `system_telemetry` + `four_layer_card` SSE | Unchanged. `evidence_coll` still created and never populated; `generate_render` still accepted and ignored. |
| **P1-13 / P1-14 / P1-15 / P1-16** | Baselines B1–B4, calibration, promotion gate, auto-rollback | Unchanged. **No MRR, ECE, coverage, Platt, or isotonic computation exists anywhere in the repository** — the only occurrences of those terms are the hardcoded literals in the trainer. |
| **P1-17** | Retention purge | Unchanged. |
| **P1-30** | Coverage gate | Unchanged. No `pytest-cov`, no threshold enforced, branch coverage for `models/` still unmeasured. |
| **P2-4** | Async I/O; `mx.fast.scaled_dot_product_attention` | Unchanged. `analyze_sensory_clip` is still synchronous; SDPA still has zero occurrences. `AGENTS.md` still declares it mandatory and its §6 checklist still asserts it is used. |

---

## 2. New P0 findings

### P0-8 — Unauthenticated endpoint publishes the pairing PIN; full auth bypass

**Location:** [`server/routes/auth.py:37-40`](server/routes/auth.py)

```python
@router.get("/pin", response_model=PairingPinResponse)
def get_current_pairing_pin() -> dict[str, Any]:
    """Retrieves current ephemeral pairing PIN for local dashboard/display."""
```

There is no `Depends(verify_auth_token)`. Any host that can reach the daemon can read the live pairing PIN and immediately complete pairing.

**Verified empirically against the running app:**

```
GET  /api/v1/auth/pin        (no credentials)  -> 200 {'pairing_pin': '501535', 'expires_in_sec': 300}
POST /api/v1/auth/pair       (that PIN + own CSR) -> 200  signed client cert + 365-day bearer token
GET  /api/v1/episodes        (attacker token)   -> 200  {'episodes': [...], 'total': N}
```

The PIN's entire purpose is to be a **user-present** factor proving physical access to the Mac. Publishing it over the network reduces pairing to "can route to the host," which `INVARIANTS.md` Invariant 1 prohibits in terms: *"mDNS discovery or LAN IP address presence alone is never treated as authentication."* The resulting token grants access to Child N's episode records, and via `/episodes/{id}/media`, to decrypted home video.

This is a regression in effect, not just in form: the previous hardcoded `"123456"` was a known-bad constant, but this endpoint hands the current secret to any caller on request.

**Required:** delete the endpoint. The pairing PIN must be displayed only through a local-console or local-only-bound surface that cannot be reached from the network — not through the same API being protected. If a dashboard must display it, bind that route to `127.0.0.1` exclusively and require an already-valid session. Add a regression test asserting `GET /api/v1/auth/pin` returns 401/404 for an unauthenticated remote caller, and an end-to-end test asserting that pairing cannot be completed without an out-of-band PIN.

### P0-9 — The trainer writes fabricated evaluation metrics into the promotion registry

**Location:** [`training/train_projection.py:113-160`](training/train_projection.py), invoked by `main()` at line 203

```python
def save_and_register_checkpoint(
    ...
    retrieval_mrr: float = 0.85,
    holdout_coverage: float = 0.92,
    ...
                0.05,  # ece_score
                "passed",
```

`main()` calls `save_and_register_checkpoint` with all defaults. Running `python -m training.train_projection` therefore inserts a row into `model_checkpoints` asserting:

- `retrieval_mrr = 0.85` — never computed
- `holdout_coverage = 0.92` — never computed
- `ece_score = 0.05` — never computed
- `evaluation_status = "passed"` — never evaluated
- `extractor_version = "git_v1.0.0"` — a hardcoded literal
- `validation_manifest_hash` — derived from `TemporalSplitter([])`, an **empty episode list**

And the weights those metrics describe were trained by `build_synthetic_triplets()` ([line 163](training/train_projection.py)) on `mx.random.normal` noise — not on Child N's episodes.

This is the most serious parity defect in the repository, for three reasons:

1. **It is the promotion gate.** `evaluation_protocol.md` §5.1 makes promotion conditional on MRR ≥ 0.65, ECE ≤ 0.12, and coverage within 60–85%. A row claiming 0.85 / 0.05 / "passed" satisfies every automated criterion without a single measurement. The gate designed to prevent unsafe deployment now has a pre-filled pass sitting in its input table.
2. **One value is not merely unmeasured but out of spec.** `holdout_coverage = 0.92` exceeds the documented 85% ceiling. A fabricated number was chosen that the real protocol would have *rejected*.
3. **It reproduces the exact defect class Round 1 was written to eliminate**, moved from the inference path into the governance path.

Round 1 required: *"no value reaching an API response, SSE payload, DB row, or vector record originates from an untrained model, a hardcoded literal, or a placeholder."* A `model_checkpoints` row is a DB row.

**Required:**
- Delete `build_synthetic_triplets` from the production entrypoint. `main()` must load real confirmed episodes from the repository and refuse to run below a stated minimum.
- Delete all metric default arguments. `save_and_register_checkpoint` must accept metrics **only** as required arguments produced by an evaluation function that actually computes them.
- Implement `training/evaluate_candidate.py` computing MRR, P@3, ECE (M=5 bins), coverage, and the zero-tolerance safety-error count on the locked holdout, per `evaluation_protocol.md` §4 — this is also the INV-5 verifier `INVARIANTS.md` §3 already names.
- `evaluation_status` must default to `'pending'` and be settable to `'passed'` only by that evaluator.
- Until the evaluator exists, the trainer must not write to `model_checkpoints` at all.

### P0-10 — The INV-6 verifier enshrines the schema violation it exists to catch

**Location:** [`tests/test_output_schema.py:1-5, 24-28`](tests/test_output_schema.py)

```python
"""
Validates that analysis output strictly partitions into Layer 1 (Sensory),
Layer 2 (Hypotheses), Layer 3 (Dyadic Suggestions), Layer 4 (Safety/Clinical).
"""
    assert "layer2_hypotheses" in out
    assert "layer3_dyadic" in out
    assert "layer4_safety" in out
```

`INVARIANTS.md:68-71` — unchanged in this round — defines L3 as **Context & Antecedents** and L4 as **Evidence Library** (cited literature with author, year, population, evidence level). `SPECS.md:454` and `SPECS.md:492-495` agree. The verifier's own docstring redefines the layers to match the code, then asserts them.

The result is worse than having no verifier. `INVARIANTS.md` §3 lists `tests/test_output_schema.py` as the enforcement mechanism for INV-6; CI will now go green on it; and the two layers that make output *traceable* rather than merely generated — caregiver context and cited evidence — remain absent while a passing test certifies compliance.

There is also a second-order problem: both tests in this file pass `frames = [np.zeros(258)]`, which takes the pre-extracted-landmark branch at [`pose_tracker.py:146`](extraction/pose_tracker.py) and is assigned a synthetic `confidence = 1.0`. So the INV-6 verifier never exercises the quality gates, and never sees a real frame.

**Required:** rewrite the verifier against `INVARIANTS.md` Invariant 6 as written — assert `L1_measured`, `L2_historical`, `L3_context`, `L4_evidence` (or the four documented layer names) are present and non-empty, and assert L4 entries carry citation metadata. Then implement L3 and L4 so it passes. If the project genuinely intends to redefine the layers, that is a decision to make in `INVARIANTS.md` first, with the evidence library moved somewhere explicit — not by quietly redefining a safety invariant inside a test docstring.

---

## 3. Elevated from Round 1 — fail-open safety semantics

### P0-11 — Quality gates abstain only when *every* modality fails

**Location:** [`signal_quality.py:129`](extraction/signal_quality.py), consumed at [`analysis_service.py:227`](server/services/analysis_service.py)

```python
all_failed = (not audio_valid) and (not video_valid)
...
if quality_report.all_modalities_failed:   # only then does the service abstain
```

`evaluation_protocol.md` §4.4: *"The system **must abstain with 100% probability** if capture quality breaches deterministic bounds"* — followed by three independently sufficient conditions.

**Verified empirically:**

```
pose_confidence = 0.10 (severe occlusion), clean audio
  -> breaches = ['Mean pose tracking confidence (0.10) below 0.40 threshold.']
  -> all_modalities_failed = False  ->  service does NOT abstain

optical flow = 200 px/s (>85 limit), clean audio
  -> all_modalities_failed = False  ->  service does NOT abstain
```

A breach is detected, recorded in `breaches`, surfaced in L1 — and then ignored. The clip proceeds to full behavioral matching on kinematics the system has just certified as untrustworthy. `AND` was used where the protocol specifies `OR`.

Two supporting fail-open defaults compound it: `evaluate_signal_quality(mean_pose_confidence: float = 1.0)` at [line 76](extraction/signal_quality.py), and `mean_pose_conf = ... if pose_confidences else 1.0` at [`analysis_service.py:206`](server/services/analysis_service.py). A missing measurement defaults to perfect confidence. Fail-closed requires `0.0`.

**Required:** abstain if `breaches` is non-empty. Change both defaults to `0.0`. Add a test per gate asserting that a single breach forces abstention.

---

## 4. Remaining P1 findings

### P1-33 — `audio_snr_db` does not compute SNR

**Location:** [`signal_quality.py:41-63`](extraction/signal_quality.py)

```python
snr_db = max(0.0, -10.0 * np.log10(max(flatness, 1e-6)))
```

This is negative log spectral flatness — a **tonality** measure, not a ratio of signal power to noise power. The docstring acknowledges the mapping is a heuristic approximation. Two predictable failure modes:

- A strong tonal interferer (appliance hum, TV) in a room where the child is barely audible yields *low* flatness → high reported "SNR" → the gate passes on unusable audio.
- A breathy, whispered, or fricative vocalization — broadband by nature, and clinically among the most interesting signals for this project — yields *high* flatness → low reported "SNR" → the system abstains on exactly the material it exists to capture.

This is the P1-21 pattern recurring: an engineering term applied to a quantity that does not carry it, with a threshold (12 dB) whose meaning comes from the real definition. Here it gates abstention.

**Required:** estimate SNR properly — e.g. noise floor from the lowest-energy percentile of frames versus active-speech frame energy — or rename the field `spectral_tonality_index`, derive its threshold empirically from Child N's recordings, and amend `evaluation_protocol.md` §4.4 to describe what is actually measured.

### P1-34 — `captured_at` is still a hardcoded literal

**Location:** [`analysis_service.py:314`](server/services/analysis_service.py)

```python
"captured_at": "2026-09-14T00:00:00Z",
```

Called out in Round 1 P0-1 and listed verbatim in the acceptance criteria ("no hardcoded timestamp"). Every episode in the database claims the same capture time, which silently breaks the chronological sort in `TemporalSplitter.__init__` and therefore every forward-chaining split built on it — the leakage control that `evaluation_protocol.md` §2.2 calls mandatory.

Note that `FinalizeRequest.metadata_json` already exists and is discarded. That is the natural carrier for real capture time; failing that, read it from the container via ffprobe.

### P1-35 — Layer 3 suggestions are still hardcoded, not grounded in retrieved history

**Location:** [`analysis_service.py:267-274`](server/services/analysis_service.py)

```python
suggested = (
    ["quiet_refuge", "sensory_break"]
    if "quiet_refuge" in CONTROLLED_ACTIONS
    else ["open_observation"]
)
```

The condition is a tautology — `"quiet_refuge"` is a literal member of `CONTROLLED_ACTIONS` — so this always returns the same two actions regardless of the clip, the child, the antecedent, or `retrieved_candidates`, which is in scope and unused. The distress branch is likewise a fixed triple.

`README.md` promises *"low-risk co-regulatory ideas grounded in past successes"*; `INVARIANTS.md` Invariant 8 requires suggestions grounded in L2 precedent. A constant list presented as a suggestion is a fabricated recommendation, and it is the field that reaches the caregiver most directly.

**Required:** derive suggestions from `retrieved_candidates`, ranked by observed settling outcome, with the supporting episode ids attached. When retrieval abstains, return `["open_observation"]` — which the code already does correctly in the quality-gate branch.

### P1-36 — `ffmpeg` is an undeclared hard dependency

`extraction/demux.py` shells out to `ffmpeg` for every analysis. It appears in no dependency manifest, no CI setup step, no README prerequisite, and no `SPECS.md` stack section. If absent, `subprocess.run` raises `FileNotFoundError`, which is an `OSError` — **not** caught by the `except (subprocess.SubprocessError, sf.SoundFileError)` handler at [`demux.py:78`](extraction/demux.py) — so it escapes as an unclassified error rather than a `MediaDecodeError`. Add it to CI (both jobs), to README prerequisites, to `SPECS.md` §1, and catch `OSError` in the handler.

### P1-37 — Test-only media generator ships in a runtime module

`create_synthetic_mp4` ([`demux.py:111`](extraction/demux.py)) is documented as "used exclusively by test suites" but lives in the production extraction package, where it is importable by production code and counted in that module's line budget. Move it to `tests/fixtures/`.

### P1-38 — Embedding cache is in-memory, unbounded, and lost on restart

`AnalysisService._clip_embeddings` ([`analysis_service.py:63`](server/services/analysis_service.py)) grows without eviction for the process lifetime, and is the sole source for `get_clip_embedding`. A caregiver confirming an outcome after any daemon restart gets `indexed_in_vector_store: false` with no explanation and no retry path — silently losing the verified record that Invariant 9 depends on. Persist the embedding with the episode row, or re-derive it on confirmation.

### P1-39 — `Accept-Ranges: bytes` advertised without range support

[`episodes.py`](server/routes/episodes.py) sets the header and always returns the full body, ignoring any `Range` request. Either implement `206 Partial Content` or drop the header.

---

## 5. Gate status

| Gate | Scope | Status |
| :--- | :--- | :--- |
| **A** — stop emitting fabricated values | Remove synthetic/mock fallbacks; gate embeddings; INV-1 + INV-6 verifiers; green CI | **Incomplete.** Fallbacks removed and embeddings gated (good). INV-1 verifier landed; INV-6 verifier asserts the wrong contract (P0-10). Nothing pushed; CI has still never run on the implementation. `captured_at` literal remains (P1-34). |
| **B** — make the sensory path real | MediaPipe; demux; quality gates; standardizer; acoustic measures; L1 aggregation | **Mostly delivered.** MediaPipe and demux are solid. Quality gates exist but are fail-open (P0-11). **Feature standardizer not delivered.** Acoustic measures materially improved but still mislabeled (P1-22, P1-33). |
| **C** — close the learning loop | Splitter; seeds; clipping; training; checkpoints; outcome endpoint; calibration | **Structurally present, scientifically empty.** Splitter, seeding, gradient clipping, checkpoint save/load, and the outcome endpoint are real and good. But training runs on random noise, no evaluation metric is computed anywhere, and the registry is populated with fabricated numbers (P0-9). τ_abstain calibration not delivered. |
| **D** — restore the documented contract | L1–L4; Parent/Therapist rendering; SSE events; vocabulary reconciliation | **One item of four.** Vocabulary reconciled and CHECK-parity test added — exemplary. L3/L4, rendering, and the two SSE event types untouched. |
| **E** — security and lifecycle parity | SQLCipher; Keychain; CA/HMAC/nonce; promote/rollback; lineage; retention; B1–B4 | **Partially started, one regression.** Real CA and token expiry are good. PIN endpoint is a live auth bypass (P0-8). SQLCipher, Keychain, HMAC/nonce, mTLS enforcement, promote/rollback, retention, and B1–B4 all outstanding. |

---

## 6. Acceptance criteria re-check

| Criterion | Round 1 | Now |
| :--- | :-- | :-- |
| Placeholder sweep clean outside tests | ✗ | **Nearly.** Only `MockKeychainProvider` (Gate E, acknowledged), `create_synthetic_mp4` (P1-37), and `build_synthetic_triplets` in the trainer (P0-9). Large improvement. |
| No hardcoded timestamp / version / PIN / token / cert / media payload | ✗ | ✗ — `captured_at` (P1-34); `"git_v1.0.0"`, `0.85`, `0.92`, `0.05`, `"passed"` in the trainer (P0-9) |
| Every emitted value traces to a measurement, checkpoint, or caregiver input | ✗ | ✗ — L3 suggestions (P1-35), `model_checkpoints` metrics (P0-9) |
| `/analyze` fails cleanly on undecodable clip, no episode row | ✗ | **✓** |
| Breach of any quality gate forces abstention | ✗ | ✗ — fail-open (P0-11), verified |
| `test_shapes.py` asserts vocabularies against live `CHECK` constraints | ✗ | **✓** |
| `test_output_schema.py` asserts L1–L4 per Invariant 6 | ✗ | ✗ — asserts the drifted schema (P0-10) |
| CI green on `macos-14` against the implementation commit | ✗ | ✗ — nothing pushed; ubuntu job still cannot install MLX; ffmpeg absent |
| Coverage gate at §8.2 thresholds incl. branch coverage for `models/` | ✗ | ✗ |
| INV-1, INV-4, INV-6, INV-7 verifiers exist and run in CI | ✗ | **Partial** — all four exist and run; INV-6 asserts the wrong contract. INV-3 and INV-9 verifiers exist but never execute. |
| End-to-end test: real media → vault → decrypt → extract → project → screen → retrieve → confirm → persist, with matcher leaving `collecting` | ✗ | ✗ — no test drives the matcher out of `collecting`, because no trained checkpoint exists |

**3 of 11 satisfied**, up from 0.

---

## 7. Required before the next review

**Immediate — security and governance (do not defer)**

1. **P0-8** — remove `GET /api/v1/auth/pin` from the network API. Regression test that unauthenticated pairing is impossible.
2. **P0-9** — remove synthetic training and all fabricated metric defaults; implement `training/evaluate_candidate.py` (real MRR / P@3 / ECE / coverage / safety-error count); block `model_checkpoints` writes until it exists.
3. **P0-11** — abstain on any non-empty `breaches`; change both `1.0` confidence defaults to `0.0`; one test per gate.
4. **P0-10** — rewrite the INV-6 verifier against `INVARIANTS.md` as written, then implement L3 and L4 to satisfy it.

**Next — parity**

5. **P1-34** `captured_at` from container or upload metadata; **P1-35** ground L3 suggestions in `retrieved_candidates`.
6. **P1-24** feature standardizer, fitted train-only, versioned with the checkpoint (carried from Gate B).
7. **P1-33** real SNR or an honest rename; **P1-22** cycle-synchronous perturbation or an honest rename; **P1-21** validate CPP against a reference on a shared fixture.
8. Load a checkpoint in `get_analysis_service()`; make `verify_frozen_weights.py` and `verify_lineage.py` actually execute; correct the INV-9 command path in `INVARIANTS.md:135`; add `test_insight_delivery.py`.
9. **P1-36** declare `ffmpeg`; catch `OSError` in the demux handler.
10. **P1-28** mark `mlx` platform-conditional, add ffmpeg to both CI jobs, **push**, and get a green run.
11. **P1-30** coverage gate including branch coverage for `models/`.

**Carried, Gate E**

12. SQLCipher; real Keychain provider; mTLS enforcement + HMAC/timestamp/nonce; promote/rollback; retention purge; B1–B4 baselines.

**Standing rule, restated**

> Any documented capability not implemented in the commit that claims its phase must be removed from `INVARIANTS.md` / `docs/SPECS.md` or relabeled *planned*, in that same commit. **A test that asserts a contract different from the one the invariant states is a documentation change made in the wrong file** — it must be made in `INVARIANTS.md`, deliberately and visibly, or not at all.

---

## 8. Closing judgment

The sensory path went from fiction to fact in one round. Real decode, real pose, real confidence propagation, an honest refusal to emit an embedding without a checkpoint, a genuine X.509 CA, an AST import scanner, and a CHECK-constraint parity test that closes the exact blind spot Round 1 identified. Several of these — particularly gating `metric_embedding` on `is_trained` and fail-closing the vector-store write — are better than what was asked for. That deserves saying plainly.

What has not changed is the failure mode. Round 1's core observation was that where the implementation could not reach the specification, it filled the gap with a silent substitute. That pattern persisted through the remediation and relocated to more consequential places: a PIN endpoint that dissolves the authentication boundary, a checkpoint registry pre-seeded with passing grades no one computed, a quality gate that detects breaches and proceeds anyway, and a verifier that resolves a documented contradiction by asserting the code is right. In each case the substitute is locally reasonable and globally false — and in each case it now sits inside a control designed to protect a child's data or a caregiver's trust in a model's output.

The work to close this is smaller than the work already done. Four fixes clear the immediate list, and two of them are deletions. But Phases 0–2 should not be signed off, and Phase 4 should not begin, until the promotion registry cannot be populated with numbers nobody measured and the daemon cannot hand its own credentials to an unauthenticated caller.

---

*Audit performed against commit `9818e05`. All findings verified by direct file inspection, repository-wide search, execution of the test suite (57 passed), live exercise of the authentication flow and signal-quality gates against the running application, and inspection of GitHub Actions run history.*
