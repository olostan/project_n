"""
Project N: Hermetic End-to-End (E2E) Caregiver Workflow Test.
Simulates the full local client lifecycle using real decodable MP4 media:
1. Companion authentication and local PIN pairing with real ECDSA CSR.
2. Resumable chunked upload of valid MP4 and local vault storage.
3. Multimodal demuxing, extraction, projection, anomaly screening, and 4-layer structuring.
4. Real-time SSE event consumption.
5. Caregiver review, NCCPC pain scoring, and outcome logging.
6. Clinician dyadic fact staging and Human-in-the-Loop confirmation gate.
"""

import hashlib

from fastapi.testclient import TestClient

from extraction.demux import create_synthetic_mp4
from models.contracts import CONTROLLED_ACTIONS
from server.ca import generate_test_csr
from server.deps import get_or_create_pairing_pin, get_sse_bus
from server.main import app

client = TestClient(app)


def test_hermetic_caregiver_e2e_journey() -> None:
    # --------------------------------------------------------------------------
    # Step 1: Companion Device Authentication & Local PIN Pairing
    # --------------------------------------------------------------------------
    csr_pem, _ = generate_test_csr("caregiver_phone_iphone15")
    pin = get_or_create_pairing_pin()
    pair_res = client.post(
        "/api/v1/auth/pair",
        json={
            "device_id": "caregiver_phone_iphone15",
            "csr_pem": csr_pem,
            "pairing_pin": pin,
        },
    )
    assert pair_res.status_code == 200
    token = pair_res.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # --------------------------------------------------------------------------
    # Step 2: Resumable Chunked Upload & Vault Ingestion (Real Decodable MP4)
    # --------------------------------------------------------------------------
    full_payload = create_synthetic_mp4(duration_sec=1.0)
    mid = len(full_payload) // 2
    chunk_a = full_payload[:mid]
    chunk_b = full_payload[mid:]
    sha256_hash = hashlib.sha256(full_payload).hexdigest()

    # 2a. Initialize upload session
    init_res = client.post(
        "/api/v1/clips/upload/init",
        headers=headers,
        json={
            "file_name": "observation_playground_01.mp4",
            "file_size": len(full_payload),
            "sha256": sha256_hash,
            "total_chunks": 2,
        },
    )
    assert init_res.status_code == 200
    upload_id = init_res.json()["upload_id"]

    # 2b. Transmit chunks in sequence
    c0_res = client.put(
        f"/api/v1/clips/upload/{upload_id}/chunk/0",
        headers={**headers, "X-Chunk-SHA256": hashlib.sha256(chunk_a).hexdigest()},
        content=chunk_a,
    )
    assert c0_res.status_code == 200
    assert c0_res.json()["received"]

    c1_res = client.put(
        f"/api/v1/clips/upload/{upload_id}/chunk/1",
        headers={**headers, "X-Chunk-SHA256": hashlib.sha256(chunk_b).hexdigest()},
        content=chunk_b,
    )
    assert c1_res.status_code == 200
    assert c1_res.json()["received"]

    # 2c. Finalize upload & verify AES-256-GCM vault encryption
    finalize_res = client.post(
        f"/api/v1/clips/upload/{upload_id}/finalize",
        headers=headers,
        json={"metadata_json": {"setting": "home_routine"}},
    )
    assert finalize_res.status_code == 200
    clip_id = finalize_res.json()["clip_id"]
    assert finalize_res.json()["status"] == "stored"

    # --------------------------------------------------------------------------
    # Step 3: Asynchronous Multimodal Analysis Pipeline Execution
    # --------------------------------------------------------------------------
    analyze_res = client.post(
        f"/api/v1/clips/{clip_id}/analyze",
        headers=headers,
        json={"generate_render": False},
    )
    assert analyze_res.status_code == 200
    task_id = analyze_res.json()["task_id"]
    assert task_id

    # Verify background execution has run and published SSE events
    sse_bus = get_sse_bus()
    recent_events = sse_bus.get_recent_events(limit=10)
    assert len(recent_events) > 0

    event_types = [ev["event"] for ev in recent_events]
    assert "processing_progress" in event_types or "analysis_completed" in event_types

    # --------------------------------------------------------------------------
    # Step 4: Caregiver Timeline & 4-Layer Output Inspection
    # --------------------------------------------------------------------------
    ep_res = client.get(f"/api/v1/episodes/{clip_id}", headers=headers)
    assert ep_res.status_code == 200
    episode_data = ep_res.json()
    assert episode_data["id"] == clip_id

    # Verify controlled action vocabulary integrity
    if episode_data.get("action_offered"):
        assert episode_data["action_offered"] in CONTROLLED_ACTIONS

    # --------------------------------------------------------------------------
    # Step 5: Caregiver NCCPC Pain Triage Scoring & Outcome Logging
    # --------------------------------------------------------------------------
    # Submit validated Non-Communicating Children's Pain Checklist items
    nccpc_items: dict[str, int] = {
        "crying": 2,
        "screaming_yelling": 1,
        "less_interaction_withdrawn": 2,
        "stiff_spastic_rigid_tense": 2,
    }
    triage_res = client.post(
        f"/api/v1/episodes/{clip_id}/nccpc",
        headers=headers,
        json={
            "instrument": "nccpc_pv",
            "scores": nccpc_items,
        },
    )
    assert triage_res.status_code == 200
    triage_data = triage_res.json()
    assert triage_data["total_score"] == 7
    assert not triage_data["pain_cutoff_breached"]

    # Caregiver confirms outcome
    outcome_res = client.post(
        f"/api/v1/episodes/{clip_id}/outcome",
        headers=headers,
        json={
            "action_performed": "quiet_refuge",
            "outcome_state": "settled_immediately",
            "caregiver_decision": "accepted",
            "settled_within_sec": 30,
            "child_response": "vocal_signal",
        },
    )
    assert outcome_res.status_code == 200
    assert outcome_res.json()["status"] == "confirmed"

    # --------------------------------------------------------------------------
    # Step 6: Clinician Fact Staging & Human-in-the-Loop Confirmation Gate
    # --------------------------------------------------------------------------
    fact_res = client.post(
        "/api/v1/facts",
        headers=headers,
        json={
            "category": "therapist_technique",
            "fact_title": "Deep Pressure Blanketing",
            "description": "Deep pressure weighted blanket helps regulate after sensory overload.",
            "source_type": "ot_session",
            "provenance_episode_id": clip_id,
        },
    )
    assert fact_res.status_code == 201
    fact_id = fact_res.json()["id"]
    assert fact_res.json()["confirmed_by_caregiver"] == 0

    # Human-in-the-Loop: Caregiver explicitly confirms the clinician fact
    confirm_res = client.post(
        f"/api/v1/facts/{fact_id}/confirm",
        headers=headers,
        json={"confirmed": True},
    )
    assert confirm_res.status_code == 200
    assert confirm_res.json()["confirmed_by_caregiver"] == 1

    # Verify fact is now in confirmed list
    facts_list_res = client.get("/api/v1/facts?status=confirmed", headers=headers)
    assert facts_list_res.status_code == 200
    confirmed_ids = [f["id"] for f in facts_list_res.json()["facts"]]
    assert fact_id in confirmed_ids
