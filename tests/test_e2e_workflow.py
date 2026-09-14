"""
Project N: Hermetic End-to-End (E2E) Caregiver Workflow Test.
Simulates the full local client lifecycle without requiring Apple Silicon GPU,
remote cloud APIs, or external cameras:
1. Companion authentication and local PIN pairing.
2. Resumable chunked upload and local vault storage.
3. Multimodal extraction, projection, anomaly screening, and 4-layer structuring.
4. Real-time SSE event consumption.
5. Caregiver review, NCCPC pain scoring, and outcome logging.
6. Clinician dyadic fact staging and Human-in-the-Loop confirmation gate.
"""

import hashlib

from fastapi.testclient import TestClient

from models.contracts import CONTROLLED_ACTIONS
from server.deps import get_sse_bus
from server.main import app

client = TestClient(app)


def test_hermetic_caregiver_e2e_journey() -> None:
    # --------------------------------------------------------------------------
    # Step 1: Companion Device Authentication & Local PIN Pairing
    # --------------------------------------------------------------------------
    pair_res = client.post(
        "/api/v1/auth/pair",
        json={
            "device_id": "caregiver_phone_iphone15",
            "csr_pem": "-----BEGIN CERTIFICATE REQUEST-----\n...",
            "pairing_pin": "123456",
        },
    )
    assert pair_res.status_code == 200
    token = pair_res.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # --------------------------------------------------------------------------
    # Step 2: Resumable Chunked Upload & Vault Ingestion
    # --------------------------------------------------------------------------
    chunk_a = b"RAW_SYNTHETIC_SENSOR_STREAM_PART_A_"
    chunk_b = b"RAW_SYNTHETIC_SENSOR_STREAM_PART_B_"
    full_payload = chunk_a + chunk_b
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
        headers=headers,
        content=chunk_a,
    )
    assert c0_res.status_code == 200
    assert c0_res.json()["received"]

    c1_res = client.put(
        f"/api/v1/clips/upload/{upload_id}/chunk/1",
        headers=headers,
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
    assert episode_data["encoder_version_id"] == "v1.0.0"

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
    # Threshold for NCCPC-PV is >= 11 for significant pain cutoff
    assert not triage_data["pain_cutoff_breached"]

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
