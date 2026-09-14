"""
Project N: Integration Tests for Tier 4 Server, REST APIs, and SSE Stream.
Verifies FastAPI endpoints using TestClient: PIN pairing, chunked upload,
episodes querying, NCCPC pain checklist scoring, facts staging, and health check.
"""

import hashlib

from fastapi.testclient import TestClient

from server.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["engine"] == "Apple MLX"
    assert "active_metal_memory_gb" in data


def test_auth_pin_pairing() -> None:
    # 1. Invalid PIN fails
    fail_res = client.post(
        "/api/v1/auth/pair",
        json={"device_id": "test_phone_01", "csr_pem": "test_csr", "pairing_pin": "999999"},
    )
    assert fail_res.status_code == 400

    # 2. Valid PIN succeeds and yields token
    success_res = client.post(
        "/api/v1/auth/pair",
        json={"device_id": "test_phone_01", "csr_pem": "test_csr", "pairing_pin": "123456"},
    )
    assert success_res.status_code == 200
    data = success_res.json()
    assert "token" in data
    assert data["token"].startswith("paired_")


def test_chunked_upload_and_finalize() -> None:
    headers = {"Authorization": "Bearer mock_local_paired_token"}

    # Prepare payload: 2 chunks
    chunk1 = b"PART_1_RAW_VIDEO_CONTENT_"
    chunk2 = b"PART_2_RAW_VIDEO_CONTENT_"
    full_payload = chunk1 + chunk2
    payload_hash = hashlib.sha256(full_payload).hexdigest()

    # 1. Init upload
    init_res = client.post(
        "/api/v1/clips/upload/init",
        headers=headers,
        json={
            "file_name": "clip_test.mp4",
            "file_size": len(full_payload),
            "sha256": payload_hash,
            "total_chunks": 2,
        },
    )
    assert init_res.status_code == 200
    upload_id = init_res.json()["upload_id"]

    # 2. Upload chunks
    c1_res = client.put(
        f"/api/v1/clips/upload/{upload_id}/chunk/0",
        headers=headers,
        content=chunk1,
    )
    assert c1_res.status_code == 200

    c2_res = client.put(
        f"/api/v1/clips/upload/{upload_id}/chunk/1",
        headers=headers,
        content=chunk2,
    )
    assert c2_res.status_code == 200

    # 3. Finalize upload
    final_res = client.post(
        f"/api/v1/clips/upload/{upload_id}/finalize",
        headers=headers,
        json={"metadata_json": {"setting": "living_room"}},
    )
    assert final_res.status_code == 200
    final_data = final_res.json()
    assert final_data["status"] == "stored"
    assert final_data["sha256"] == payload_hash
    assert "clip_id" in final_data


def test_episodes_and_nccpc_scoring() -> None:
    headers = {"Authorization": "Bearer mock_local_paired_token"}

    # 1. First insert an episode into repository directly
    from server.deps import get_episode_repo

    repo = get_episode_repo()
    ep_id = "ep_server_test_01"
    repo.insert_episode(
        {
            "id": ep_id,
            "vault_uri": "vault://clips/test.enc",
            "encoder_version_id": "v1.0.0",
            "captured_at": "2026-09-14T10:00:00Z",
            "duration_ms": 30000,
            "windows_count": 6,
            "antecedent_id": "mealtime",
            "action_offered": "hydration_water",
        }
    )

    # 2. Query episode via API
    get_res = client.get(f"/api/v1/episodes/{ep_id}", headers=headers)
    assert get_res.status_code == 200
    assert get_res.json()["id"] == ep_id

    # 3. List episodes
    list_res = client.get("/api/v1/episodes", headers=headers)
    assert list_res.status_code == 200
    assert list_res.json()["total"] >= 1

    # 4. Stream media
    media_res = client.get(f"/api/v1/episodes/{ep_id}/media", headers=headers)
    assert media_res.status_code == 200
    assert media_res.headers["content-type"] == "video/mp4"

    # 5. Submit NCCPC pain checklist
    # Breau et al. (2002): NCCPC-PV cut-off >= 11 indicates pain
    high_pain_scores = {
        "crying": 3,
        "screaming_yelling": 3,
        "stiff_spastic_rigid_tense": 3,
        "sharp_breath_gasping": 3,  # sum = 12 >= 11
    }
    nccpc_res = client.post(
        f"/api/v1/episodes/{ep_id}/nccpc",
        headers=headers,
        json={
            "instrument": "nccpc_pv",
            "scores": high_pain_scores,
            "duration_min": 10,
        },
    )
    assert nccpc_res.status_code == 200
    nccpc_data = nccpc_res.json()
    assert nccpc_data["total_score"] == 12
    assert nccpc_data["pain_cutoff_breached"] is True
    assert nccpc_data["escalation_required"] is True
    assert "MEDICAL ESCALATION REQUIRED" in nccpc_data["guidance"]


def test_facts_staging_and_confirmation() -> None:
    headers = {"Authorization": "Bearer mock_local_paired_token"}

    # 1. Stage a clinician technique
    post_res = client.post(
        "/api/v1/facts",
        headers=headers,
        json={
            "category": "therapist_technique",
            "fact_title": "Deep pressure weighted blanket",
            "description": "Offering 5 lb weighted lap pad during seated transitions",
            "source_type": "ot_session",
            "clinician_role": "OT",
            "clinician_id": "clinician_01",
        },
    )
    assert post_res.status_code == 201
    fact_id = post_res.json()["id"]
    assert post_res.json()["confirmed_by_caregiver"] == 0

    # 2. Confirmed list should not contain it yet
    list_res = client.get("/api/v1/facts?confirmed_only=true", headers=headers)
    assert not any(f["id"] == fact_id for f in list_res.json()["facts"])

    # 3. Confirm the technique via Confirmation Gate
    confirm_res = client.post(
        f"/api/v1/facts/{fact_id}/confirm",
        headers=headers,
        json={"confirmed": True},
    )
    assert confirm_res.status_code == 200
    assert confirm_res.json()["confirmed_by_caregiver"] is True

    # 4. Now it should appear in confirmed list
    list_after = client.get("/api/v1/facts?confirmed_only=true", headers=headers)
    assert any(f["id"] == fact_id for f in list_after.json()["facts"])
