"""
Project N: Integration Tests for Tier 4 Server, REST APIs, and SSE Stream.
Verifies FastAPI endpoints using TestClient: PIN pairing, chunked upload,
episodes querying, media streaming from vault, outcome confirmation, NCCPC pain scoring, and facts staging.
"""

import hashlib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from extraction.demux import create_synthetic_mp4
from server.ca import generate_test_csr
from server.deps import (
    get_episode_repo,
    get_or_create_pairing_pin,
    get_vault,
    reset_pairing_state,
)
from server.main import app

client = TestClient(app)


def _get_authenticated_headers() -> dict[str, str]:
    """Helper to pair a device and obtain authenticated Bearer headers."""
    reset_pairing_state()
    csr_pem, _ = generate_test_csr("test_paired_device")
    pin = get_or_create_pairing_pin()
    pair_res = client.post(
        "/api/v1/auth/pair",
        json={
            "device_id": "test_device_01",
            "csr_pem": csr_pem,
            "pairing_pin": pin,
        },
    )
    assert pair_res.status_code == 200
    token = pair_res.json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_health_check() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["engine"] == "Apple MLX"
    assert "active_metal_memory_gb" in data


def test_auth_pin_endpoint_does_not_exist() -> None:
    """Security Invariant: GET /api/v1/auth/pin must not exist on public HTTP interface."""
    res = client.get("/api/v1/auth/pin")
    assert res.status_code == 404


def test_auth_pin_pairing() -> None:
    reset_pairing_state()
    csr_pem, _ = generate_test_csr("device_pairing_test")
    pin = get_or_create_pairing_pin()

    # Verify pin was written to ~/.project_n/pairing.pin
    pin_file = Path.home() / ".project_n" / "pairing.pin"
    assert pin_file.exists()
    assert pin_file.read_text().strip() == pin

    # 1. Invalid PIN fails (attempt 1)
    fail_res = client.post(
        "/api/v1/auth/pair",
        json={"device_id": "test_phone_01", "csr_pem": csr_pem, "pairing_pin": "00000000"},
    )
    assert fail_res.status_code == 400

    # 2. Invalid CSR fails (attempt 2)
    bad_csr_res = client.post(
        "/api/v1/auth/pair",
        json={"device_id": "test_phone_01", "csr_pem": "not_a_csr", "pairing_pin": pin},
    )
    assert bad_csr_res.status_code == 400

    # 3. Valid PIN and CSR succeeds and yields token + X.509 cert
    success_res = client.post(
        "/api/v1/auth/pair",
        json={"device_id": "test_phone_01", "csr_pem": csr_pem, "pairing_pin": pin},
    )
    assert success_res.status_code == 200
    data = success_res.json()
    assert "token" in data
    assert data["token"].startswith("paired_")
    assert "BEGIN CERTIFICATE" in data["client_cert_pem"]


def test_auth_pin_rate_limiting_lockout() -> None:
    """Ensure after 5 failed pairing attempts, lockout returns 429."""
    reset_pairing_state()
    csr_pem, _ = generate_test_csr("device_lockout_test")
    get_or_create_pairing_pin()

    # 5 failed attempts
    for _ in range(5):
        res = client.post(
            "/api/v1/auth/pair",
            json={"device_id": "test_bad", "csr_pem": csr_pem, "pairing_pin": "999999"},
        )
        assert res.status_code == 400

    # 6th attempt is locked out with 429
    locked_res = client.post(
        "/api/v1/auth/pair",
        json={"device_id": "test_bad", "csr_pem": csr_pem, "pairing_pin": "999999"},
    )
    assert locked_res.status_code == 429
    assert "locked out" in locked_res.json()["detail"].lower()
    reset_pairing_state()


def test_auth_env_pin_test_mode_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """PROJECT_N_PAIRING_PIN is only respected when PROJECT_N_TEST_MODE == '1'."""
    reset_pairing_state()
    monkeypatch.setenv("PROJECT_N_PAIRING_PIN", "123456")
    monkeypatch.delenv("PROJECT_N_TEST_MODE", raising=False)

    # Without test mode, env pin should be ignored
    pin1 = get_or_create_pairing_pin()
    assert pin1 != "123456"

    # With test mode, env pin is honored
    reset_pairing_state()
    monkeypatch.setenv("PROJECT_N_TEST_MODE", "1")
    pin2 = get_or_create_pairing_pin()
    assert pin2 == "123456"
    reset_pairing_state()


def test_chunked_upload_and_finalize() -> None:
    headers = _get_authenticated_headers()

    # Prepare real payload: synthetic MP4
    full_payload = create_synthetic_mp4(duration_sec=1.0)
    payload_hash = hashlib.sha256(full_payload).hexdigest()
    mid = len(full_payload) // 2
    chunk1 = full_payload[:mid]
    chunk2 = full_payload[mid:]

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
        headers={**headers, "X-Chunk-SHA256": hashlib.sha256(chunk1).hexdigest()},
        content=chunk1,
    )
    assert c1_res.status_code == 200

    c2_res = client.put(
        f"/api/v1/clips/upload/{upload_id}/chunk/1",
        headers={**headers, "X-Chunk-SHA256": hashlib.sha256(chunk2).hexdigest()},
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
    headers = _get_authenticated_headers()
    repo = get_episode_repo()
    vault = get_vault()

    ep_id = "ep_server_test_01"
    repo.insert_episode(
        {
            "id": ep_id,
            "vault_uri": f"vault://{ep_id}.enc",
            "encoder_version_id": "v1.0.0",
            "captured_at": "2026-09-14T10:00:00Z",
            "duration_ms": 30000,
            "windows_count": 6,
            "antecedent_id": "mealtime",
            "action_offered": "hydration_water",
        }
    )

    # Store real media in vault for this episode
    real_media = create_synthetic_mp4(duration_sec=1.0)
    vault.store_clip(ep_id, real_media)

    # 1. Query episode via API
    get_res = client.get(f"/api/v1/episodes/{ep_id}", headers=headers)
    assert get_res.status_code == 200
    assert get_res.json()["id"] == ep_id

    # 2. List episodes
    list_res = client.get("/api/v1/episodes", headers=headers)
    assert list_res.status_code == 200
    assert list_res.json()["total"] >= 1

    # 3. Stream real media from vault
    media_res = client.get(f"/api/v1/episodes/{ep_id}/media", headers=headers)
    assert media_res.status_code == 200
    assert media_res.headers["content-type"] == "video/mp4"
    assert len(media_res.content) == len(real_media)

    # 4. Caregiver Outcome Confirmation
    outcome_res = client.post(
        f"/api/v1/episodes/{ep_id}/outcome",
        headers=headers,
        json={
            "action_performed": "hydration_water",
            "outcome_state": "settled_immediately",
            "caregiver_decision": "accepted",
            "settled_within_sec": 45,
            "child_response": "vocal_signal",
            "response_channel": "vocal",
            "response_independence": "independent",
        },
    )
    assert outcome_res.status_code == 200
    assert outcome_res.json()["status"] == "confirmed"

    # 5. Submit NCCPC pain checklist
    high_pain_scores: dict[str, Any] = {
        "crying": 3,
        "screaming_yelling": 3,
        "stiff_spastic_rigid_tense": 3,
        "sharp_breath_gasping": 3,
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
    headers = _get_authenticated_headers()

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


def test_local_session_token_issuance() -> None:
    """Verifies that loopback clients can obtain a local session token without CSR."""
    res = client.post("/api/v1/auth/local-token")
    assert res.status_code == 200
    data = res.json()
    assert "token" in data
    assert data["token"].startswith("local_")
    assert "expires_at" in data

    # Verify that the issued token is accepted by protected endpoints
    headers = {"Authorization": f"Bearer {data['token']}"}
    init_res = client.post(
        "/api/v1/clips/upload/init",
        headers=headers,
        json={
            "file_name": "local_test.mp4",
            "file_size": 1000,
            "sha256": "0" * 64,
            "total_chunks": 1,
        },
    )
    assert init_res.status_code == 200
    assert "upload_id" in init_res.json()
