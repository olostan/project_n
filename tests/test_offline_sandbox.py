"""
Project N: Invariant 1 Hermetic Offline Sandbox Verification.
Enforces zero external network connections during sensory extraction,
inference, and episode storage.
"""

import socket

import pytest
from starlette.testclient import TestClient

from extraction.demux import create_synthetic_mp4
from server.deps import get_analysis_service, get_or_create_pairing_pin, register_token
from server.main import app


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocks any outbound socket connection attempts outside localhost."""
    orig_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: tuple[str, int] | str) -> None:
        if isinstance(address, tuple):
            host, _ = address[0], address[1]
            if host in ("localhost", "127.0.0.1", "::1", "testserver"):
                return orig_connect(self, address)
        raise ConnectionRefusedError(
            f"INVARIANT 1 BREACH: Unauthorized network attempt to {address} in offline mode."
        )

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


def test_offline_sandbox_zero_exfiltration() -> None:
    """Verifies that full local backend operations function with zero external network."""
    client = TestClient(app)

    # 1. Pairing verification
    pin = get_or_create_pairing_pin()
    assert len(pin) == 6

    # Register authenticated test token
    test_token = "test_offline_bearer_token"
    register_token(test_token, 9999999999.0)
    headers = {"Authorization": f"Bearer {test_token}"}

    # 2. Resumable upload of real valid MP4 bytes
    real_mp4 = create_synthetic_mp4(duration_sec=1.0)
    import hashlib

    mp4_hash = hashlib.sha256(real_mp4).hexdigest()

    init_res = client.post(
        "/api/v1/clips/upload/init",
        headers=headers,
        json={
            "file_name": "offline_test.mp4",
            "file_size": len(real_mp4),
            "sha256": mp4_hash,
            "total_chunks": 1,
        },
    )
    assert init_res.status_code == 200
    upload_id = init_res.json()["upload_id"]

    # Upload chunk
    chunk_res = client.put(
        f"/api/v1/clips/upload/{upload_id}/chunk/0",
        headers={"Authorization": f"Bearer {test_token}", "X-Chunk-SHA256": mp4_hash},
        content=real_mp4,
    )
    assert chunk_res.status_code == 200

    # Finalize
    finalize_res = client.post(
        f"/api/v1/clips/upload/{upload_id}/finalize",
        headers=headers,
        json={"metadata_json": {"notes": "offline verification"}},
    )
    assert finalize_res.status_code == 200
    clip_id = finalize_res.json()["clip_id"]

    # Trigger analysis synchronously via service directly
    import numpy as np

    analysis_service = get_analysis_service()
    res = analysis_service.analyze_sensory_clip(
        clip_id=clip_id,
        audio_pcm=np.zeros(240000, dtype=np.float32),
        video_frames=[np.zeros((180, 320, 3), dtype=np.uint8) for _ in range(30)],
    )

    assert "L1_measured" in res
    assert "L2_historical" in res
    assert "L3_context" in res
    assert "L4_evidence" in res
    assert "safety_triage" in res
    assert "dyadic_suggestions" in res
    assert "parent_view_text" in res


def test_ast_forbidden_imports_invariant() -> None:
    """Verifies that no production Python module imports external cloud AI SDKs or PyTorch."""
    from pathlib import Path

    from tests.test_imports import verify_no_forbidden_imports

    repo_root = Path(__file__).resolve().parent.parent
    violations = verify_no_forbidden_imports(repo_root)
    assert not violations, f"Forbidden imports detected: {violations}"
