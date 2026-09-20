"""
Project N: Integration tests for Web UI Dashboard and Model Routes.
Verifies SPA index serving, asset delivery, client-side routing fallback, and models routes.
"""

from fastapi.testclient import TestClient

from server.main import app

client = TestClient(app)


def test_spa_root_serves_html() -> None:
    """Verifies that GET / serves the compiled React SPA index.html."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert '<div id="root"></div>' in response.text


def test_spa_client_side_route_fallback() -> None:
    """Verifies that non-API paths (e.g., /diary, /inspector) fall back to index.html for SPA routing."""
    response = client.get("/diary")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert '<div id="root"></div>' in response.text


def test_api_route_not_found_returns_404() -> None:
    """Verifies that missing /api/ routes return 404 and are NOT intercepted by SPA fallback."""
    response = client.get("/api/v1/non_existent_endpoint")
    assert response.status_code == 404
    assert response.json()["detail"] == "API endpoint not found."


def test_models_checkpoints_route() -> None:
    """Verifies GET /api/v1/models/checkpoints returns checkpoint registry."""
    response = client.get("/api/v1/models/checkpoints")
    assert response.status_code == 200
    data = response.json()
    assert "checkpoints" in data
    assert "total" in data
