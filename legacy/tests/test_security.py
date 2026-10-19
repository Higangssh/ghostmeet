"""Local-only posture.

The backend serves meeting transcripts with no authentication, so the only thing
keeping them private is who can reach it. Binding every interface and echoing any
Origin put that on the network for anyone nearby.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import app as app_module


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "RECORDINGS_DIR", tmp_path)
    monkeypatch.setattr(app_module, "DB_PATH", tmp_path / "ghostmeet.db")
    with TestClient(app_module.app) as c:
        yield c


def test_default_bind_address_is_loopback(monkeypatch):
    """Regression: the default was 0.0.0.0, which exposed transcripts to the LAN."""
    monkeypatch.delenv("GHOSTMEET_HOST", raising=False)

    assert app_module.default_host() == "127.0.0.1"


def test_bind_address_can_still_be_overridden(monkeypatch):
    """Deliberate exposure stays possible, it just is not the default."""
    monkeypatch.setenv("GHOSTMEET_HOST", "0.0.0.0")

    assert app_module.default_host() == "0.0.0.0"


def test_the_extension_origin_is_allowed(client):
    origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"

    response = client.get("/api/health", headers={"Origin": origin})

    assert response.headers.get("access-control-allow-origin") == origin


def test_a_website_origin_is_not_allowed(client):
    """Regression: allow_origins=["*"] let any page a user visited read their meetings."""
    response = client.get(
        "/api/health", headers={"Origin": "https://evil.example"}
    )

    assert "access-control-allow-origin" not in response.headers


def test_preflight_from_a_website_origin_is_refused(client):
    response = client.options(
        "/api/sessions/x/summarize",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert "access-control-allow-origin" not in response.headers


def test_preflight_from_the_extension_is_accepted(client):
    origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"

    response = client.options(
        "/api/sessions/x/summarize",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
    )

    assert response.headers.get("access-control-allow-origin") == origin


def test_a_local_dashboard_origin_is_allowed(client):
    """Leaves room for a local web UI without reopening the door to the internet."""
    response = client.get(
        "/api/health", headers={"Origin": "http://127.0.0.1:8877"}
    )

    assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:8877"
