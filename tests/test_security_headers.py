"""Security response headers: always-on set + opt-in CSP."""

from __future__ import annotations

import pytest

from src.api.security_headers import DEFAULT_CSP, _build_headers


def test_always_on_headers_present(monkeypatch):
    monkeypatch.delenv("CSP_ENABLE", raising=False)
    monkeypatch.delenv("FRAME_DENY", raising=False)
    h = _build_headers()
    assert h["X-Content-Type-Options"] == "nosniff"
    assert h["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "camera=()" in h["Permissions-Policy"]
    assert "microphone=()" in h["Permissions-Policy"]
    # X-Frame-Options defaults to DENY.
    assert h["X-Frame-Options"] == "DENY"


def test_csp_off_by_default(monkeypatch):
    monkeypatch.delenv("CSP_ENABLE", raising=False)
    assert "Content-Security-Policy" not in _build_headers()


@pytest.mark.parametrize("flag", ["true", "TRUE", "1", "yes", "on"])
def test_csp_on_when_enabled(monkeypatch, flag):
    monkeypatch.setenv("CSP_ENABLE", flag)
    h = _build_headers()
    assert h["Content-Security-Policy"] == DEFAULT_CSP


@pytest.mark.parametrize("flag", ["false", "FALSE", "0", "no", "off", ""])
def test_csp_respects_falsy_flags(monkeypatch, flag):
    monkeypatch.setenv("CSP_ENABLE", flag)
    assert "Content-Security-Policy" not in _build_headers()


def test_csp_policy_override(monkeypatch):
    monkeypatch.setenv("CSP_ENABLE", "true")
    monkeypatch.setenv("CSP_POLICY", "default-src 'none'")
    assert _build_headers()["Content-Security-Policy"] == "default-src 'none'"


def test_frame_deny_can_be_disabled(monkeypatch):
    monkeypatch.setenv("FRAME_DENY", "false")
    assert "X-Frame-Options" not in _build_headers()


def test_default_csp_covers_svelte_assets():
    """Sanity check: the default CSP must not break the app's known asset hosts."""
    # SvelteKit loads its own bundle, inline svelte styles, Google Fonts stylesheet
    # and font files. Our default CSP needs to allow those.
    assert "script-src 'self'" in DEFAULT_CSP
    assert "'unsafe-inline'" in DEFAULT_CSP  # Svelte scoped styles use inline
    assert "https://fonts.googleapis.com" in DEFAULT_CSP
    assert "https://fonts.gstatic.com" in DEFAULT_CSP
    assert "data:" in DEFAULT_CSP  # pasted images and font data URLs
    assert "blob:" in DEFAULT_CSP  # image previews
    assert "frame-ancestors 'none'" in DEFAULT_CSP
