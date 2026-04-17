"""Sentry init is no-op without DSN, succeeds with one, never crashes startup."""

import sys


def _reload_observability():
    for m in list(sys.modules):
        if m.startswith('src.api.observability') or m.startswith('sentry_sdk'):
            del sys.modules[m]
    from src.api.observability import init_sentry
    return init_sentry


def test_init_returns_false_when_dsn_unset(monkeypatch):
    monkeypatch.delenv('SENTRY_DSN', raising=False)
    init_sentry = _reload_observability()
    assert init_sentry() is False


def test_init_returns_true_with_valid_dsn(monkeypatch):
    monkeypatch.setenv('SENTRY_DSN', 'https://abc123@o0.ingest.sentry.io/1')
    monkeypatch.setenv('SENTRY_ENVIRONMENT', 'test')
    init_sentry = _reload_observability()
    assert init_sentry() is True


def test_init_swallows_bogus_dsn(monkeypatch):
    """An obviously malformed DSN must not crash the app on startup."""
    monkeypatch.setenv('SENTRY_DSN', 'not-a-real-dsn')
    init_sentry = _reload_observability()
    # Result depends on sentry-sdk's parser; the contract is "doesn't raise".
    result = init_sentry()
    assert result in (True, False)


def test_dsn_with_only_whitespace_treated_as_unset(monkeypatch):
    monkeypatch.setenv('SENTRY_DSN', '   ')
    init_sentry = _reload_observability()
    assert init_sentry() is False
