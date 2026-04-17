"""
Optional error tracking via Sentry.

Activated only when ``SENTRY_DSN`` is set in the environment, so the
default install ships nothing to any external service. A self-hosted
Sentry / GlitchTip instance is the natural target for an OSS project.

Tunables (env):
  - ``SENTRY_DSN``: opt-in switch + project DSN
  - ``SENTRY_ENVIRONMENT``: defaults to ``APP_ENV`` if set, else ``development``
  - ``SENTRY_RELEASE``: short release tag (e.g. git SHA), optional
  - ``SENTRY_TRACES_SAMPLE_RATE``: 0.0–1.0; default 0.0 (errors only, no APM)
  - ``SENTRY_SEND_PII``: ``true`` to attach client IPs / cookies; default off
"""

import logging
import os

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def init_sentry() -> bool:
    """Initialize the Sentry SDK if ``SENTRY_DSN`` is set.

    Returns True if Sentry was initialized, False otherwise. Failures to
    import the SDK or initialize are logged but never raised — observability
    must never block the app from starting.
    """
    dsn = os.environ.get('SENTRY_DSN', '').strip()
    if not dsn:
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
    except ImportError:
        logger.warning(
            "SENTRY_DSN is set but sentry-sdk is not installed; "
            "install with `pip install sentry-sdk[fastapi]` to enable."
        )
        return False

    try:
        sentry_sdk.init(
            dsn=dsn,
            environment=os.environ.get('SENTRY_ENVIRONMENT') or os.environ.get('APP_ENV', 'development'),
            release=os.environ.get('SENTRY_RELEASE') or None,
            traces_sample_rate=_env_float('SENTRY_TRACES_SAMPLE_RATE', 0.0),
            send_default_pii=os.environ.get('SENTRY_SEND_PII', '').lower() == 'true',
            integrations=[
                StarletteIntegration(),
                FastApiIntegration(),
            ],
        )
        logger.info("Sentry error tracking initialized")
        return True
    except Exception as e:
        # Catch broadly: misconfigured DSN, network issues at init, etc.
        # The app must keep starting.
        logger.warning(f"Sentry init failed; continuing without it: {e}")
        return False
