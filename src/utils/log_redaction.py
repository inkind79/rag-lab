"""
Logging filter that scrubs known secrets from log records before they hit
console or file handlers.

Why this exists: it's easy for a stack trace, a urllib error, an httpx
exception, or just a hand-rolled ``logger.debug(f"... {api_key}")`` to
leak credentials into ``logs/app.log``. A central filter is a safety net,
not a substitute for not logging secrets in the first place.

What we redact:
  - Anything explicitly registered via ``register_secret(value)`` — used
    for ``OLLAMA_API_KEY`` and ``JWT_SECRET`` so updates at runtime
    (e.g. via the settings UI) keep the redaction in sync.
  - HTTP bearer tokens: ``Authorization: Bearer <opaque>`` and bare
    ``Bearer <token>`` substrings.
  - Hugging Face access tokens (``hf_…``).
  - OpenAI-style ``sk-…`` keys.
"""

import logging
import re
import threading
from typing import Set

REDACTED = "***REDACTED***"

# Regex patterns are matched on the formatted log message text.
_PATTERNS = [
    # Authorization: Bearer <token>
    re.compile(r'(Authorization\s*[:=]\s*Bearer\s+)([A-Za-z0-9_\-.~+/]+=*)', re.IGNORECASE),
    # Bare "Bearer <token>"
    re.compile(r'(\bBearer\s+)([A-Za-z0-9_\-.~+/]{16,}=*)'),
    # HuggingFace access tokens
    re.compile(r'\bhf_[A-Za-z0-9]{20,}\b'),
    # OpenAI-style keys
    re.compile(r'\bsk-[A-Za-z0-9]{20,}\b'),
]

_secrets_lock = threading.Lock()
_registered_secrets: Set[str] = set()


def register_secret(value: str) -> None:
    """Add ``value`` to the redaction set. Short / empty values are ignored."""
    if not value or len(value) < 8:
        return
    with _secrets_lock:
        _registered_secrets.add(value)


def unregister_secret(value: str) -> None:
    """Remove ``value`` from the redaction set."""
    if not value:
        return
    with _secrets_lock:
        _registered_secrets.discard(value)


def _snapshot_secrets() -> Set[str]:
    with _secrets_lock:
        return set(_registered_secrets)


def _redact_text(text: str) -> str:
    """Apply all redaction rules to ``text``. Returns the cleaned string."""
    if not text:
        return text

    # Registered exact-string secrets first — straight substring replacement.
    for secret in _snapshot_secrets():
        if secret in text:
            text = text.replace(secret, REDACTED)

    # Then the structured patterns (auth headers, hf_, sk-, etc.)
    for pat in _PATTERNS:
        if pat.groups >= 2:
            text = pat.sub(lambda m: m.group(1) + REDACTED, text)
        else:
            text = pat.sub(REDACTED, text)

    return text


class RedactingFilter(logging.Filter):
    """Mutates ``record`` in place so the rendered message is redacted.

    ``logging.Filter.filter`` is allowed to modify the record; we resolve the
    fully-formatted message via ``record.getMessage()``, swap it back in as a
    literal ``msg`` with empty ``args``, and let downstream handlers format
    as usual. That way a single pass covers every handler attached to the
    root logger.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            redacted = _redact_text(msg)
            if redacted != msg:
                record.msg = redacted
                record.args = ()
        except Exception:
            # Never let the filter itself break logging.
            pass
        return True
