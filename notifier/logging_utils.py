"""Structured JSON logging with secret redaction.

Mirrors the conventions in fantasy-league-importer/worker/sync.py so both
workers emit the same shape into journald.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

LOGGER = logging.getLogger("fantasy-news-notifier")

SENSITIVE_LOG_KEYS = frozenset(
    {
        "apikey",
        "authorization",
        "bottoken",
        "cookie",
        "credentials",
        "espns2",
        "openrouterapikey",
        "password",
        "proxyurl",
        "secret",
        "session",
        "swid",
        "telegrambottoken",
        "token",
    }
)

MAX_LOG_TEXT = 1000


def redact_log_text(value: str) -> str:
    """Strip credentials that can appear inside free-form strings."""
    redacted = re.sub(
        r"\b([a-z][a-z0-9+.-]*:\/\/)[^\/\s:@]+:[^@\s\/]+@",
        r"\1[REDACTED]@",
        value,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(
        r"Bearer\s+[^\s,;]+",
        "Bearer [REDACTED]",
        redacted,
        flags=re.IGNORECASE,
    )
    # Telegram bot tokens contain a numeric bot id, a colon, and a long secret.
    # They also appear inside URLs as /bot<token>/, where a leading \b fails:
    # there is no word boundary between the "t" of "bot" and the first digit.
    # This leaked the live token into journald seven times before it was caught.
    redacted = re.sub(r"(?:bot)?\d{6,12}:[A-Za-z0-9_-]{30,}", "[REDACTED]", redacted)
    redacted = re.sub(
        r"(api\.telegram\.org/)[^/\s]+", r"\1[REDACTED]", redacted, flags=re.IGNORECASE
    )
    # OpenRouter keys are sk-or-v1-<hex>
    redacted = re.sub(r"\bsk-or-[A-Za-z0-9-]{10,}", "[REDACTED]", redacted)
    redacted = re.sub(
        r"((?:api[_-]?key|authorization|cookie|credential|espn[_-]?s2|"
        r"password|proxy[_-]?url|secret|session|swid|token)"
        r"\s*[=:]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        redacted,
        flags=re.IGNORECASE,
    )
    return redacted[:MAX_LOG_TEXT]


def safe_log_value(key: str, value: Any, depth: int = 0) -> Any:
    normalized_key = "".join(character for character in key.lower() if character.isalnum())
    if (
        normalized_key in SENSITIVE_LOG_KEYS
        or normalized_key.endswith("token")
        or normalized_key.endswith("secret")
        or normalized_key.endswith("apikey")
    ):
        return "[REDACTED]"
    if depth > 4:
        return "[TRUNCATED]"
    if isinstance(value, str):
        return redact_log_text(value)
    if isinstance(value, dict):
        return {
            nested_key: safe_log_value(nested_key, nested_value, depth + 1)
            for nested_key, nested_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [safe_log_value(key, item, depth + 1) for item in value[:50]]
    return value


def structured_log(level: int, event: str, *, run_id: str | None = None, **fields: Any) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "level": logging.getLevelName(level).lower(),
        "service": "fantasy-news-notifier",
        "operation": "news-notify",
        "event": event,
        "runId": run_id,
        **fields,
    }
    LOGGER.log(
        level,
        json.dumps(
            {key: safe_log_value(key, value) for key, value in record.items()},
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def configure_logging(verbose: bool = False) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOGGER.propagate = False


class NotifierError(RuntimeError):
    """Fatal, operator-actionable error."""
