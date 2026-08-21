"""Optional third opinion from a free OpenRouter stealth model.

Stealth models are unreleased models published under a codename, free while
the lab's promotional window is open, then withdrawn. So this is strictly
best-effort:

  * no stealth model available -> the feature disables itself
  * the model errors, 404s, or cannot hold the JSON schema -> that item is
    skipped silently

It is free, so a failure costs nothing and must never degrade the alert or the
paid second opinion. Availability is re-checked periodically because the
namespace rotates without notice.
"""

from __future__ import annotations

import logging
import threading
import time

import requests

from .logging_utils import structured_log
from .model_registry import free_candidates
from .openrouter import chat

RECHECK_SECONDS = 6 * 60 * 60
REQUEST_TIMEOUT = 60
# Reasoning-mandatory models spend the completion budget thinking, so leave
# room or the JSON truncates.
MAX_TOKENS = 2000

SYSTEM_PROMPT = """You rate NFL news severity for a fantasy football manager.
Return ONLY JSON: {"severity":int,"note":str}
severity 1-5 by FANTASY CONSEQUENCE, not dramatic wording:
1=noise 2=worth knowing 3=notable 4=major 5=season-defining
note: under 100 characters."""


class StealthReviewer:
    """Resolves a free stealth model and asks it for a severity, or gives up."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._session = requests.Session()
        self._lock = threading.Lock()
        self._model: str | None = None
        self._checked_at = 0.0

    def model(self) -> str | None:
        """Current free stealth model, re-resolved on a slow interval."""
        with self._lock:
            fresh = (time.time() - self._checked_at) < RECHECK_SECONDS
            if fresh:
                return self._model

        found: str | None = None
        try:
            candidates = free_candidates(self._session, stealth_only=True)
            found = candidates[0].model_id if candidates else None
        except (requests.RequestException, ValueError) as error:
            structured_log(logging.WARNING, "stealth.discovery_failed", error=str(error))

        with self._lock:
            changed = found != self._model
            self._model = found
            self._checked_at = time.time()

        if changed:
            structured_log(
                logging.INFO,
                "stealth.model_changed" if found else "stealth.disabled",
                model=found,
                reason=None if found else "no free stealth model published",
            )
        return found

    def review(self, headline: str, body: str, first_severity: int) -> tuple[int, str] | None:
        """Return (severity, note), or None if unavailable or unusable."""
        model = self.model()
        if not model:
            return None

        prompt = (
            f"Headline: {headline}\n"
            f"Report: {body[:500]}\n\n"
            f"Another model rated this severity {first_severity}/5."
        )
        try:
            response = chat(
                self._session,
                self._api_key,
                model,
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as error:
            structured_log(logging.DEBUG, "stealth.request_failed", model=model,
                           error=str(error))
            return None

        if not response.ok:
            # 404 usually means the free window closed; force rediscovery.
            if response.status_code == 404:
                with self._lock:
                    self._model = None
                    self._checked_at = 0.0
                structured_log(logging.INFO, "stealth.withdrawn", model=model)
            else:
                structured_log(logging.DEBUG, "stealth.http_error", model=model,
                               httpStatus=response.status_code)
            return None

        from .classify import _extract_json

        try:
            content = response.json()["choices"][0]["message"].get("content")
            parsed = _extract_json(content)
            severity = max(1, min(5, int(parsed["severity"])))
        except (KeyError, IndexError, TypeError, ValueError):
            # A free model that cannot hold the schema is simply skipped. It
            # costs nothing and the paid verifier already covered this item.
            structured_log(logging.DEBUG, "stealth.unusable_response", model=model)
            return None

        return severity, str(parsed.get("note") or "").strip()[:120]
