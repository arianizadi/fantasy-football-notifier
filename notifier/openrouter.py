"""OpenRouter chat helper that negotiates per-model reasoning support.

Models disagree about reasoning tokens in opposite directions:

  deepseek-v4-flash  emits them by default, burning the whole token budget and
                     returning no content at all. Must be disabled.
  stealth/ox-alpha   rejects the request outright:
                     400 "Reasoning is mandatory for this endpoint".

A single hardcoded setting therefore breaks one model or the other. This tries
the preferred mode, downgrades on the specific 400, and remembers the result so
the discovery cost is paid once per model per process.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import requests

from .logging_utils import structured_log

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 90

# Models that mandate reasoning say so in the 400 body.
_MANDATORY_MARKERS = ("reasoning is mandatory", "cannot be disabled")

_lock = threading.Lock()
_reasoning_required: dict[str, bool] = {}


def reasoning_is_required(model: str) -> bool:
    with _lock:
        return _reasoning_required.get(model, False)


def _remember(model: str, required: bool) -> None:
    with _lock:
        _reasoning_required[model] = required


def chat(
    session: requests.Session,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 400,
    temperature: float = 0,
    json_mode: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    prefer_no_reasoning: bool = True,
    reasoning_effort: str | None = None,
) -> requests.Response:
    """POST a chat completion, adapting to the model's reasoning policy.

    reasoning_effort opts the caller INTO reasoning at a given level. Use it
    for background work where quality matters more than latency; leave it None
    on the alert path, where reasoning costs 60+ seconds.
    """

    def build(disable_reasoning: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
            "provider": {"sort": "throughput"},
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if disable_reasoning:
            body["reasoning"] = {"enabled": False}
        elif reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        return body

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    disable = (
        prefer_no_reasoning
        and not reasoning_effort
        and not reasoning_is_required(model)
    )

    response = session.post(API_URL, timeout=timeout, headers=headers, json=build(disable))

    if response.status_code == 400 and disable:
        detail = ""
        try:
            detail = str(response.json().get("error", {}).get("message", "")).lower()
        except ValueError:
            detail = response.text.lower()
        if any(marker in detail for marker in _MANDATORY_MARKERS):
            _remember(model, True)
            structured_log(
                logging.INFO,
                "openrouter.reasoning_required",
                model=model,
                hint="Retrying with reasoning enabled; budget more max_tokens.",
            )
            # Reasoning tokens consume the completion budget, so a model that
            # mandates them needs far more headroom or it truncates mid-JSON.
            body = build(False)
            body["max_tokens"] = max(max_tokens, 2000)
            response = session.post(API_URL, timeout=timeout, headers=headers, json=body)

    return response
