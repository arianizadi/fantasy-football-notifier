"""Classify a news item with DeepSeek V4 Flash via OpenRouter.

Every draft-relevant or in-season feed item can reach this stage.
Classification failure is non-fatal: an unclassifiable item still alerts,
because a missed injury costs more than an unlabelled one.
"""

from __future__ import annotations

import json
import logging
import re
import time

import requests

from .config import Config
from .health import HEALTH
from .logging_utils import structured_log
from .models import Classification, NewsItem

API_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT = 20
MAX_BODY_CHARS = 600
MAX_REQUEST_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 0.5
MAX_RETRY_BACKOFF_SECONDS = 2.0

EVENT_TYPES = (
    "injury",
    "practice_report",
    "inactive",
    "return",
    "trade",
    "signing",
    "release",
    "suspension",
    "depth_chart",
    "usage",
    "other",
)

SYSTEM_PROMPT = """You classify NFL news for a fantasy football manager.

Return ONLY a JSON object with these keys:
  event_type: one of {event_types}
  direction: one of positive, negative, mixed, neutral for this player's
    fantasy outlook (not the emotional tone of the writing)
  severity: integer 1-5 for fantasy relevance
    1 = noise (preseason rest, routine veteran day off, minor note)
    2 = worth knowing (limited practice, small role change)
    3 = notable (questionable tag, timeshare shift, DNP Wednesday)
    4 = major (ruled out, multi-week injury, starter change, trade)
    5 = season-defining (IR, ACL/Achilles tear, suspension, released)
  fantasy_impact: one factual sentence, max 140 chars, describing the likely
    fantasy consequence only. Never give add/drop/start/sit/activate/draft
    instructions; deterministic application code owns all roster advice.
  is_actionable: true if the manager should consider a lineup or waiver move

Judge severity by fantasy consequence, not by how dramatic the wording is.
Preseason starters being rested is severity 1."""


# Deterministic floor against the model under-rating unambiguous news.
# A model that calls a torn ACL "severity 1" would otherwise be silently
# dropped below the alert threshold and never second-guessed, because
# thresholding would otherwise silently drop important reports.
HIGH_SIGNAL = re.compile(
    r"\b((?:torn|tore)\s+(acl|achilles|mcl|patell?a)|"
    r"activated\b[^.\n]{0,100}\b(?:from|off)\s+(?:the\s+)?(?:active/)?"
    r"(?:pup|physically\s+unable\s+to\s+perform|injured\s+reserve|ir)|ruptured|"
    r"injured\s+reserve|placed\s+on\s+ir|out\s+for\s+the\s+season|"
    r"season[-\s]ending|suspended|released|waived|carted\s+off|"
    r"ruled\s+out|will\s+miss\s+\d+\s+(game|week))",
    re.IGNORECASE,
)
HIGH_SIGNAL_FLOOR = 4


def _extract_json(text: str) -> dict:
    """Parse the model's JSON, tolerating markdown fences or stray prose."""
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        braced = re.search(r"\{.*\}", candidate, re.DOTALL)
        if braced:
            candidate = braced.group(0)
    return json.loads(candidate)


def _has_high_signal(item: NewsItem) -> bool:
    return bool(HIGH_SIGNAL.search(f"{item.headline} {item.body}"))


def _fallback_event_type(item: NewsItem) -> str:
    """Best-effort deterministic label when the model is unavailable."""
    text = f"{item.headline} {item.body}".lower()
    if re.search(r"\b(released|waived)\b", text):
        return "release"
    if "suspend" in text:
        return "suspension"
    if re.search(r"\b(activated|return(?:ed|s|ing)?|cleared)\b", text):
        return "return"
    if re.search(r"\b(ruled\s+out|inactive)\b", text):
        return "inactive"
    if _has_high_signal(item):
        return "injury"
    return "other"


def _fallback(reason: str, item: NewsItem) -> Classification:
    # Severity 3 keeps an unclassified item above a default threshold of 2
    # so a model outage degrades into more alerts, never fewer.
    # High-signal news must retain the same deterministic floor used for a
    # successful but under-rated model response. Otherwise preseason's 4+
    # gate silently drops torn ACL/IR news precisely while the model is down.
    high_signal = _has_high_signal(item)
    event_type = _fallback_event_type(item)
    direction = (
        "positive"
        if event_type == "return"
        else "negative"
        if event_type in {"injury", "inactive", "release", "suspension"}
        else "neutral"
    )
    HEALTH.mark("model", ok=False, detail=reason)
    return Classification(
        event_type=event_type,
        severity=HIGH_SIGNAL_FLOOR if high_signal else 3,
        fantasy_impact="Automatic classification unavailable; source requires review.",
        is_actionable=True,
        raw={
            "error": reason,
            "high_signal_floor": high_signal,
            "direction": direction,
        },
    )


def _retryable_request(error: requests.RequestException) -> bool:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status is None or status in {408, 409, 425, 429} or status >= 500


def classify(
    session: requests.Session,
    config: Config,
    item: NewsItem,
    *,
    context: str = "",
) -> Classification:
    user_prompt = (
        f"Player: {item.player_name or 'unknown'}\n"
        f"Headline: {item.headline}\n"
        f"Report: {item.body[:MAX_BODY_CHARS]}"
    )
    # Depth-chart facts are computed in code and handed to the model as
    # grounding so its advice references real backups, never invented ones.
    if context:
        user_prompt += f"\n{context}"

    parsed = None
    failure_reason = "request_failed"
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            response = session.post(
                API_URL,
                timeout=REQUEST_TIMEOUT,
                headers={
                    "Authorization": f"Bearer {config.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/local/fantasy-football-notifier",
                    "X-Title": "fantasy-football-notifier",
                },
                json={
                    "model": config.openrouter_model,
                    "temperature": 0,
                    "max_tokens": 400,
                    "response_format": {"type": "json_object"},
                    # DeepSeek v4 emits reasoning tokens by default, which for this
                    # task burned 1500+ tokens, returned NO content at all, and took
                    # 66s per call. Disabling it is 37x faster and 11x cheaper with
                    # identical classifications. Measured, not assumed.
                    "reasoning": {"enabled": False},
                    # Default routing sprays across ~6 providers and the tail
                    # reached 8.9s. Pinning by throughput held max at 1.5s --
                    # a 5.8x better worst case, which matters more than median
                    # for a breaking-news alert.
                    "provider": {"sort": "throughput"},
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT.format(
                                event_types=", ".join(EVENT_TYPES)
                            ),
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = _extract_json(content)
            break
        except requests.RequestException as error:
            failure_reason = "request_failed"
            retrying = attempt < MAX_REQUEST_ATTEMPTS and _retryable_request(error)
            structured_log(
                logging.WARNING,
                "classify.request_failed",
                error=str(error),
                attempt=attempt,
                retrying=retrying,
            )
            if not retrying:
                break
        except (KeyError, IndexError, TypeError, ValueError) as error:
            # A provider occasionally returns a successful HTTP response with
            # empty content. Retry it just like a transient transport failure.
            failure_reason = "unparseable_response"
            retrying = attempt < MAX_REQUEST_ATTEMPTS
            structured_log(
                logging.WARNING,
                "classify.unparseable_response",
                error=str(error),
                attempt=attempt,
                retrying=retrying,
            )
            if not retrying:
                break

        delay = min(
            RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)),
            MAX_RETRY_BACKOFF_SECONDS,
        )
        time.sleep(delay)

    if parsed is None:
        return _fallback(failure_reason, item)

    HEALTH.mark("model", ok=True, detail=config.openrouter_model)

    event_type = str(parsed.get("event_type") or "other").strip().lower()
    if event_type not in EVENT_TYPES:
        event_type = "other"

    try:
        severity = max(1, min(5, int(parsed.get("severity", 3))))
    except (TypeError, ValueError):
        severity = 3

    high_signal = _has_high_signal(item)
    if severity < HIGH_SIGNAL_FLOOR and high_signal:
        structured_log(
            logging.WARNING,
            "classify.severity_floor_applied",
            player=item.player_name,
            modelSeverity=severity,
            flooredTo=HIGH_SIGNAL_FLOOR,
            headline=item.headline,
        )
        severity = HIGH_SIGNAL_FLOOR
    if event_type == "other" and high_signal:
        event_type = _fallback_event_type(item)

    return Classification(
        event_type=event_type,
        severity=severity,
        fantasy_impact=str(parsed.get("fantasy_impact") or "").strip()[:140],
        is_actionable=bool(parsed.get("is_actionable", False)),
        raw=parsed,
    )
