"""Classify a news item with DeepSeek V4 Flash via OpenRouter.

Only items that already matched the roster or the trending watchlist reach
this stage, so volume is roughly 10-40 calls a day rather than one per feed
item. Classification failure is non-fatal: an unclassifiable item still
alerts, because a missed injury costs more than an unlabelled one.
"""

from __future__ import annotations

import json
import logging
import re

import requests

from .config import Config
from .logging_utils import structured_log
from .models import Classification, NewsItem

API_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT = 20
MAX_BODY_CHARS = 600

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
  severity: integer 1-5 for fantasy relevance
    1 = noise (preseason rest, routine veteran day off, minor note)
    2 = worth knowing (limited practice, small role change)
    3 = notable (questionable tag, timeshare shift, DNP Wednesday)
    4 = major (ruled out, multi-week injury, starter change, trade)
    5 = season-defining (IR, ACL/Achilles tear, suspension, released)
  fantasy_impact: one sentence, max 140 chars, what the manager should DO
  is_actionable: true if the manager should consider a lineup or waiver move

Judge severity by fantasy consequence, not by how dramatic the wording is.
Preseason starters being rested is severity 1."""


# Deterministic floor against the model under-rating unambiguous news.
# A model that calls a torn ACL "severity 1" would otherwise be silently
# dropped below the alert threshold and never second-guessed, because
# verification only runs on alerts that were actually sent.
HIGH_SIGNAL = re.compile(
    r"\b(torn?\s+(acl|achilles|mcl|patell?a)|ruptured|"
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


def _fallback(reason: str) -> Classification:
    # Severity 3 keeps an unclassified item above a default threshold of 2
    # so a model outage degrades into more alerts, never fewer.
    return Classification(
        event_type="other",
        severity=3,
        fantasy_impact="Could not classify automatically - check the link.",
        is_actionable=True,
        raw={"error": reason},
    )


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
                        "content": SYSTEM_PROMPT.format(event_types=", ".join(EVENT_TYPES)),
                    },
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = _extract_json(content)
    except requests.RequestException as error:
        structured_log(logging.WARNING, "classify.request_failed", error=str(error))
        return _fallback("request_failed")
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as error:
        structured_log(logging.WARNING, "classify.unparseable_response", error=str(error))
        return _fallback("unparseable_response")

    event_type = str(parsed.get("event_type") or "other").strip().lower()
    if event_type not in EVENT_TYPES:
        event_type = "other"

    try:
        severity = max(1, min(5, int(parsed.get("severity", 3))))
    except (TypeError, ValueError):
        severity = 3

    if severity < HIGH_SIGNAL_FLOOR and HIGH_SIGNAL.search(f"{item.headline} {item.body}"):
        structured_log(
            logging.WARNING,
            "classify.severity_floor_applied",
            player=item.player_name,
            modelSeverity=severity,
            flooredTo=HIGH_SIGNAL_FLOOR,
            headline=item.headline,
        )
        severity = HIGH_SIGNAL_FLOOR

    return Classification(
        event_type=event_type,
        severity=severity,
        fantasy_impact=str(parsed.get("fantasy_impact") or "").strip()[:140],
        is_actionable=bool(parsed.get("is_actionable", False)),
        raw=parsed,
    )
