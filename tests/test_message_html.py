"""Guard the Telegram HTML contract.

A literal "<--" marker in the depth chart once made Telegram reject every
alert with "Unsupported start tag" for a full day, including a 5/5
season-ending injury. These assert the rendered message only ever contains
tags Telegram accepts.
"""

from __future__ import annotations

import re

from notifier.logging_utils import redact_log_text
from notifier.models import Alert, Classification, LeagueRef, NewsItem
from notifier.notify import format_alert
from notifier.plays import DepthEntry, TeamContext

TELEGRAM_ALLOWED_TAGS = {"b", "i", "u", "s", "a", "code", "pre", "tg-spoiler", "blockquote"}
TAG = re.compile(r"<\s*/?\s*([^\s>/]*)")


def _alert(**kwargs) -> Alert:
    item = NewsItem(
        source="twitter",
        guid="t:1",
        player_name="George Kittle",
        headline="Visited Dr. ElAttrache about his torn Achilles",
        body="Sources: 49ers TE George Kittle visited Wednesday <in LA> & it went well.",
        url="https://x.com/AdamSchefter/status/1",
        published_at=None,
    )
    context = TeamContext(
        team="SF",
        subject_position="TE",
        same_position=[
            DepthEntry("George Kittle", "TE", 1, 90, True, {}),
            DepthEntry("Jake Tonges", "TE", 2, 178, False, {}),
        ],
        adjacent=[DepthEntry("Brock Purdy", "QB", 1, 55, False, {})],
    )
    return Alert(
        item=item,
        classification=Classification("injury", 4, "Monitor his status.", True, {}),
        tier=kwargs.get("tier", "preseason"),
        per_league=[],
        context=context,
        all_leagues=kwargs.get("all_leagues", []),
    )


def test_rendered_message_uses_only_telegram_supported_tags():
    text = format_alert(_alert())
    for tag in TAG.findall(text):
        assert tag.lower() in TELEGRAM_ALLOWED_TAGS, f"unsupported tag <{tag}> in message"


def test_injured_marker_does_not_open_a_tag():
    text = format_alert(_alert())
    assert "[INJURED]" in text
    assert "<--" not in text


def test_untrusted_body_is_escaped():
    text = format_alert(_alert())
    assert "<in LA>" not in text
    assert "&lt;in LA&gt;" in text


def test_telegram_token_is_redacted_inside_urls():
    # Structurally identical to a real token, but fabricated. Never put a live
    # credential in a test: it lands in git history permanently.
    token = "1234567890:" + "A" * 35
    for sample in (
        f"400 Client Error for url: https://api.telegram.org/bot{token}/sendMessage",
        f"bot{token}",
        f"token={token}",
    ):
        assert token not in redact_log_text(sample)
        assert token.split(":")[1] not in redact_log_text(sample)
