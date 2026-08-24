"""Guard the Telegram HTML contract.

A literal "<--" marker in the depth chart once made Telegram reject every
alert with "Unsupported start tag" for a full day, including a 5/5
season-ending injury. These assert the rendered message only ever contains
tags Telegram accepts.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime, timezone

from notifier.logging_utils import redact_log_text
from notifier.models import Alert, Classification, LeagueRef, NewsItem, RosterCapacity
from notifier.notify import TELEGRAM_TEXT_LIMIT, _visible_units, format_alert
from notifier.plays import Beneficiary, DepthEntry, LeaguePlays, TeamContext

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


def test_subject_marker_is_event_aware_and_does_not_invent_status():
    text = format_alert(_alert())
    assert "report subject" in text
    assert "<b>INJURY</b>" in text
    assert "[INJURED]" not in text
    assert "<--" not in text


def test_untrusted_body_is_escaped():
    text = format_alert(_alert())
    assert "<in LA>" not in text
    assert "&lt;in LA&gt;" in text


def test_source_url_cannot_inject_html_attributes():
    alert = _alert()
    malicious = replace(
        alert.item,
        url='https://example.test/report/" onclick="bad',
    )

    text = format_alert(replace(alert, item=malicious))

    assert '" onclick="bad' not in text
    assert "&quot; onclick=&quot;bad" in text


def test_unsafe_source_url_scheme_is_not_linked():
    alert = _alert()
    text = format_alert(
        replace(alert, item=replace(alert.item, url="javascript:alert(1)"))
    )

    assert "javascript:" not in text
    assert "X source" in text


def test_mobile_waiver_alert_has_scan_first_sections_and_one_option_per_line():
    league = LeagueRef("espn", "1", "The Certified Sped League 2.0", "Mine")
    ownership = {
        league.key: ("rostered", "CMC Tax Ret"),
    }
    alert = Alert(
        item=NewsItem(
            source="rotowire",
            guid="rotowire:henderson",
            player_name="TreVeyon Henderson",
            headline="Exits practice with lower leg issue",
            body=(
                "Henderson left Monday's practice with a right lower leg issue, "
                "Mike Kadlick of Sports Illustrated reports."
            ),
            url="https://example.test/henderson",
            published_at=datetime(2026, 8, 24, 9, 42, tzinfo=timezone.utc),
        ),
        classification=Classification("injury", 3, "", True, {}),
        tier="claimable",
        per_league=[
            LeaguePlays(
                league=league,
                subject_state="rostered",
                subject_owner="CMC Tax Ret",
                beneficiaries=[
                    Beneficiary("Jam Miller", "RB", 3, "free_agent"),
                    Beneficiary("Reggie Gilliam", "RB", 4, "free_agent"),
                ],
                capacity=RosterCapacity(5, 5, 0, 1),
            )
        ],
        context=TeamContext(
            team="NE",
            subject_position="RB",
            same_position=[
                DepthEntry(
                    "TreVeyon Henderson",
                    "RB",
                    1,
                    39,
                    True,
                    ownership,
                ),
                DepthEntry(
                    "Rhamondre Stevenson",
                    "RB",
                    2,
                    58,
                    False,
                    {league.key: ("rostered", "Bak Choi Cr")},
                ),
                DepthEntry(
                    "Jam Miller",
                    "RB",
                    3,
                    999,
                    False,
                    {league.key: ("free_agent", "")},
                ),
            ],
            adjacent=[
                DepthEntry("A.J. Brown", "WR", 1, 10, False, {}),
            ],
            player_index_refreshed_at=datetime(
                2026, 8, 24, 8, 12, tzinfo=timezone.utc
            ),
        ),
        all_leagues=[league],
    )

    text = format_alert(alert)

    assert text.startswith(
        "🟡 <b>[3/5] WAIVER WATCH · TreVeyon Henderson</b>\n"
        "<b>INJURY</b> · Exits practice with lower leg issue"
    )
    assert "\n\n🎯 <b>YOUR OPTIONS</b>\n" in text
    assert "\n🟢 <b>Jam Miller</b> · Sleeper RB3\n" in text
    assert "\n🟢 <b>Reggie Gilliam</b> · Sleeper RB4\n" in text
    assert "Jam Miller</b> |" not in text
    assert "Roster space · Bench 5/5 full · IR 0/1 open" in text
    assert "\n\n📰 <b>REPORT</b>\n<blockquote>" in text
    assert "\n\n📋 <b>NE RB DEPTH</b>\n" in text
    assert text.index("🎯") < text.index("📰") < text.index("📋")
    assert "OTHER SLEEPER DEPTH LEADERS" not in text
    assert "A.J. Brown" not in text
    assert "Sleeper #999" not in text
    assert "\n\n\n" not in text


def test_mobile_roster_role_alert_leads_with_next_step_and_clear_ownership():
    league = LeagueRef("espn", "1", "The Certified Sped League 2.0", "Mine")
    alert = Alert(
        item=NewsItem(
            source="rotowire",
            guid="rotowire:brooks",
            player_name="Jonathon Brooks",
            headline="Possible committee with Hubbard's return",
            body=(
                "Coach Dave Canales said Carolina plans a Week 1 committee with "
                "Brooks and Chuba Hubbard if Hubbard is healthy."
            ),
            url="https://example.test/brooks",
            published_at=datetime(2026, 8, 24, 9, 35, tzinfo=timezone.utc),
        ),
        classification=Classification("depth_chart", 3, "", True, {}),
        tier="mine",
        context=TeamContext(
            team="CAR",
            subject_position="RB",
            same_position=[
                DepthEntry(
                    "Chuba Hubbard",
                    "RB",
                    1,
                    66,
                    False,
                    {league.key: ("rostered", "Bak Choi Cr")},
                    "Questionable",
                ),
                DepthEntry(
                    "Jonathon Brooks",
                    "RB",
                    2,
                    88,
                    True,
                    {league.key: ("mine", "")},
                ),
                DepthEntry(
                    "AJ Dillon",
                    "RB",
                    3,
                    688,
                    False,
                    {league.key: ("free_agent", "")},
                ),
            ],
            player_index_refreshed_at=datetime(
                2026, 8, 24, 8, 12, tzinfo=timezone.utc
            ),
        ),
        all_leagues=[league],
    )

    text = format_alert(alert)

    assert text.startswith("🟡 <b>[3/5] YOUR ROSTER · Jonathon Brooks</b>")
    assert "🎯 <b>YOUR OPTIONS</b>" not in text
    assert "⚠️ <b>NEXT STEP</b>" in text
    assert "<blockquote>Coach Dave Canales" in text
    assert "🔒 <b>RB1 Chuba Hubbard</b>\n↳ Owned by Bak Choi Cr" in text
    assert "➡️ <b>RB2 Jonathon Brooks</b> · report subject\n↳ Your roster" in text
    assert "🟢 <b>RB3 AJ Dillon</b>\n↳ Available" in text
    assert text.index("NEXT STEP") < text.index("REPORT") < text.index("CAR RB DEPTH")


def test_preescaped_feed_text_is_normalized_before_telegram_escaping():
    alert = _alert()
    item = replace(
        alert.item,
        headline="Experience, command &amp; mobility were key",
        body="Experience, command &amp; mobility were key",
    )

    text = format_alert(replace(alert, item=item))

    assert "command &amp; mobility" in text
    assert "&amp;amp;" not in text


def test_failed_live_ownership_refresh_uses_neutral_depth_icons():
    league = LeagueRef("sleeper", "1", "Home", "Mine")
    alert = replace(
        _alert(all_leagues=[league]),
        availability_refresh_failed=True,
    )

    text = format_alert(alert)

    assert "Live league ownership unavailable" in text
    assert "🟢 <b>TE2 Jake Tonges</b>" not in text
    assert "• <b>TE2 Jake Tonges</b>" in text
    assert "Available" not in text
    assert "Owned by" not in text
    assert "Your roster" not in text


def test_different_league_ownership_is_split_across_mobile_lines():
    espn = LeagueRef("espn", "1", "Sunday Crew", "Mine")
    sleeper = LeagueRef("sleeper", "2", "No Punt Intended", "Mine")
    context = TeamContext(
        team="SF",
        subject_position="TE",
        same_position=[
            DepthEntry(
                "George Kittle",
                "TE",
                1,
                90,
                True,
                {
                    espn.key: ("mine", ""),
                    sleeper.key: ("free_agent", ""),
                },
            )
        ],
    )
    alert = replace(
        _alert(all_leagues=[espn, sleeper]),
        context=context,
    )

    text = format_alert(alert)

    assert "↳ Sunday Crew: Your roster\n↳ No Punt Intended: Available" in text
    assert "Sunday Crew: Your roster · No Punt Intended: Available" not in text


def test_pathological_alert_is_safely_shortened_for_telegram():
    alert = _alert()
    oversized = "George Kittle update " + ("🏈" * 5000)
    item = replace(alert.item, headline=oversized, body=oversized)

    text = format_alert(replace(alert, item=item))

    assert _visible_units(text) <= TELEGRAM_TEXT_LIMIT
    assert "Some details omitted to fit Telegram" in text
    assert text.count("<i>") == text.count("</i>")
    assert text.count("<b>") == text.count("</b>")


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
