from __future__ import annotations

from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser

import pytest

from notifier.daily_recap import TELEGRAM_TEXT_LIMIT, format_daily_recap


NOW = datetime(2026, 8, 24, 18, tzinfo=timezone.utc)


def _row(
    player: str,
    event: str,
    severity: int,
    *,
    minutes_ago: int = 30,
    tier: str = "league",
    source: str = "twitter",
    headline: str | None = None,
    body: str = "",
    url: str = "https://x.com/reporter/status/1",
    **extra,
) -> dict:
    timestamp = NOW - timedelta(minutes=minutes_ago)
    return {
        "source": source,
        "guid": f"{source}:{player}:{minutes_ago}",
        "player_name": player,
        "headline": headline or f"{player} has a new {event} report",
        "body": body,
        "url": url,
        "published_at": timestamp.isoformat(),
        "received_at": timestamp.isoformat(),
        "event_type": event,
        "direction": "neutral",
        "severity": severity,
        "summary": "",
        "is_actionable": 0,
        "tier": tier,
        "outcome": "alert_ready",
        "feedback": None,
        **extra,
    }


class _TagChecker(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag not in {"b", "i", "a", "blockquote"}:
            raise AssertionError(f"unexpected tag: {tag}")
        self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        assert self.stack and self.stack.pop() == tag


def _visible_units(markup: str) -> int:
    class Text(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.parts: list[str] = []

        def handle_data(self, data: str) -> None:
            self.parts.append(data)

    parser = Text()
    parser.feed(markup)
    return len("".join(parser.parts).encode("utf-16-le")) // 2


def test_sections_follow_severity_and_roster_relevance() -> None:
    rows = [
        _row("Roster Player", "injury", 3, tier="mine"),
        _row("Available Player", "usage", 3, tier="claimable", minutes_ago=31),
        _row("League Star", "trade", 4, minutes_ago=32),
        _row("Practice Player", "practice_report", 3, minutes_ago=33),
        _row("Minor Player", "injury", 2, minutes_ago=34),
        _row("Generic Chatter", "other", 2, minutes_ago=35),
        _row("Tiny Note", "injury", 1, minutes_ago=36),
    ]

    recap = format_daily_recap(rows, now=NOW)

    assert [item.player_name for item in recap.big_news] == [
        "League Star",
        "Roster Player",
        "Available Player",
    ]
    assert [item.player_name for item in recap.smaller_moves] == [
        "Practice Player",
        "Minor Player",
    ]
    rendered = "\n".join(recap.parts)
    assert "Generic Chatter" not in rendered
    assert "Tiny Note" not in rendered


def test_repeated_player_event_reports_collapse_to_latest_fact_and_keep_sources() -> None:
    rows = [
        _row(
            "Deshaun Watson",
            "depth_chart",
            3,
            tier="mine",
            minutes_ago=20,
            headline="First report said a decision was coming",
        ),
        _row(
            "Deshaun Watson",
            "depth_chart",
            3,
            tier="league",
            source="rotowire",
            minutes_ago=10,
            headline="Browns named Watson their Week 1 starter",
            url="https://www.rotowire.com/football/player/deshaun-watson-12110",
        ),
    ]

    recap = format_daily_recap(rows, now=NOW)

    assert len(recap.big_news) == 1
    item = recap.big_news[0]
    assert item.report_count == 2
    assert item.tier == "mine"
    assert item.headline == "Browns named Watson their Week 1 starter"
    rendered = "\n".join(recap.parts)
    assert "First report said" not in rendered
    assert "Browns named Watson" in rendered
    assert ">RotoWire</a>" in rendered
    assert ">X</a>" in rendered
    assert "2 reports combined" in rendered


def test_window_feedback_and_timestamp_fallback_are_enforced() -> None:
    stale = _row("Stale", "injury", 5)
    stale["published_at"] = (NOW - timedelta(hours=24, seconds=1)).isoformat()
    future = _row("Future", "injury", 5)
    future["published_at"] = (NOW + timedelta(seconds=1)).isoformat()
    wrong = _row("Wrong", "injury", 5, feedback="wrong")
    fallback = _row("Fallback", "return", 4, minutes_ago=15)
    fallback["published_at"] = None

    recap = format_daily_recap([stale, future, wrong, fallback], now=NOW)

    assert [item.player_name for item in recap.big_news] == ["Fallback"]
    rendered = "\n".join(recap.parts)
    assert "Stale" not in rendered
    assert "Future" not in rendered
    assert "Wrong" not in rendered


def test_ambiguous_subject_never_replays_model_advice_or_assigns_player() -> None:
    row = _row(
        "Mike Washington",
        "injury",
        4,
        headline="Washington handled work after Jeanty left practice",
        summary="ADD Mike Washington everywhere immediately",
        is_actionable=1,
        subject_confident=False,
    )

    recap = format_daily_recap([row], now=NOW)
    rendered = "\n".join(recap.parts)

    assert "Washington handled work after Jeanty left practice" in rendered
    assert "ADD Mike Washington" not in rendered
    assert "4/5 · League report" in rendered
    assert "Player attribution is unclear" in rendered
    assert "4/5 · Mike Washington" not in rendered


def test_upstream_html_is_normalized_escaped_and_unsafe_links_are_not_emitted() -> None:
    row = _row(
        "A <script>Player</script>",
        "trade",
        4,
        headline="Role &amp; mobility <b>changed</b>",
        url="javascript:alert(1)",
    )

    recap = format_daily_recap([row], now=NOW)
    rendered = "\n".join(recap.parts)

    assert "<script>" not in rendered
    assert "&lt;script&gt;Player&lt;/script&gt;" in rendered
    assert "Role &amp; mobility &lt;b&gt;changed&lt;/b&gt;" in rendered
    assert "javascript:" not in rendered
    assert "🔗 X · Aug 24" in rendered


def test_learn_note_is_deterministic_and_not_player_advice() -> None:
    recap = format_daily_recap(
        [
            _row("Injured Player", "injury", 4),
            _row("Role Player", "depth_chart", 3),
            _row("Returning Player", "return", 3),
        ],
        now=NOW,
    )

    assert recap.learn_note == (
        "Injury news can open opportunity, but depth order alone does not "
        "guarantee snaps or touches. Role changes become meaningful when later "
        "reports confirm routes, carries, or playing time."
    )
    rendered = "\n".join(recap.parts)
    assert "LEARN THE GAME" in rendered
    assert "add Injured Player" not in rendered.casefold()


def test_long_recap_splits_only_between_complete_items_with_valid_html() -> None:
    rows = [
        _row(
            f"Player {index:02d}",
            "injury",
            4,
            minutes_ago=index,
            headline=(f"Player {index:02d} received a significant injury update. " * 8),
            body=("This is a separate source-backed report sentence. " * 10),
            url=f"https://example.com/news/{index}",
        )
        for index in range(30)
    ]

    recap = format_daily_recap(rows, now=NOW)

    assert len(recap.parts) > 1
    combined = "\n".join(recap.parts)
    for index in range(30):
        assert combined.count(f"4/5 · Player {index:02d}") == 1
    for part in recap.parts:
        assert _visible_units(part) <= TELEGRAM_TEXT_LIMIT
        parser = _TagChecker()
        parser.feed(part)
        parser.close()
        assert parser.stack == []
    assert sum("LEARN THE GAME" in part for part in recap.parts) == 1


def test_peak_day_recap_caps_output_and_reports_deterministic_omissions() -> None:
    big_rows = [
        _row(
            f"Big Player {index:03d}",
            "injury",
            5,
            minutes_ago=index + 1,
            headline=(
                f"Big Player {index:03d} received a significant injury update. " * 8
            ),
            body=("This is a separate source-backed report sentence. " * 10),
            url=f"https://example.com/big/{index}",
        )
        for index in range(100)
    ]
    smaller_rows = [
        _row(
            f"Small Player {index:03d}",
            "signing",
            2,
            minutes_ago=index + 101,
            headline=(
                f"Small Player {index:03d} completed a minor roster move. " * 8
            ),
            body=("This is a separate source-backed transaction sentence. " * 10),
            url=f"https://example.com/small/{index}",
        )
        for index in range(50)
    ]

    recap = format_daily_recap([*smaller_rows, *big_rows], now=NOW)
    rendered = "\n".join(recap.parts)

    assert len(recap.parts) <= 20
    assert rendered.count("+ 100 more saved reports") == 1

    # The cap keeps the deterministic priority order: severity first, then
    # the newest reports.  Lower-ranked overflow is summarized, not rendered.
    assert rendered.index("5/5 · Big Player 000") < rendered.index(
        "5/5 · Big Player 029"
    )
    assert "5/5 · Big Player 030" not in rendered
    assert rendered.index("2/5 · Small Player 000") < rendered.index(
        "2/5 · Small Player 019"
    )
    assert "2/5 · Small Player 020" not in rendered
    for part in recap.parts:
        assert _visible_units(part) <= TELEGRAM_TEXT_LIMIT


def test_empty_recap_is_explicit_and_still_educational() -> None:
    recap = format_daily_recap([], now=NOW)

    assert recap.big_news == ()
    assert recap.smaller_moves == ()
    assert len(recap.parts) == 1
    rendered = recap.parts[0]
    assert "No major reports" in rendered
    assert "No smaller fantasy-relevant moves" in rendered
    assert "A news headline is one data point" in rendered


@pytest.mark.parametrize("hours", [0, 169])
def test_invalid_window_is_rejected(hours: int) -> None:
    with pytest.raises(ValueError, match="hours"):
        format_daily_recap([], now=NOW, hours=hours)
