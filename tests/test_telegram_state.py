from __future__ import annotations

from datetime import datetime, timezone

from notifier.models import Alert, Classification, NewsItem
from notifier.telegram_state import (
    TelegramState,
    alert_token,
    feedback_markup,
    feedback_markup_for_token,
)


def _alert(guid: str = "twitter:1:George Kittle") -> Alert:
    return Alert(
        item=NewsItem(
            source="twitter",
            guid=guid,
            player_name="George Kittle",
            headline="49ers activated George Kittle from active/PUP",
            body="George Kittle was activated from active/PUP.",
            url="https://x.com/example/status/1",
            published_at=None,
        ),
        classification=Classification(
            event_type="return",
            severity=4,
            fantasy_impact="Draft outlook improves; confirm full-practice participation.",
            is_actionable=True,
            raw={},
        ),
        tier="preseason",
    )


def test_player_reply_chain_and_feedback_survive_restart(tmp_path) -> None:
    path = tmp_path / "telegram-state.json"
    state = TelegramState(path, thread_hours=168)
    alert = _alert()

    assert state.previous_message_id("George Kittle") is None
    token = state.record_sent(alert, 321)
    assert token == alert_token(alert.item.guid)
    assert state.previous_message_id("Kittle") is None  # exact player key only
    assert state.previous_message_id("George Kittle") == 321
    assert state.record_feedback(token, "useful") is True

    reloaded = TelegramState(path, thread_hours=168)
    assert reloaded.previous_message_id("George Kittle") == 321
    assert reloaded.feedback_verdict(token) == "useful"


def test_expired_reply_chain_is_not_reused(tmp_path) -> None:
    state = TelegramState(tmp_path / "telegram-state.json", thread_hours=1)
    state.record_sent(_alert(), 99)

    assert state.previous_message_id("George Kittle", now=10**12) is None


def test_feedback_markup_uses_stable_token_and_selected_label() -> None:
    guid = "twitter:1:George Kittle"
    token = alert_token(guid)
    initial = feedback_markup(guid)
    selected = feedback_markup_for_token(token, selected="wrong")

    assert initial["inline_keyboard"][0][0]["callback_data"] == f"feedback:{token}:useful"
    assert selected["inline_keyboard"][0][1]["text"] == "✓ Wrong"
    assert selected["inline_keyboard"][0][1]["callback_data"] == f"feedback:{token}:wrong"


def test_daily_digest_is_ordered_by_severity_and_marked_once(tmp_path) -> None:
    state = TelegramState(tmp_path / "telegram-state.json")
    state.record_sent(_alert("one"), 1)
    lower = _alert("two")
    lower = Alert(
        item=lower.item,
        classification=Classification("return", 2, "Monitor practice reports.", False, {}),
        tier=lower.tier,
    )
    state.record_sent(lower, 2)
    now = datetime.now(timezone.utc)

    text = state.format_digest(now, timezone_name="America/Los_Angeles")
    assert "Daily fantasy action digest" in text
    assert text.index("4/5") < text.index("2/5")
    assert "Confirm full practice and Week 1 status" in text
    assert state.digest_due(now, hour=0, timezone_name="America/Los_Angeles") is True
    state.mark_digest_sent(now, timezone_name="America/Los_Angeles")
    assert state.digest_due(now, hour=0, timezone_name="America/Los_Angeles") is False


def test_digest_never_replays_model_authored_roster_instruction(tmp_path) -> None:
    state = TelegramState(tmp_path / "telegram-state.json")
    alert = _alert("directive")
    alert = Alert(
        item=alert.item,
        classification=Classification(
            "injury",
            4,
            "Add his backup in every league immediately.",
            True,
            {},
        ),
        tier="league",
    )
    state.record_sent(alert, 8)

    text = state.format_digest(
        datetime.now(timezone.utc),
        timezone_name="America/Los_Angeles",
    )

    assert "Add his backup" not in text
    assert "Verify official availability" in text


def test_feedback_tokens_are_retained_for_every_alert_inside_seven_days(tmp_path) -> None:
    state = TelegramState(tmp_path / "telegram-state.json")
    first_token = state.record_sent(_alert("alert-0"), 1)
    for index in range(1, 502):
        state.record_sent(_alert(f"alert-{index}"), index + 1)

    assert state.record_feedback(first_token, "useful") is True
