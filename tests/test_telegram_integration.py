from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import requests

import notifier.notify as notify_module
import notifier.telegram_state as telegram_state_module
from notifier.dedupe import semantic_event_type
from notifier.models import ActionUrgency, Alert, Classification, NewsItem
from notifier.notify import TELEGRAM_TEXT_LIMIT, _visible_units, send_alert
from notifier.telegram_control import ScheduledReport, TelegramControl
from notifier.telegram_state import TelegramState, alert_token


class Response:
    def __init__(self, message_id: int = 1) -> None:
        self.ok = True
        self.status_code = 200
        self._message_id = message_id

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"ok": True, "result": {"message_id": self._message_id}}


class FailedEditResponse(Response):
    def __init__(self) -> None:
        super().__init__()
        self.ok = False
        self.status_code = 400
        self.text = ""

    def raise_for_status(self) -> None:
        raise requests.HTTPError("400 for editMessageText")

    def json(self) -> dict:
        return {"ok": False, "description": "message to edit not found"}


class NotModifiedEditResponse(FailedEditResponse):
    def json(self) -> dict:
        return {
            "ok": False,
            "description": "Bad Request: message is not modified",
        }


class Session:
    def __init__(
        self,
        *,
        failed_edits: int = 0,
        not_modified_edit_numbers: set[int] | None = None,
    ) -> None:
        self.payloads: list[dict] = []
        self.urls: list[str] = []
        self._failed_edits = failed_edits
        self._edit_calls = 0
        self._not_modified_edit_numbers = not_modified_edit_numbers or set()
        self._next_message_id = 100

    def post(self, url: str, *, json: dict, timeout) -> Response:
        del timeout
        self.urls.append(url)
        self.payloads.append(json)
        if url.endswith("/editMessageText"):
            self._edit_calls += 1
            if self._failed_edits:
                self._failed_edits -= 1
                return FailedEditResponse()
            if self._edit_calls in self._not_modified_edit_numbers:
                return NotModifiedEditResponse()
            return Response(int(json["message_id"]))
        response = Response(self._next_message_id)
        self._next_message_id += 1
        return response


def _config(tmp_path, *, dry_run: bool = False, controls: bool = True):
    return SimpleNamespace(
        telegram_bot_token="fake-token",
        telegram_chat_id="123",
        state_dir=tmp_path,
        player_thread_hours=168,
        telegram_controls_enabled=controls,
        dry_run=dry_run,
        daily_digest_enabled=True,
        daily_digest_hour=18,
        daily_digest_timezone="America/Los_Angeles",
    )


def _alert(
    guid: str,
    *,
    headline: str = "49ers activated George Kittle",
    body: str | None = None,
    event_type: str = "return",
    severity: int = 4,
    player_name: str = "George Kittle",
    source: str = "twitter",
    subject_confident: bool = True,
) -> Alert:
    item = NewsItem(
        source=source,
        guid=guid,
        player_name=player_name,
        headline=headline,
        body=body if body is not None else headline,
        url="https://x.com/example/status/1",
        published_at=None,
        subject_confident=subject_confident,
    )
    return Alert(
        item=item,
        classification=Classification(
            event_type,
            severity,
            "Availability changed.",
            True,
            {},
        ),
        tier="preseason",
    )


def _trade_alert(
    guid: str,
    headline: str,
    *,
    body: str | None = None,
    severity: int = 3,
    source: str = "twitter",
) -> Alert:
    return _alert(
        guid,
        headline=headline,
        body=body,
        event_type="trade",
        severity=severity,
        player_name="Kayshon Boutte",
        source=source,
    )


def test_same_event_corroboration_edits_existing_message_and_digest(tmp_path) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session()

    assert send_alert(session, config, _alert("tweet:1")) == 100
    state = notify_module.telegram_state(config)
    first_token = alert_token(_alert("tweet:1").item)
    assert state.record_feedback(first_token, "useful") is True
    updated = _alert(
        "tweet:2",
        headline="49ers confirm George Kittle was activated and returned to practice",
    )
    assert send_alert(session, config, updated) == 100

    first, second = session.payloads
    assert session.urls[0].endswith("/sendMessage")
    assert session.urls[1].endswith("/editMessageText")
    assert "reply_parameters" not in first
    assert first["reply_markup"]["inline_keyboard"][0][0]["callback_data"].startswith(
        "feedback:"
    )
    assert second["message_id"] == 100
    assert second["parse_mode"] == "HTML"
    assert "<b>" in second["text"]
    assert second["link_preview_options"] == {"is_disabled": True}
    second_token = alert_token(updated.item)
    assert second["reply_markup"] != first["reply_markup"]
    assert second["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == (
        f"feedback:{second_token}:useful"
    )
    assert state.feedback_verdict(first_token) == "useful"
    assert state.feedback_verdict(second_token) == ""
    assert state.record_feedback(second_token, "wrong") is True
    assert state.feedback_verdict(first_token) == "useful"
    assert state.feedback_verdict(second_token) == "wrong"
    assert state.previous_message_id("George Kittle") == 100
    persisted = json.loads((tmp_path / "telegram-state.json").read_text())
    thread = persisted["threads"]["georgekittle"]
    assert thread["eventType"] == "return"
    assert thread["severity"] == 4
    assert thread["eventStatus"] == "cleared"
    assert thread["eventFactSignature"] == "unspecified"
    assert thread["token"] == second_token
    assert thread["latestHeadline"] == updated.item.headline
    assert len(persisted["alerts"]) == 1
    assert persisted["alerts"][0]["messageId"] == 100
    assert persisted["alerts"][0]["token"] == second_token
    assert persisted["alerts"][0]["headline"] == updated.item.headline
    assert first_token in persisted["feedbackTargets"]
    assert second_token in persisted["feedbackTargets"]
    assert not (tmp_path / "sent-messages.json").exists()


def test_same_story_edit_uses_revalidated_action_urgency(tmp_path) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session()
    first = replace(
        _alert("tweet:urgent", severity=4),
        urgency=ActionUrgency("act_today", "act_today"),
    )
    update = replace(
        _alert(
            "tweet:urgent-update",
            headline="49ers confirm George Kittle was activated and returned to practice",
            severity=3,
        ),
        urgency=ActionUrgency("monitor", "monitor"),
    )

    assert send_alert(session, config, first) == 100
    assert send_alert(session, config, update) == 100

    assert session.urls[-1].endswith("/editMessageText")
    assert "⏰ <b>ACT TODAY</b>" not in session.payloads[-1]["text"]
    assert "👀 <b>MONITOR</b>" in session.payloads[-1]["text"]
    persisted = json.loads((tmp_path / "telegram-state.json").read_text())
    assert persisted["threads"]["georgekittle"]["urgencyLevel"] == "monitor"


def test_same_story_edit_does_not_downgrade_urgency_on_provider_failure(
    tmp_path,
) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session()
    headline = "The 49ers ruled out George Kittle for Week 1"
    first = replace(
        _alert(
            "tweet:urgent-out",
            headline=headline,
            event_type="injury",
            severity=4,
        ),
        urgency=ActionUrgency(
            "act_now",
            "act_now",
            reason_codes=("starter_unavailable",),
            action_available=True,
            roster_relevant=True,
        ),
    )
    unverified_update = replace(
        _alert(
            "rotowire:urgent-out",
            headline=headline,
            event_type="injury",
            severity=3,
            source="rotowire",
        ),
        availability_refresh_failed=True,
        urgency=ActionUrgency(
            "monitor",
            "monitor",
            reason_codes=("availability_unverified",),
            action_available=False,
            roster_relevant=True,
            availability_verified=False,
        ),
    )

    assert send_alert(session, config, first) == 100
    assert send_alert(session, config, unverified_update) == 100

    assert session.urls[-1].endswith("/editMessageText")
    edited_text = session.payloads[-1]["text"]
    assert "🚨 <b>ACT NOW</b>" in edited_text
    assert "👀 <b>MONITOR</b>" not in edited_text
    assert "League availability could not be refreshed" in edited_text
    persisted = json.loads((tmp_path / "telegram-state.json").read_text())
    assert persisted["threads"]["georgekittle"]["urgencyLevel"] == "act_now"


def test_low_severity_verified_action_buzzes_but_monitor_stays_silent(tmp_path) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session()
    today = replace(
        _alert("tweet:today", severity=2, player_name="Player One"),
        urgency=ActionUrgency("act_today", "act_today"),
    )
    monitor = replace(
        _alert("tweet:monitor", severity=2, player_name="Player Two"),
        urgency=ActionUrgency("monitor", "monitor"),
    )

    assert send_alert(session, config, today) == 100
    assert send_alert(session, config, monitor) == 101

    assert session.payloads[0]["disable_notification"] is False
    assert session.payloads[1]["disable_notification"] is True


def test_exact_completed_trade_burst_edits_one_message(
    tmp_path,
) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session()
    reports = [
        _trade_alert(
            "twitter:trade:1",
            (
                "Trade! The Patriots are sending WR Kayshon Boutte to the "
                "Texans for safety Jaylen Reed and a draft pick."
            ),
            severity=3,
        ),
        _trade_alert(
            "twitter:trade:2",
            (
                "It's a 28 7th rounder, the #Saints one, going in the trade. "
                "Kayshon Boutte heads to the #Texans."
            ),
            severity=3,
        ),
        _trade_alert(
            "rotowire:trade:3",
            "Dealt to Texans",
            body=(
                "The Patriots traded Boutte to the Texans for safety Jaylen "
                "Reed and a draft pick Monday."
            ),
            severity=3,
            source="rotowire",
        ),
        _trade_alert(
            "twitter:trade:4",
            (
                "Trade: The Patriots are sending WR Kayshon Boutte to the "
                "Texans in exchange for safety Jaylen Reed and a 2028 "
                "seventh-round pick."
            ),
        ),
        _trade_alert(
            "twitter:trade:5",
            (
                "Full terms: Texans get: WR Kayshon Boutte. Patriots get: S "
                "Jaylen Reed, 2028 seventh-round pick."
            ),
        ),
    ]

    assert [send_alert(session, config, report) for report in reports] == [
        100,
        100,
        100,
        100,
        100,
    ]
    assert [url.rsplit("/", 1)[-1] for url in session.urls] == [
        "sendMessage",
        "editMessageText",
        "editMessageText",
        "editMessageText",
        "editMessageText",
    ]
    assert all(
        payload.get("message_id") == 100 for payload in session.payloads[1:]
    )

    persisted = json.loads((tmp_path / "telegram-state.json").read_text())
    thread = persisted["threads"]["kayshonboutte"]
    assert thread["messageId"] == 100
    assert thread["severity"] == 3
    assert thread["eventStatus"] == "trade"
    assert thread["eventFactSignature"] == "trade:completed:to:HOU"
    assert len(persisted["alerts"]) == 1
    assert persisted["alerts"][0]["headline"] == reports[-1].item.headline


def test_trade_destination_refinement_edits_despite_severity_drift(tmp_path) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session()
    terse = _trade_alert(
        "twitter:trade:terse",
        "Kayshon Boutte has been traded.",
        severity=3,
    )
    destination = _trade_alert(
        "twitter:trade:destination",
        "The Patriots traded Kayshon Boutte to the Texans.",
        severity=4,
    )

    assert send_alert(session, config, terse) == 100
    assert send_alert(session, config, destination) == 100
    assert session.urls[-1].endswith("/editMessageText")

    persisted = json.loads((tmp_path / "telegram-state.json").read_text())
    thread = persisted["threads"]["kayshonboutte"]
    assert thread["severity"] == 4
    assert thread["eventFactSignature"] == "trade:completed:to:HOU"
    assert len(persisted["alerts"]) == 1


def test_trade_cancellation_and_different_destination_send_new_replies(tmp_path) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session()
    completed = _trade_alert(
        "twitter:trade:hou",
        "Patriots traded Kayshon Boutte to the Texans.",
    )
    cancelled = _trade_alert(
        "twitter:trade:cancelled",
        "The Kayshon Boutte trade to Houston was cancelled after a failed physical.",
        severity=4,
    )
    rerouted = _trade_alert(
        "twitter:trade:dal",
        "New deal: the Patriots traded Kayshon Boutte to the Cowboys.",
        severity=4,
    )

    assert send_alert(session, config, completed) == 100
    assert send_alert(session, config, cancelled) == 101
    assert send_alert(session, config, rerouted) == 102

    assert all(url.endswith("/sendMessage") for url in session.urls)
    assert session.payloads[1]["reply_parameters"]["message_id"] == 100
    assert session.payloads[2]["reply_parameters"]["message_id"] == 101
    persisted = json.loads((tmp_path / "telegram-state.json").read_text())
    assert len(persisted["alerts"]) == 3
    assert persisted["threads"]["kayshonboutte"]["eventFactSignature"] == (
        "trade:completed:to:DAL"
    )


def test_unresolved_explicit_trade_correction_sends_a_new_reply(tmp_path) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session()
    completed = _trade_alert(
        "twitter:trade:hou-before-correction",
        "The Patriots traded Kayshon Boutte to the Texans.",
    )
    correction = _trade_alert(
        "twitter:trade:pronoun-correction",
        (
            "Correction: Kayshon Boutte wasn't traded to the Texans; he was "
            "dealt to the Cowboys."
        ),
    )

    assert send_alert(session, config, completed) == 100
    assert send_alert(session, config, correction) == 101
    assert all(url.endswith("/sendMessage") for url in session.urls)
    assert session.payloads[1]["reply_parameters"]["message_id"] == 100

    persisted = json.loads((tmp_path / "telegram-state.json").read_text())
    assert len(persisted["alerts"]) == 2
    assert persisted["threads"]["kayshonboutte"]["eventFactSignature"] == (
        "trade:correction:to:?"
    )


def test_legacy_trade_thread_is_upgraded_in_place(tmp_path) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session()
    first = _trade_alert(
        "twitter:trade:legacy",
        "Patriots traded Kayshon Boutte to the Texans.",
    )
    assert send_alert(session, config, first) == 100

    path = tmp_path / "telegram-state.json"
    payload = json.loads(path.read_text())
    payload["threads"]["kayshonboutte"]["eventFactSignature"] = "unspecified"
    payload["alerts"][0]["eventFactSignature"] = "unspecified"
    path.write_text(json.dumps(payload))
    notify_module._TELEGRAM_STATES.clear()

    confirmation = _trade_alert(
        "rotowire:trade:legacy-confirmation",
        "Boutte was dealt to Houston for a 2028 draft pick.",
        severity=4,
        source="rotowire",
    )
    assert send_alert(session, config, confirmation) == 100
    assert session.urls[-1].endswith("/editMessageText")

    persisted = json.loads(path.read_text())
    assert persisted["threads"]["kayshonboutte"]["eventFactSignature"] == (
        "trade:completed:to:HOU"
    )
    assert len(persisted["alerts"]) == 1


def test_send_and_edit_payloads_stay_within_telegram_text_limit(tmp_path) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session()
    oversized = "49ers activated George Kittle from active/PUP. " + ("🏈" * 5000)
    first = _alert("tweet:long:1", headline=oversized, body=oversized)
    updated_text = (
        "49ers confirm George Kittle was activated and returned to practice. "
        + ("🏈" * 5000)
    )
    updated = _alert(
        "tweet:long:2",
        headline=updated_text,
        body=updated_text,
    )

    assert send_alert(session, config, first) == 100
    assert send_alert(session, config, updated) == 100

    assert session.urls[0].endswith("/sendMessage")
    assert session.urls[1].endswith("/editMessageText")
    for payload in session.payloads:
        assert _visible_units(payload["text"]) <= TELEGRAM_TEXT_LIMIT
        assert "Some details omitted to fit Telegram" in payload["text"]


def test_watson_starter_burst_collapses_across_labels_and_wording(
    tmp_path, monkeypatch
) -> None:
    notify_module._TELEGRAM_STATES.clear()
    clock = [1_000_000.0]
    monkeypatch.setattr(telegram_state_module.time, "time", lambda: clock[0])
    config = _config(tmp_path)
    session = Session()
    reports = [
        (
            "depth_chart",
            "The Browns have named Deshaun Watson their Week 1 starter, per source.",
            "twitter",
        ),
        (
            "depth_chart",
            "Browns pick Deshaun Watson as QB1 for Week 1. His experience is key.",
            "twitter",
        ),
        (
            "depth_chart",
            "Browns notified both QBs of the decision to start Deshaun Watson for Week 1.",
            "twitter",
        ),
        (
            "other",
            "More about the Browns naming Deshaun Watson, not Sanders, as their QB1.",
            "twitter",
        ),
        (
            "depth_chart",
            "Some perspective on the Browns naming Deshaun Watson starting quarterback.",
            "twitter",
        ),
        (
            "other",
            "Deshaun Watson starts Week 1; the team doesn't anticipate a week-to-week thing.",
            "twitter",
        ),
        (
            "depth_chart",
            "The Browns named Watson their starting quarterback.",
            "rotowire",
        ),
    ]

    for index, (event_type, headline, source) in enumerate(reports):
        if index:
            clock[0] += 20 * 60
        alert = _alert(
            f"tweet:watson:{index}",
            headline=headline,
            event_type=event_type,
            severity=4 if index == 5 else 3,
            player_name="Deshaun Watson",
            source=source,
        )
        assert send_alert(session, config, alert) == 100

    assert session.urls[0].endswith("/sendMessage")
    assert all(url.endswith("/editMessageText") for url in session.urls[1:])
    persisted = json.loads((tmp_path / "telegram-state.json").read_text())
    assert len(persisted["alerts"]) == 1
    assert persisted["threads"]["deshaunwatson"]["eventType"] == "depth_chart"
    assert persisted["threads"]["deshaunwatson"]["eventStatus"] == "role_starter"
    assert persisted["threads"]["deshaunwatson"]["eventFactSignature"] == "role:starter"

    clock[0] += 20 * 60
    reversal = _alert(
        "tweet:watson:benched",
        headline="The Browns benched Deshaun Watson and named Sanders the starter.",
        event_type="depth_chart",
        severity=3,
        player_name="Deshaun Watson",
    )
    assert send_alert(session, config, reversal) == 101
    assert session.urls[-1].endswith("/sendMessage")
    assert session.payloads[-1]["reply_parameters"]["message_id"] == 100


def test_legacy_watson_role_thread_is_migrated_before_first_repeat(tmp_path) -> None:
    path = tmp_path / "telegram-state.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "threads": {
                    "deshaunwatson": {
                        "messageId": 94,
                        "sentAt": time.time(),
                        "player": "Deshaun Watson",
                        "token": "legacy-token",
                        "eventType": "depth_chart",
                        "severity": 3,
                        "eventStatus": "depth_chart",
                        "eventFactSignature": "season_week:1|week_to_week",
                        "latestHeadline": (
                            "Deshaun Watson starts Week 1; the team doesn't anticipate "
                            "this being a week-to-week thing."
                        ),
                    }
                },
                "alerts": [
                    {
                        "messageId": 94,
                        "token": "legacy-token",
                        "eventType": "depth_chart",
                        "severity": 3,
                        "eventStatus": "depth_chart",
                        "eventFactSignature": "season_week:1|week_to_week",
                        "headline": "Deshaun Watson starts Week 1",
                    }
                ],
            }
        )
    )

    state = TelegramState(path, thread_hours=168)
    repeat = _alert(
        "tweet:watson:repeat",
        headline="Browns confirm Deshaun Watson as their QB1.",
        event_type="other",
        severity=4,
        player_name="Deshaun Watson",
    )

    target = state.coalescing_target(repeat)
    assert target is not None
    assert target.message_id == 94
    persisted = json.loads(path.read_text())
    thread = persisted["threads"]["deshaunwatson"]
    assert thread["eventStatus"] == "role_starter"
    assert thread["eventFactSignature"] == "role:starter"
    assert persisted["alerts"][0]["eventStatus"] == "role_starter"


def test_live_watson_state_recovers_role_target_behind_generic_commentary(
    tmp_path,
) -> None:
    path = tmp_path / "telegram-state.json"
    sent_at = time.time()
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "threads": {
                    "deshaunwatson": {
                        "messageId": 95,
                        "sentAt": sent_at + 60,
                        "player": "Deshaun Watson",
                        "token": "commentary-token",
                        "eventType": "other",
                        "severity": 3,
                        "eventStatus": "other",
                        "eventFactSignature": "unspecified",
                        "latestHeadline": (
                            "Todd Monken discusses the full body of work in the QB "
                            "battle between Deshaun Watson and Shedeur Sanders"
                        ),
                    }
                },
                "alerts": [
                    {
                        "messageId": 94,
                        "sentAt": sent_at,
                        "player": "Deshaun Watson",
                        "token": "role-token",
                        "eventType": "depth_chart",
                        "severity": 3,
                        "eventStatus": "depth_chart",
                        "eventFactSignature": "season_week:1|week_to_week",
                        "headline": (
                            "Deshaun Watson starts Week 1; the team doesn't "
                            "anticipate this being a week-to-week thing."
                        ),
                    },
                    {
                        "messageId": 95,
                        "sentAt": sent_at + 60,
                        "player": "Deshaun Watson",
                        "token": "commentary-token",
                        "eventType": "other",
                        "severity": 3,
                        "eventStatus": "other",
                        "eventFactSignature": "unspecified",
                        "headline": "Todd Monken discusses the QB battle",
                    },
                ],
            }
        )
    )

    state = TelegramState(path, thread_hours=168)
    repeat = _alert(
        "tweet:watson:post-deploy",
        headline="The Browns confirm Deshaun Watson as their QB1.",
        event_type="depth_chart",
        severity=4,
        player_name="Deshaun Watson",
    )

    target = state.coalescing_target(repeat)
    assert target is not None
    assert target.message_id == 94
    persisted = json.loads(path.read_text())
    thread = persisted["threads"]["deshaunwatson"]
    assert thread["messageId"] == 94
    assert thread["eventStatus"] == "role_starter"
    assert persisted["roleMetadataMigration"] == 1


def test_past_qb_battle_commentary_is_not_an_unresolved_role_decision() -> None:
    ambiguous = _alert(
        "tweet:watson:battle",
        headline=(
            "After the earlier QB battle, Watson and Sanders discussed the Browns' decision."
        ),
        event_type="other",
        severity=3,
        player_name="Deshaun Watson",
    )

    assert semantic_event_type(ambiguous.item, "other") == "other"


def test_lower_severity_corroboration_edits_without_downgrading_original(tmp_path) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session()

    assert send_alert(session, config, _alert("tweet:high", severity=4)) == 100
    assert send_alert(
        session,
        config,
        _alert("tweet:lower", headline="49ers confirm Kittle was activated", severity=3),
    ) == 100

    assert session.urls[1].endswith("/editMessageText")
    assert "[4/5]" in session.payloads[1]["text"]
    persisted = json.loads((tmp_path / "telegram-state.json").read_text())
    assert persisted["threads"]["georgekittle"]["severity"] == 4


def test_severity_escalation_and_new_status_send_new_replies(tmp_path) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session()

    limited = _alert(
        "tweet:limited",
        headline="George Kittle was limited in practice",
        event_type="injury",
        severity=3,
    )
    escalated = _alert(
        "tweet:escalated",
        headline="George Kittle remains limited in practice",
        event_type="injury",
        severity=4,
    )
    ruled_out = _alert(
        "tweet:out",
        headline="George Kittle has been ruled out",
        event_type="injury",
        severity=4,
    )

    assert send_alert(session, config, limited) == 100
    assert send_alert(session, config, escalated) == 101
    assert send_alert(session, config, ruled_out) == 102

    assert all(url.endswith("/sendMessage") for url in session.urls)
    assert session.payloads[1]["reply_parameters"]["message_id"] == 100
    assert session.payloads[2]["reply_parameters"]["message_id"] == 101
    persisted = json.loads((tmp_path / "telegram-state.json").read_text())
    assert len(persisted["alerts"]) == 3
    assert persisted["threads"]["georgekittle"]["eventStatus"] == "inactive"


def test_reused_guid_questionable_to_ruled_out_has_distinct_feedback_targets(
    tmp_path,
) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session()
    questionable = _alert(
        "tweet:reused-status",
        headline="George Kittle injury update",
        body="George Kittle is questionable for Sunday.",
        event_type="injury",
        severity=3,
    )
    ruled_out = _alert(
        "tweet:reused-status",
        headline="George Kittle injury update",
        body="George Kittle has been ruled out for Sunday.",
        event_type="injury",
        severity=4,
    )

    assert send_alert(session, config, questionable) == 100
    state = notify_module.telegram_state(config)
    first_token = alert_token(questionable.item)
    assert state.record_feedback(first_token, "useful") is True
    assert send_alert(session, config, ruled_out) == 101

    second_token = alert_token(ruled_out.item)
    assert first_token != second_token
    assert all(url.endswith("/sendMessage") for url in session.urls)
    assert session.payloads[0]["reply_markup"]["inline_keyboard"][0][0][
        "callback_data"
    ] == f"feedback:{first_token}:useful"
    assert session.payloads[1]["reply_markup"]["inline_keyboard"][0][0][
        "callback_data"
    ] == f"feedback:{second_token}:useful"
    assert state.record_feedback(second_token, "wrong") is True
    assert state.feedback_verdict(first_token) == "useful"
    assert state.feedback_verdict(second_token) == "wrong"
    persisted = json.loads((tmp_path / "telegram-state.json").read_text())
    assert first_token in persisted["feedbackTargets"]
    assert second_token in persisted["feedbackTargets"]
    assert persisted["feedbackTargets"][first_token]["severity"] == 3
    assert persisted["feedbackTargets"][second_token]["severity"] == 4


def test_distinct_condition_with_same_event_status_and_severity_sends_new_reply(
    tmp_path,
) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session()

    ankle = _alert(
        "tweet:reused-condition",
        headline="George Kittle injury update",
        body="George Kittle is dealing with an ankle injury.",
        event_type="injury",
        severity=4,
    )
    concussion = _alert(
        "tweet:reused-condition",
        headline="George Kittle injury update",
        body="George Kittle is now in the concussion protocol.",
        event_type="injury",
        severity=4,
    )

    assert send_alert(session, config, ankle) == 100
    state = notify_module.telegram_state(config)
    ankle_token = alert_token(ankle.item)
    assert state.record_feedback(ankle_token, "useful") is True
    assert send_alert(session, config, concussion) == 101

    concussion_token = alert_token(concussion.item)
    assert ankle_token != concussion_token
    assert all(url.endswith("/sendMessage") for url in session.urls)
    assert session.payloads[1]["reply_parameters"]["message_id"] == 100
    assert session.payloads[0]["reply_markup"]["inline_keyboard"][0][0][
        "callback_data"
    ] == f"feedback:{ankle_token}:useful"
    assert session.payloads[1]["reply_markup"]["inline_keyboard"][0][0][
        "callback_data"
    ] == f"feedback:{concussion_token}:useful"
    assert state.record_feedback(concussion_token, "wrong") is True
    assert state.feedback_verdict(ankle_token) == "useful"
    assert state.feedback_verdict(concussion_token) == "wrong"
    persisted = json.loads((tmp_path / "telegram-state.json").read_text())
    assert len(persisted["alerts"]) == 2
    assert persisted["threads"]["georgekittle"]["eventFactSignature"] == "concussion"
    assert ankle_token in persisted["feedbackTargets"]
    assert concussion_token in persisted["feedbackTargets"]


def test_same_ruled_out_fact_edits_even_when_sources_phrase_it_differently(
    tmp_path,
) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session()

    assert send_alert(
        session,
        config,
        _alert(
            "tweet:out",
            headline="The 49ers ruled out George Kittle",
            event_type="injury",
            severity=4,
        ),
    ) == 100
    assert send_alert(
        session,
        config,
        _alert(
            "rotowire:out",
            headline="Team confirms George Kittle will not play",
            event_type="injury",
            severity=4,
        ),
    ) == 100

    assert session.urls[1].endswith("/editMessageText")


def test_successful_telegram_edit_is_retryable_when_state_commit_fails(
    tmp_path, monkeypatch
) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session(not_modified_edit_numbers={2})

    assert send_alert(session, config, _alert("tweet:before")) == 100
    state = notify_module.telegram_state(config)
    real_save = state._save_locked
    save_attempts = 0

    def flaky_save() -> bool:
        nonlocal save_attempts
        save_attempts += 1
        return False if save_attempts == 1 else real_save()

    monkeypatch.setattr(state, "_save_locked", flaky_save)

    updated = _alert(
        "tweet:after",
        headline="49ers confirm George Kittle was activated",
    )
    assert send_alert(session, config, updated) is None

    assert [url.rsplit("/", 1)[-1] for url in session.urls] == [
        "sendMessage",
        "editMessageText",
    ]
    persisted = json.loads((tmp_path / "telegram-state.json").read_text())
    assert persisted["threads"]["georgekittle"]["token"] == alert_token(
        _alert("tweet:before").item
    )
    assert persisted["alerts"][0]["headline"] == _alert("tweet:before").item.headline
    assert state.coalescing_target(updated) is not None

    assert send_alert(session, config, updated) == 100
    assert [url.rsplit("/", 1)[-1] for url in session.urls] == [
        "sendMessage",
        "editMessageText",
        "editMessageText",
    ]
    persisted = json.loads((tmp_path / "telegram-state.json").read_text())
    assert persisted["threads"]["georgekittle"]["token"] == alert_token(
        updated.item
    )
    assert persisted["alerts"][0]["headline"] == updated.item.headline


def test_successful_telegram_send_is_retryable_when_state_commit_fails(
    tmp_path, monkeypatch
) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session()
    state = notify_module.telegram_state(config)
    real_save = state._save_locked
    save_attempts = 0

    def flaky_save() -> bool:
        nonlocal save_attempts
        save_attempts += 1
        return False if save_attempts == 1 else real_save()

    monkeypatch.setattr(state, "_save_locked", flaky_save)
    alert = _alert("tweet:initial-state-failure")

    assert send_alert(session, config, alert) is None
    assert [url.rsplit("/", 1)[-1] for url in session.urls] == ["sendMessage"]
    assert state.previous_message_id(alert.item.player_name) is None
    assert not (tmp_path / "telegram-state.json").exists()

    assert send_alert(session, config, alert) == 101
    assert [url.rsplit("/", 1)[-1] for url in session.urls] == [
        "sendMessage",
        "sendMessage",
    ]
    persisted = json.loads((tmp_path / "telegram-state.json").read_text())
    assert persisted["threads"]["georgekittle"]["messageId"] == 101
    assert persisted["threads"]["georgekittle"]["token"] == alert_token(alert.item)
    assert len(persisted["alerts"]) == 1


def test_edit_failure_falls_back_to_normal_send(tmp_path) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session(failed_edits=1)

    assert send_alert(session, config, _alert("tweet:first")) == 100
    corroboration = _alert(
        "tweet:second",
        headline="49ers confirm George Kittle remains activated",
    )
    assert send_alert(session, config, corroboration) == 101

    assert [url.rsplit("/", 1)[-1] for url in session.urls] == [
        "sendMessage",
        "editMessageText",
        "sendMessage",
    ]
    fallback = session.payloads[2]
    assert fallback["reply_parameters"] == {
        "message_id": 100,
        "allow_sending_without_reply": True,
    }
    assert "reply_markup" in fallback
    assert notify_module.telegram_state(config).previous_message_id("George Kittle") == 101
    persisted = json.loads((tmp_path / "telegram-state.json").read_text())
    assert len(persisted["alerts"]) == 2


def test_legacy_thread_without_event_metadata_fails_safe_to_new_send(tmp_path) -> None:
    state_path = tmp_path / "telegram-state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "updateOffset": 0,
                "threads": {
                    "georgekittle": {
                        "messageId": 77,
                        "sentAt": time.time(),
                        "player": "George Kittle",
                    }
                },
                "alerts": [],
                "feedback": {},
                "lastDigestDate": "",
                "lastTelegramSuccess": 0,
            }
        )
    )
    notify_module._TELEGRAM_STATES.clear()
    session = Session()

    assert send_alert(session, _config(tmp_path), _alert("tweet:new")) == 100

    assert len(session.urls) == 1
    assert session.urls[0].endswith("/sendMessage")
    assert session.payloads[0]["reply_parameters"]["message_id"] == 77


def test_coalescing_window_is_24_hours_from_original_send(tmp_path) -> None:
    state = TelegramState(tmp_path / "telegram-state.json")
    alert = _alert("tweet:window")
    state.record_sent(alert, 42)
    persisted = json.loads((tmp_path / "telegram-state.json").read_text())
    sent_at = persisted["threads"]["georgekittle"]["sentAt"]

    assert state.coalescing_target(alert, now=sent_at + (24 * 60 * 60) - 1) is not None
    assert state.coalescing_target(alert, now=sent_at + (24 * 60 * 60)) is None


def test_same_ir_phase_edits_even_when_later_report_adds_diagnosis(tmp_path) -> None:
    state = TelegramState(tmp_path / "telegram-state.json")
    first = _alert(
        "tweet:ir:first",
        player_name="Jordyn Tyson",
        headline="Jordyn Tyson was placed on injured reserve",
        event_type="injury",
        severity=5,
    )
    state.record_sent(first, 42)
    detail = _alert(
        "tweet:ir:detail",
        player_name="Jordyn Tyson",
        headline="Jordyn Tyson is on injured reserve after knee surgery",
        event_type="injury",
        severity=5,
    )

    assert state.coalescing_target(detail).message_id == 42


def test_same_suspension_phase_edits_but_urgency_escalation_is_new(tmp_path) -> None:
    state = TelegramState(tmp_path / "telegram-state.json")
    first = replace(
        _alert(
            "tweet:suspension:first",
            player_name="Example Player",
            headline="Example Player was suspended",
            event_type="suspension",
            severity=4,
        ),
        urgency=ActionUrgency(rule_level="monitor", level="monitor"),
    )
    state.record_sent(first, 42)
    detail = replace(
        _alert(
            "tweet:suspension:detail",
            player_name="Example Player",
            headline="Example Player's suspension will last four games",
            event_type="suspension",
            severity=4,
        ),
        urgency=ActionUrgency(rule_level="monitor", level="monitor"),
    )
    escalated = replace(
        detail,
        urgency=ActionUrgency(rule_level="act_now", level="act_now"),
    )

    assert state.coalescing_target(detail).message_id == 42
    assert state.coalescing_target(escalated) is None


def test_dry_run_does_not_create_telegram_state(tmp_path) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path, dry_run=True)

    assert send_alert(Session(), config, _alert("tweet:dry")) == -1
    assert not (tmp_path / "telegram-state.json").exists()


def test_alert_omits_dead_feedback_buttons_when_controls_are_disabled(tmp_path) -> None:
    notify_module._TELEGRAM_STATES.clear()
    session = Session()

    assert send_alert(session, _config(tmp_path, controls=False), _alert("tweet:plain")) == 100
    assert "reply_markup" not in session.payloads[0]


def test_control_routes_commands_and_persists_feedback(tmp_path) -> None:
    config = _config(tmp_path)
    state = TelegramState(tmp_path / "telegram-state.json")
    alert = _alert("tweet:feedback")
    state.record_sent(alert, 77)
    feedback = Mock(return_value=True)
    control = TelegramControl(
        config,
        state,
        status_provider=lambda: "status text",
        player_provider=lambda query: f"player {query}",
        search_provider=lambda query: f"news {query}",
        feedback_provider=feedback,
    )
    replies: list[str] = []
    control._reply = lambda text: replies.append(text) or 1

    for text in ("/status", "/player Kittle", "/news active PUP", "/digest"):
        control._handle_update({"message": {"chat": {"id": 123}, "text": text}})

    assert replies[0] == "status text"
    assert replies[1] == "player Kittle"
    assert replies[2] == "news active PUP"
    assert "Daily fantasy action digest" in replies[3]

    control._session = SimpleNamespace(post=Mock(return_value=Response()))
    token = alert_token(alert.item)
    control._handle_callback(
        {
            "id": "callback-1",
            "data": f"feedback:{token}:useful",
            "message": {"chat": {"id": 123}, "message_id": 77},
        }
    )
    assert state.feedback_verdict(token) == "useful"
    feedback.assert_called_once_with(token, "useful")


def test_expired_scheduled_report_is_not_sent(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    state = TelegramState(tmp_path / "telegram-state.json")
    key = "waiver:espn:1:expired"
    assert state.register_scheduled_report(
        key,
        kind="waiver_report",
        parts=("expired waiver report",),
        notify_first=True,
    )
    control = TelegramControl(
        config,
        state,
        status_provider=lambda: "status",
        player_provider=lambda query: query,
    )
    send = Mock(return_value=100)
    monkeypatch.setattr("notifier.telegram_control.send_plain", send)
    report = ScheduledReport(
        key=key,
        kind="waiver_report",
        parts=("expired waiver report",),
        notify_first=True,
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )

    assert control._deliver_registered_report(report) is False
    send.assert_not_called()
    assert state.next_scheduled_report_part(key) == (
        0,
        "expired waiver report",
        None,
        True,
    )
