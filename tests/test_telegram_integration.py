from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import requests

import notifier.notify as notify_module
from notifier.models import Alert, Classification, NewsItem
from notifier.notify import send_alert
from notifier.telegram_control import TelegramControl
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
) -> Alert:
    item = NewsItem(
        source="twitter",
        guid=guid,
        player_name="George Kittle",
        headline=headline,
        body=body if body is not None else headline,
        url="https://x.com/example/status/1",
        published_at=None,
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


def test_coalescing_window_is_six_hours_from_original_send(tmp_path) -> None:
    state = TelegramState(tmp_path / "telegram-state.json")
    alert = _alert("tweet:window")
    state.record_sent(alert, 42)
    persisted = json.loads((tmp_path / "telegram-state.json").read_text())
    sent_at = persisted["threads"]["georgekittle"]["sentAt"]

    assert state.coalescing_target(alert, now=sent_at + (6 * 60 * 60) - 1) is not None
    assert state.coalescing_target(alert, now=sent_at + (6 * 60 * 60)) is None


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
