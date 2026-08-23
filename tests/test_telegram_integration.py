from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

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


class Session:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def post(self, _url: str, *, json: dict, timeout) -> Response:
        del timeout
        self.payloads.append(json)
        return Response(100 + len(self.payloads) - 1)


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


def _alert(guid: str) -> Alert:
    item = NewsItem(
        source="twitter",
        guid=guid,
        player_name="George Kittle",
        headline="49ers activated George Kittle",
        body="49ers activated George Kittle",
        url="https://x.com/example/status/1",
        published_at=None,
    )
    return Alert(
        item=item,
        classification=Classification("return", 4, "Availability improved.", True, {}),
        tier="preseason",
    )


def test_alerts_have_feedback_and_form_a_rolling_player_reply_chain(tmp_path) -> None:
    notify_module._TELEGRAM_STATES.clear()
    config = _config(tmp_path)
    session = Session()

    assert send_alert(session, config, _alert("tweet:1")) == 100
    assert send_alert(session, config, _alert("tweet:2")) == 101

    first, second = session.payloads
    assert "reply_parameters" not in first
    assert first["reply_markup"]["inline_keyboard"][0][0]["callback_data"].startswith(
        "feedback:"
    )
    assert second["reply_parameters"] == {
        "message_id": 100,
        "allow_sending_without_reply": True,
    }
    state = notify_module.telegram_state(config)
    assert state.previous_message_id("George Kittle") == 101
    assert not (tmp_path / "sent-messages.json").exists()


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
    token = alert_token(alert.item.guid)
    control._handle_callback(
        {
            "id": "callback-1",
            "data": f"feedback:{token}:useful",
            "message": {"chat": {"id": 123}, "message_id": 77},
        }
    )
    assert state.feedback_verdict(token) == "useful"
    feedback.assert_called_once_with(token, "useful")
