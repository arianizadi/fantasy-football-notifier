from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from notifier.classify import classify
from notifier.config import DEFAULT_MODEL
from notifier.models import Classification, NewsItem, RosterSnapshot
from notifier.pipeline import Notifier


def _item(text: str) -> NewsItem:
    return NewsItem(
        source="twitter",
        guid="twitter:1:Player",
        player_name="Example Player",
        headline=text,
        body=text,
        url="https://x.com/example/status/1",
        published_at=None,
    )


def _config():
    return SimpleNamespace(
        openrouter_api_key="test-key",
        openrouter_model="test/model",
    )


def _response(payload: object) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": payload}}],
    }
    return response


MALFORMED_CONTENT = [
    pytest.param(None, id="missing-content"),
    pytest.param({}, id="object-content"),
    pytest.param([], id="list-content"),
    pytest.param("", id="empty-content"),
    pytest.param('{"event_type":"injury",', id="truncated-json"),
    pytest.param("null", id="json-null"),
    pytest.param("[]", id="json-list"),
    pytest.param('[{"severity":4}]', id="json-list-of-objects"),
    pytest.param('"not an object"', id="json-string"),
    pytest.param('"{}"', id="json-string-with-braces"),
    pytest.param("```json\n[{}]\n```", id="fenced-json-list"),
]


@pytest.mark.parametrize("content", MALFORMED_CONTENT)
def test_malformed_model_content_recovers_on_later_response(
    monkeypatch, content: object
) -> None:
    session = Mock()
    session.post.side_effect = [
        _response(content),
        _response(
            '{"event_type":"practice_report","direction":"neutral",'
            '"severity":2,"fantasy_impact":"Limited practice participation",'
            '"is_actionable":false}'
        ),
    ]
    sleep = Mock()
    monkeypatch.setattr("notifier.classify.time.sleep", sleep)

    result = classify(session, _config(), _item("Example Player was limited"))

    assert result.event_type == "practice_report"
    assert result.severity == 2
    assert result.is_actionable is False
    assert "error" not in result.raw
    assert session.post.call_count == 2
    sleep.assert_called_once_with(0.5)


@pytest.mark.parametrize("content", MALFORMED_CONTENT)
@pytest.mark.parametrize(
    ("headline", "event_type", "severity", "high_signal"),
    [
        ("Routine player update", "other", 3, False),
        ("Example Player suffered a torn ACL", "injury", 4, True),
    ],
)
def test_malformed_model_content_falls_back_after_bounded_retries(
    monkeypatch,
    content: object,
    headline: str,
    event_type: str,
    severity: int,
    high_signal: bool,
) -> None:
    session = Mock()
    session.post.return_value = _response(content)
    sleep = Mock()
    monkeypatch.setattr("notifier.classify.time.sleep", sleep)

    result = classify(session, _config(), _item(headline))

    assert result.event_type == event_type
    assert result.severity == severity
    assert result.is_actionable is True
    assert result.raw["error"] == "unparseable_response"
    assert result.raw["high_signal_floor"] is high_signal
    assert session.post.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [0.5, 1.0]


@pytest.mark.parametrize(
    "wrapper",
    ["{payload}", "```json\n{payload}\n```", "Classification: {payload}"],
)
def test_valid_json_objects_still_allow_fences_and_prose(wrapper: str) -> None:
    session = Mock()
    payload = (
        '{"event_type":"practice_report","severity":2,'
        '"fantasy_impact":"Limited practice participation",'
        '"is_actionable":false}'
    )
    session.post.return_value = _response(wrapper.format(payload=payload))

    result = classify(session, _config(), _item("Example Player was limited"))

    assert result.event_type == "practice_report"
    assert result.severity == 2
    assert "error" not in result.raw
    session.post.assert_called_once()


def test_default_model_request_preserves_fast_json_contract() -> None:
    session = Mock()
    session.post.return_value = _response(
        '{"event_type":"practice_report","direction":"neutral",'
        '"severity":2,"fantasy_impact":"Limited practice participation",'
        '"is_actionable":false}'
    )
    config = _config()
    config.openrouter_model = DEFAULT_MODEL

    result = classify(session, config, _item("Example Player was limited"))

    assert result.severity == 2
    session.post.assert_called_once()
    request = session.post.call_args
    assert request.args == ("https://openrouter.ai/api/v1/chat/completions",)
    assert request.kwargs["timeout"] == 20
    payload = request.kwargs["json"]
    assert payload["model"] == DEFAULT_MODEL
    assert payload["reasoning"] == {"enabled": False}
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == 400
    assert payload["temperature"] == 0
    assert payload["provider"] == {"sort": "throughput"}


def test_transient_model_failures_retry_with_bounded_backoff(monkeypatch) -> None:
    session = Mock()
    session.post.side_effect = [
        requests.Timeout("one"),
        requests.ConnectionError("two"),
        _response(
            '{"event_type":"injury","severity":4,'
            '"fantasy_impact":"Major availability concern",'
            '"is_actionable":true}'
        ),
    ]
    sleep = Mock()
    monkeypatch.setattr("notifier.classify.time.sleep", sleep)

    result = classify(session, _config(), _item("Example Player hurt his knee"))

    assert result.severity == 4
    assert result.event_type == "injury"
    assert session.post.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [0.5, 1.0]


def test_high_signal_floor_survives_total_model_outage(monkeypatch) -> None:
    session = Mock()
    session.post.side_effect = requests.Timeout("offline")
    monkeypatch.setattr("notifier.classify.time.sleep", Mock())

    result = classify(
        session,
        _config(),
        _item("Example Player was placed on IR and is out for the season"),
    )

    assert session.post.call_count == 3
    assert result.severity == 4
    assert result.event_type == "injury"
    assert result.raw["high_signal_floor"] is True


def test_high_signal_model_outage_still_passes_preseason_gate(monkeypatch) -> None:
    session = Mock()
    session.post.side_effect = requests.Timeout("offline")
    monkeypatch.setattr("notifier.classify.time.sleep", Mock())
    notifier = Notifier.__new__(Notifier)
    notifier.session = session
    notifier.config = _config()
    notifier.snapshot = RosterSnapshot(generated_at=None)
    notifier.depth = SimpleNamespace(team_context=Mock(return_value=None))

    alert = notifier._evaluate_preseason(
        _item("Example Player tore his ACL and is out for the season"),
        {"search_rank": 100},
    )

    assert alert is not None
    assert alert.classification.severity == 4
    assert alert.classification.event_type == "injury"


def test_release_keeps_event_label_without_universal_severity_floor(monkeypatch) -> None:
    session = Mock()
    session.post.side_effect = requests.Timeout("offline")
    monkeypatch.setattr("notifier.classify.time.sleep", Mock())

    fallback = classify(
        session,
        _config(),
        _item("Example Player was waived during final roster cuts"),
    )

    assert fallback.event_type == "release"
    assert fallback.severity == 3
    assert fallback.raw["high_signal_floor"] is False

    session = Mock()
    session.post.return_value = _response(
        '{"event_type":"release","direction":"negative","severity":1,'
        '"fantasy_impact":"No meaningful fantasy role changes",'
        '"is_actionable":false}'
    )

    classified = classify(
        session,
        _config(),
        _item("Example Player was waived during final roster cuts"),
    )

    assert classified.event_type == "release"
    assert classified.severity == 1


@pytest.mark.parametrize(
    "text",
    [
        "Example Player was arrested and charged with DUI",
        "Example Player could be suspended after an allegation",
        "Example Player faces a possible two-game suspension after an allegation",
        "Example Player was not suspended after the investigation",
    ],
)
def test_legal_speculation_is_not_a_high_signal_suspension_during_outage(
    monkeypatch,
    text: str,
) -> None:
    session = Mock()
    session.post.side_effect = requests.Timeout("offline")
    monkeypatch.setattr("notifier.classify.time.sleep", Mock())

    result = classify(session, _config(), _item(text))

    assert result.event_type == "other"
    assert result.severity == 3
    assert result.raw["high_signal_floor"] is False


@pytest.mark.parametrize(
    "text",
    [
        "The NFL suspended Example Player for two games",
        "Example Player was placed on the Commissioner Exempt List",
        "Example Player's two-game suspension was upheld",
    ],
)
def test_confirmed_suspension_keeps_high_signal_floor_during_outage(
    monkeypatch,
    text: str,
) -> None:
    session = Mock()
    session.post.side_effect = requests.Timeout("offline")
    monkeypatch.setattr("notifier.classify.time.sleep", Mock())

    result = classify(session, _config(), _item(text))

    assert result.event_type == "suspension"
    assert result.severity == 4
    assert result.raw["high_signal_floor"] is True


@pytest.mark.parametrize(
    ("severity", "should_alert"),
    [(2, False), (3, True)],
)
def test_preseason_gate_starts_at_three(monkeypatch, severity, should_alert) -> None:
    notifier = Notifier.__new__(Notifier)
    notifier.session = object()
    notifier.config = _config()
    notifier.snapshot = RosterSnapshot(generated_at=None)
    notifier.depth = SimpleNamespace(team_context=Mock(return_value=None))
    monkeypatch.setattr(
        "notifier.pipeline.classify",
        Mock(
            return_value=Classification(
                "usage",
                severity,
                "Role update",
                False,
                {"direction": "neutral"},
            )
        ),
    )

    alert = notifier._evaluate_preseason(
        _item("Example Player role update"),
        {"search_rank": 100},
    )

    assert (alert is not None) is should_alert


def test_pup_activation_model_outage_keeps_return_above_preseason_gate(
    monkeypatch,
) -> None:
    session = Mock()
    session.post.side_effect = requests.Timeout("offline")
    monkeypatch.setattr("notifier.classify.time.sleep", Mock())

    classification = classify(
        session,
        _config(),
        _item("49ers activated George Kittle from the active/PUP list"),
    )

    assert classification.event_type == "return"
    assert classification.severity == 4
    assert classification.raw["direction"] == "positive"


def test_nonretryable_model_rejection_stops_immediately(monkeypatch) -> None:
    response = Mock(status_code=401)
    session = Mock()
    session.post.side_effect = requests.HTTPError("unauthorized", response=response)
    sleep = Mock()
    monkeypatch.setattr("notifier.classify.time.sleep", sleep)

    result = classify(session, _config(), _item("Routine update"))

    assert session.post.call_count == 1
    sleep.assert_not_called()
    assert result.severity == 3
