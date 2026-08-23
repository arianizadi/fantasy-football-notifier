from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from notifier.classify import classify
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


def _response(payload: str) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": payload}}],
    }
    return response


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
