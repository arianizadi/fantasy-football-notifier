from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from notifier.sources.fantasypros_live import (
    LIVE_NEWS_REQUEST_BUCKET,
    FantasyProsLiveNews,
)


NOW = datetime(2026, 8, 26, 16, 30, tzinfo=timezone.utc)


def _payload() -> dict:
    return {
        "sport": "NFL",
        "count": 1,
        "items": [
            {
                "id": 604265,
                "created": "2026-08-26 16:27:00",
                "author": "Ari Koslow",
                "player_id": 16393,
                "team_id": "GB",
                "title": "Josh Jacobs: Packers preparing for possible suspension",
                "sport_id": "NFL",
                "categories": ["Rumors"],
                "link": "https://www.fantasypros.com/nfl/news/604265/item.php",
                "desc": (
                    "Packers GM Brian Gutekunst said they have been preparing "
                    "for the scenario of Josh Jacobs being suspended."
                ),
                "impact": (
                    "If he is suspended, MarShawn Lloyd would figure to lead "
                    "the backfield."
                ),
            }
        ],
    }


class FakeFantasyPros:
    enabled = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def status(self):
        return SimpleNamespace(request_cap=425)

    def get_json(self, path: str, **kwargs):
        self.calls.append((path, kwargs))
        return _payload()


def _source(tmp_path):
    provider = FakeFantasyPros()
    source = FantasyProsLiveNews(
        provider,  # type: ignore[arg-type]
        enabled=True,
        request_limit=240,
        request_reserve=75,
        state_dir=tmp_path,
        clock=lambda: NOW,
    )
    source.set_player_index(
        {
            "one": {
                "full_name": "Josh Jacobs",
                "position": "RB",
                "team": "GB",
            },
            "two": {
                "full_name": "MarShawn Lloyd",
                "position": "RB",
                "team": "GB",
            },
        }
    )
    return source, provider


def test_live_news_uses_shared_budget_and_keeps_title_subject(tmp_path) -> None:
    source, provider = _source(tmp_path)

    result = source.fetch()

    assert len(result.items) == 1
    item = result.items[0]
    assert item.source == "fantasypros"
    assert item.guid == "fantasypros:604265"
    assert item.player_name == "Josh Jacobs"
    assert item.subject_confident is True
    assert "MarShawn Lloyd" in item.body
    assert item.published_at == datetime(
        2026, 8, 26, 16, 27, tzinfo=timezone.utc
    )
    path, kwargs = provider.calls[0]
    assert path == "nfl/news"
    assert kwargs["params"] == {"limit": 100, "order_by": "created"}
    assert kwargs["request_ceiling"] == 350
    assert kwargs["request_bucket"] == LIVE_NEWS_REQUEST_BUCKET
    assert kwargs["request_bucket_limit"] == 240


def test_live_news_priming_marker_is_persistent(tmp_path) -> None:
    source, _provider = _source(tmp_path)
    assert source.initialized is False
    assert source.mark_initialized(fetched_at=NOW, item_count=100) is True
    assert source.initialized is True

    restarted, _provider = _source(tmp_path)
    assert restarted.initialized is True


def test_live_news_never_attributes_backup_named_only_in_impact(tmp_path) -> None:
    source, _provider = _source(tmp_path)
    source.set_player_index(
        {
            "two": {
                "full_name": "MarShawn Lloyd",
                "position": "RB",
                "team": "GB",
            }
        }
    )

    item = source.fetch().items[0]

    assert item.player_name == ""
    assert item.subject_confident is False


def test_disabled_live_news_accepts_a_smaller_shared_cap(tmp_path) -> None:
    provider = FakeFantasyPros()
    provider.enabled = False
    provider.status = lambda: SimpleNamespace(request_cap=50)  # type: ignore[method-assign]

    source = FantasyProsLiveNews(
        provider,  # type: ignore[arg-type]
        enabled=False,
        request_limit=240,
        request_reserve=75,
        state_dir=tmp_path,
    )

    assert source.enabled is False
