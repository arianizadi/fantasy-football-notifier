import queue
from types import SimpleNamespace
from unittest.mock import Mock

import requests

from notifier.models import NewsItem
from notifier.pipeline import Notifier


class InlinePool:
    def map(self, function, items):
        return [function(item) for item in items]


class SeenRecorder:
    def __init__(self) -> None:
        self.recorded = []
        self.saved = 0

    def is_new(self, item) -> bool:
        return True

    def record(self, item) -> None:
        self.recorded.append(item)

    def save(self) -> None:
        self.saved += 1


def test_tweets_are_processed_when_rotowire_is_unavailable() -> None:
    tweet = NewsItem(
        source="twitter",
        guid="twitter:1:Example Player",
        player_name="Example Player",
        headline="Example Player was ruled out",
        body="Example Player was ruled out",
        url="https://x.com/example/status/1",
        published_at=None,
    )
    tweet_queue = queue.Queue()
    tweet_queue.put(tweet)
    seen = SeenRecorder()
    evaluate = Mock(return_value=None)
    notifier = SimpleNamespace(
        _reload_roster_if_changed=Mock(),
        _refresh_trending=Mock(),
        poller=SimpleNamespace(fetch=Mock(side_effect=requests.RequestException("offline"))),
        session=object(),
        _tweet_queue=tweet_queue,
        seen=seen,
        _pool=InlinePool(),
        _evaluate=evaluate,
    )

    assert Notifier.poll_once(notifier) == 0
    evaluate.assert_called_once_with(tweet)
    assert seen.recorded == [tweet]
    assert seen.saved == 1
    assert tweet_queue.empty()
