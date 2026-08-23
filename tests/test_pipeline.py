import queue
from types import SimpleNamespace
from unittest.mock import Mock

import requests

from notifier.models import NewsItem
from notifier.pipeline import Notifier


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
    seen = SimpleNamespace(is_new=Mock(return_value=True))
    process = Mock(return_value=0)
    notifier = SimpleNamespace(
        _reload_roster_if_changed=Mock(),
        deliver_pending=Mock(return_value=0),
        poller=SimpleNamespace(fetch=Mock(side_effect=requests.RequestException("offline"))),
        session=object(),
        _tweet_queue=tweet_queue,
        seen=seen,
        _pool=object(),
        _process_items=process,
    )

    assert Notifier.poll_once(notifier) == 0
    process.assert_called_once_with([tweet], notifier._pool)
    assert tweet_queue.empty()
