from __future__ import annotations

import queue
from datetime import datetime, timezone
from types import SimpleNamespace

from notifier.sources.rotowire import parse_feed
from notifier.sources.twitter import TwitterStream


def test_rotowire_feed_preserves_source_time_and_cleans_markup() -> None:
    xml = """
    <rss><channel><item>
      <title>George Kittle: Returns to practice</title>
      <guid>kittle-1</guid>
      <description><![CDATA[<b>Kittle</b> returned. Visit RotoWire.com for more analysis.]]></description>
      <link>https://www.rotowire.com//football/player.php?id=1</link>
      <pubDate>Sun, 23 Aug 2026 17:30:00 +0000</pubDate>
    </item></channel></rss>
    """

    item = parse_feed(xml)[0]

    assert item.player_name == "George Kittle"
    assert item.headline == "Returns to practice"
    assert item.body == "Kittle returned."
    assert item.published_at == datetime(2026, 8, 23, 17, 30, tzinfo=timezone.utc)
    assert item.url == "https://www.rotowire.com/football/player.php?id=1"


def test_twitter_payload_keeps_created_at_and_player_match() -> None:
    stream = TwitterStream("fake", queue.Queue())
    stream._names = SimpleNamespace(find=lambda _text: ["George Kittle"])
    payload = {
        "data": {
            "id": "42",
            "author_id": "7",
            "created_at": "2026-08-23T17:30:00.000Z",
            "text": "The 49ers activated George Kittle from active/PUP.",
        },
        "includes": {"users": [{"id": "7", "username": "Reporter"}]},
    }

    item = stream._to_items(payload)[0]

    assert item.guid == "twitter:42:George Kittle"
    assert item.player_name == "George Kittle"
    assert item.published_at == datetime(2026, 8, 23, 17, 30, tzinfo=timezone.utc)
    assert item.url == "https://x.com/Reporter/status/42"
    stream._session.close()
