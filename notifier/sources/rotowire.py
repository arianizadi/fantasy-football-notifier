"""RotoWire NFL news RSS.

The feed is hard-capped at 5 items regardless of any count parameter, so the
poll interval has to stay short enough that fewer than 5 items publish between
polls. fetch() reports when every item was new, which means older items may
have rolled off unseen.
"""

from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ElementTree
from datetime import datetime
from email.utils import parsedate_to_datetime

import requests

from ..logging_utils import structured_log
from ..models import NewsItem

FEED_URL = "https://www.rotowire.com/rss/news.php?sport=NFL"
FEED_CAPACITY = 5
REQUEST_TIMEOUT = 10
USER_AGENT = "fantasy-news-notifier/1.0 (personal fantasy league use)"

# RotoWire titles are "Player Name: Headline text".
TITLE_PATTERN = re.compile(r"^\s*(?P<player>[^:]{2,60}?)\s*:\s*(?P<headline>.+)$")
BOILERPLATE = re.compile(r"\s*Visit RotoWire\.com for more analy.*$", re.IGNORECASE | re.DOTALL)


def _clean(value: str | None) -> str:
    if not value:
        return ""
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_published(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None


def parse_feed(xml_text: str) -> list[NewsItem]:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as error:
        structured_log(logging.WARNING, "rotowire.parse_failed", error=str(error))
        return []

    items: list[NewsItem] = []
    for element in root.findall(".//item"):
        title = _clean(element.findtext("title"))
        guid = _clean(element.findtext("guid")) or title
        if not title or not guid:
            continue

        match = TITLE_PATTERN.match(title)
        player_name = match.group("player") if match else ""
        headline = match.group("headline") if match else title

        body = BOILERPLATE.sub("", _clean(element.findtext("description"))).strip()

        items.append(
            NewsItem(
                source="rotowire",
                guid=f"rotowire:{guid}",
                player_name=player_name,
                headline=headline,
                body=body,
                url=_clean(element.findtext("link")).replace("//football", "/football"),
                published_at=_parse_published(element.findtext("pubDate")),
            )
        )
    return items


class FeedPoller:
    """Polls the feed using conditional GET.

    At a 15s interval this issues ~5,760 requests/day. Sending back the
    ETag/Last-Modified means the vast majority return 304 with no body, which
    is both faster for us and courteous to RotoWire.
    """

    def __init__(self) -> None:
        self._etag: str | None = None
        self._last_modified: str | None = None

    def fetch(self, session: requests.Session) -> tuple[list[NewsItem], bool, bool]:
        """Return (items, feed_was_full, was_modified)."""
        headers = {"User-Agent": USER_AGENT, "Accept": "application/rss+xml, text/xml"}
        if self._etag:
            headers["If-None-Match"] = self._etag
        if self._last_modified:
            headers["If-Modified-Since"] = self._last_modified

        response = session.get(FEED_URL, timeout=REQUEST_TIMEOUT, headers=headers)

        if response.status_code == 304:
            return [], False, False

        response.raise_for_status()
        self._etag = response.headers.get("ETag") or self._etag
        self._last_modified = response.headers.get("Last-Modified") or self._last_modified

        items = parse_feed(response.text)
        return items, len(items) >= FEED_CAPACITY, True


def fetch(session: requests.Session) -> tuple[list[NewsItem], bool]:
    """One-shot unconditional fetch, used by --prime."""
    response = session.get(
        FEED_URL,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, text/xml"},
    )
    response.raise_for_status()
    items = parse_feed(response.text)
    return items, len(items) >= FEED_CAPACITY
