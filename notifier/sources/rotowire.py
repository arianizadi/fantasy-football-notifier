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
from dataclasses import replace
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

import requests

from ..logging_utils import structured_log
from ..matcher import compact_key, normalize_name, player_name_in_text
from ..models import NewsItem
from .twitter import attributed_absence_subject

FEED_URL = "https://www.rotowire.com/rss/news.php?sport=NFL"
FEED_CAPACITY = 5
# 10s produced 20 read timeouts in two days; the feed is often just slow.
REQUEST_TIMEOUT = 20
USER_AGENT = "fantasy-news-notifier/1.0 (personal fantasy league use)"

# RotoWire titles are "Player Name: Headline text".
TITLE_PATTERN = re.compile(r"^\s*(?P<player>[^:]{2,60}?)\s*:\s*(?P<headline>.+)$")
BOILERPLATE = re.compile(r"\s*Visit RotoWire\.com for more analy.*$", re.IGNORECASE | re.DOTALL)

# Some RotoWire articles are filed under the player who benefits rather than
# the teammate whose injury created the opportunity.  For example, the title
# can be ``Mike Washington: Sees extra work after Jeanty injury`` while the
# report says Ashton Jeanty left practice.  The title prefix alone is therefore
# not always the medical subject.
#
# Keep this intentionally narrow.  Re-attribution requires both an explicit
# opportunity statement and a causal bridge; the known-player/depth checks in
# reattribute_beneficiary_report() provide the remaining proof.
OPPORTUNITY_CUE = re.compile(
    r"\b(?:see(?:s|ing)?|saw|get(?:s|ting)?|got|receive(?:s|d|ing)?|"
    r"handle(?:s|d|ing)?|take(?:s|n|ing)?|took|log(?:s|ged|ging)?|"
    r"earn(?:s|ed|ing)?)\b[^.;!?\n]{0,80}"
    r"\b(?:extra|more|most|increased|additional|larger|expanded)\b"
    r"[^.;!?\n]{0,50}\b(?:work|workload|carries|reps|touches|snaps|"
    r"opportunit(?:y|ies)|role)\b|"
    r"\b(?:expanded|larger|increased)\s+(?:role|workload)\b",
    re.IGNORECASE,
)
CAUSAL_OPPORTUNITY_CUE = re.compile(
    r"\b(?:after|following|once|because(?:\s+of)?|due\s+to|"
    r"in\s+place\s+of|replac(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)


def _preferred_records(player_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Canonical player records, preferring Sleeper's active duplicate."""
    records: dict[str, dict[str, Any]] = {}
    for raw in player_index.values():
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("full_name") or "").strip()
        key = compact_key(name)
        if not key:
            continue
        existing = records.get(key)
        raw_active = str(raw.get("status") or "").strip().casefold() == "active"
        existing_active = (
            str(existing.get("status") or "").strip().casefold() == "active"
            if existing is not None
            else False
        )
        if existing is None or (raw_active and not existing_active):
            records[key] = raw
    return records


def _depth_order(record: dict[str, Any]) -> int | None:
    try:
        return int(record.get("depth_chart_order"))
    except (TypeError, ValueError):
        return None


def _linked_opportunity(
    item: NewsItem,
    *,
    article_name: str,
    absence_subject: str,
    teammate_names: list[str],
) -> bool:
    """Prove the workload gain and starter absence belong to one clause."""
    article_tokens = normalize_name(article_name).split()
    article_surname = article_tokens[-1] if article_tokens else ""
    same_surname_count = sum(
        1
        for name in teammate_names
        if normalize_name(name).split()
        and normalize_name(name).split()[-1] == article_surname
    )

    for section, article_is_implicit in ((item.headline, True), (item.body, False)):
        for clause in re.split(r"[;.!?\n]", section):
            if not (
                OPPORTUNITY_CUE.search(clause)
                and CAUSAL_OPPORTUNITY_CUE.search(clause)
                and player_name_in_text(absence_subject, clause)
            ):
                continue
            article_is_named = player_name_in_text(article_name, clause)
            if (
                not article_is_named
                and article_surname
                and same_surname_count == 1
            ):
                article_is_named = player_name_in_text(article_surname, clause)
            if article_is_implicit or article_is_named:
                return True
    return False


def reattribute_beneficiary_report(
    item: NewsItem,
    player_index: dict[str, Any],
) -> NewsItem:
    """Center a proven indirect injury report on the unavailable starter.

    The raw headline and body remain byte-for-byte unchanged so revision-aware
    dedupe still recognizes a report recorded before this attribution upgrade.
    The title player remains recoverable from RotoWire's player URL for the
    later depth-chart ``named in report`` annotation. If any deterministic
    proof is missing, return the original item unchanged.
    """
    if item.source != "rotowire" or not item.player_name or not player_index:
        return item

    report_text = f"{item.headline}. {item.body}".strip()
    if not (
        OPPORTUNITY_CUE.search(report_text)
        and CAUSAL_OPPORTUNITY_CUE.search(report_text)
    ):
        return item

    records = _preferred_records(player_index)
    article_record = records.get(compact_key(item.player_name))
    if article_record is None:
        return item
    team = str(article_record.get("team") or "").strip()
    position = str(article_record.get("position") or "").strip()
    article_order = _depth_order(article_record)
    if not team or not position or article_order is None:
        return item

    teammate_names = [
        str(record.get("full_name") or "").strip()
        for record in records.values()
        if str(record.get("team") or "").strip() == team
        and str(record.get("position") or "").strip() == position
    ]
    absence_subject = attributed_absence_subject(report_text, teammate_names)
    if (
        not absence_subject
        or compact_key(absence_subject) == compact_key(item.player_name)
    ):
        return item

    absence_record = records.get(compact_key(absence_subject))
    absence_order = _depth_order(absence_record or {})
    # The article player must actually sit behind the unavailable teammate.
    # This prevents an RB1 article from being re-centered on an RB3 who merely
    # appears in a separate injury sentence.
    if absence_order is None or absence_order >= article_order:
        return item

    article_name = str(article_record.get("full_name") or item.player_name).strip()
    if not _linked_opportunity(
        item,
        article_name=article_name,
        absence_subject=absence_subject,
        teammate_names=teammate_names,
    ):
        return item
    return replace(
        item,
        player_name=str(absence_record.get("full_name") or absence_subject).strip(),
    )


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
