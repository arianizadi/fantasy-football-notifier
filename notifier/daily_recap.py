"""Deterministic, source-backed daily fantasy news recaps.

The breaking-news path is intentionally fast and terse.  This module provides
the complementary once-a-day view: collapse repeated reports, sort the facts
that mattered, and render mobile-friendly Telegram HTML.  It accepts plain
``EventStore``-style row dictionaries so callers can decide how to query and
schedule delivery.  It never calls a model, a league provider, or the network.

Classifier summaries are deliberately not rendered.  Older prompts allowed
those summaries to contain roster instructions, while a recap cannot recheck
availability.  The only report prose shown here is the upstream headline/body;
the educational footer is fixed code keyed only by event type.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

TELEGRAM_TEXT_LIMIT = 4096
DEFAULT_TIMEZONE = "America/Los_Angeles"
MAX_BIG_NEWS_ITEMS = 30
MAX_SMALLER_MOVES_ITEMS = 20

BIG_TIERS = frozenset({"mine", "claimable"})
USEFUL_SMALL_EVENTS = frozenset(
    {
        "injury",
        "practice_report",
        "inactive",
        "return",
        "trade",
        "signing",
        "release",
        "suspension",
        "depth_chart",
        "usage",
    }
)
IGNORED_FEEDBACK = frozenset({"wrong", "noisy"})

EVENT_LABELS = {
    "injury": "INJURY",
    "practice_report": "PRACTICE",
    "inactive": "OUT / INACTIVE",
    "return": "RETURN",
    "trade": "TRADE",
    "signing": "SIGNING",
    "release": "RELEASE",
    "suspension": "SUSPENSION",
    "depth_chart": "DEPTH CHART",
    "usage": "ROLE / USAGE",
    "other": "UPDATE",
}

SEVERITY_ICONS = {
    1: "⚪",
    2: "🔵",
    3: "🟡",
    4: "🟠",
    5: "🔴",
}

SOURCE_LABELS = {
    "twitter": "X",
    "x": "X",
    "rotowire": "RotoWire",
    "espn": "ESPN",
    "sleeper": "Sleeper",
    "fantasypros": "FantasyPros",
}

_GENERIC_SUBJECTS = frozenset(
    {"", "league news", "league report", "unknown", "unknown player"}
)
_SPACE_RE = re.compile(r"\s+")
_KEY_RE = re.compile(r"[^a-z0-9]+")


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _visible_units(markup: str) -> int:
    """Return Telegram's post-HTML, UTF-16 message length."""
    parser = _VisibleTextParser()
    parser.feed(markup)
    parser.close()
    return len("".join(parser.parts).encode("utf-16-le")) // 2


def _plain(value: Any) -> str:
    """Normalize upstream text once without inventing or rephrasing facts."""
    if value is None:
        return ""
    return _SPACE_RE.sub(" ", html.unescape(str(value))).strip()


def _escape(value: Any) -> str:
    return html.escape(_plain(value), quote=False)


def _clip(value: str, limit: int) -> str:
    normalized = _plain(value)
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(1, limit - 1)].rstrip() + "…"


def _key(value: str) -> str:
    return _KEY_RE.sub("", _plain(value).casefold())


def _event_type(value: Any) -> str:
    normalized = _plain(value).casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "practice": "practice_report",
        "out": "inactive",
        "out_inactive": "inactive",
        "depthchart": "depth_chart",
        "role": "usage",
        "role_usage": "usage",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in EVENT_LABELS else "other"


def _severity(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if 1 <= parsed <= 5 else None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value is None or str(value).strip() == "":
        return None
    else:
        raw = str(value).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _subject_confident(row: Mapping[str, Any], player_name: str) -> bool:
    explicit = _optional_bool(row.get("subject_confident"))
    if explicit is not None:
        return explicit

    attribution = _plain(row.get("subject_attribution")).casefold()
    raw = row.get("raw")
    if not attribution and isinstance(raw, Mapping):
        attribution = _plain(raw.get("subject_attribution")).casefold()
    if attribution in {"ambiguous", "uncertain", "unknown"}:
        return False
    return player_name.casefold() not in _GENERIC_SUBJECTS


def _safe_url(value: str) -> str:
    raw = _plain(value)
    if not raw or len(raw) > 2048:
        return ""
    parsed = urlsplit(raw)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return ""
    return raw


def _source_label(source: str) -> str:
    normalized = _plain(source)
    return SOURCE_LABELS.get(normalized.casefold(), normalized or "Source")


@dataclass(frozen=True)
class SourceAttribution:
    source: str
    url: str
    reported_at: datetime


@dataclass(frozen=True)
class RecapItem:
    """One deduplicated player/event fact shown in the daily recap."""

    player_name: str
    event_type: str
    severity: int
    tier: str
    headline: str
    body: str
    reported_at: datetime
    subject_confident: bool
    attributions: tuple[SourceAttribution, ...]
    report_count: int = 1

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "RecapItem | None":
        severity = _severity(row.get("severity"))
        headline = _plain(row.get("headline"))
        body = _plain(row.get("body"))
        if severity is None or not (headline or body):
            return None

        # Published time is the source fact.  Received time is a conservative
        # fallback for feeds that omit it.
        reported_at = _timestamp(row.get("published_at")) or _timestamp(
            row.get("received_at")
        )
        if reported_at is None:
            return None

        player_name = _plain(row.get("player_name"))
        source = _plain(row.get("source"))
        return cls(
            player_name=player_name,
            event_type=_event_type(row.get("event_type")),
            severity=severity,
            tier=_plain(row.get("tier")).casefold() or "league",
            headline=headline or body,
            body=body,
            reported_at=reported_at,
            subject_confident=_subject_confident(row, player_name),
            attributions=(
                SourceAttribution(
                    source=source,
                    url=_safe_url(_plain(row.get("url"))),
                    reported_at=reported_at,
                ),
            ),
        )


@dataclass(frozen=True)
class DailyRecap:
    """Selected recap facts plus their already-safe Telegram HTML parts."""

    generated_at: datetime
    window_start: datetime
    big_news: tuple[RecapItem, ...]
    smaller_moves: tuple[RecapItem, ...]
    learn_note: str
    parts: tuple[str, ...]
    omitted_big_news: int = 0
    omitted_smaller_moves: int = 0


def _dedupe_key(item: RecapItem) -> tuple[str, str]:
    player = _key(item.player_name)
    if not player or not item.subject_confident:
        # Never merge two ambiguously attributed league reports merely because
        # their classifier labels both as ``other`` or ``injury``.
        player = "report:" + _key(item.headline)
    return player, item.event_type


def _tier_rank(tier: str) -> int:
    return {"mine": 5, "claimable": 4, "rival": 3, "league": 2, "preseason": 1}.get(
        tier, 0
    )


def _merge_group(group: Sequence[RecapItem]) -> RecapItem:
    # The newest source owns the displayed fact.  Severity and relevance are
    # the maxima across corroborating reports so a later terse wire update
    # cannot make an already important event disappear from the recap.
    newest = max(group, key=lambda item: item.reported_at)
    severity = max(item.severity for item in group)
    tier = max(group, key=lambda item: (_tier_rank(item.tier), item.reported_at)).tier

    unique: dict[tuple[str, str], SourceAttribution] = {}
    for item in sorted(group, key=lambda entry: entry.reported_at, reverse=True):
        for attribution in item.attributions:
            identity = (attribution.source.casefold(), attribution.url)
            unique.setdefault(identity, attribution)

    attributions = tuple(
        sorted(unique.values(), key=lambda entry: entry.reported_at, reverse=True)
    )
    return replace(
        newest,
        severity=severity,
        tier=tier,
        subject_confident=all(item.subject_confident for item in group),
        attributions=attributions,
        report_count=len(group),
    )


def _select_items(
    rows: Iterable[Mapping[str, Any]],
    *,
    now: datetime,
    hours: int,
) -> tuple[tuple[RecapItem, ...], tuple[RecapItem, ...]]:
    cutoff = now - timedelta(hours=hours)
    groups: dict[tuple[str, str], list[RecapItem]] = {}
    for row in rows:
        if _plain(row.get("feedback")).casefold() in IGNORED_FEEDBACK:
            continue
        item = RecapItem.from_row(row)
        if item is None or not (cutoff <= item.reported_at <= now):
            continue
        groups.setdefault(_dedupe_key(item), []).append(item)

    merged = [_merge_group(group) for group in groups.values()]
    big: list[RecapItem] = []
    smaller: list[RecapItem] = []
    for item in merged:
        if item.severity >= 4 or (
            item.severity >= 3 and item.tier in BIG_TIERS
        ):
            big.append(item)
        elif 2 <= item.severity <= 3 and (
            item.event_type in USEFUL_SMALL_EVENTS
            or item.tier in BIG_TIERS
        ):
            smaller.append(item)

    order = lambda item: (-item.severity, -item.reported_at.timestamp(), _key(item.player_name))
    big.sort(key=order)
    smaller.sort(key=order)
    return tuple(big), tuple(smaller)


def _time_label(value: datetime, zone: ZoneInfo) -> str:
    local = value.astimezone(zone)
    hour = local.strftime("%I").lstrip("0") or "0"
    return f"{local:%b} {local.day} · {hour}:{local:%M} {local:%p} {local:%Z}"


def _attribution_markup(
    attribution: SourceAttribution,
    zone: ZoneInfo,
) -> str:
    label = _escape(_source_label(attribution.source))
    if attribution.url:
        label = f'<a href="{html.escape(attribution.url, quote=True)}">{label}</a>'
    return f"{label} · {_escape(_time_label(attribution.reported_at, zone))}"


def _item_markup(item: RecapItem, zone: ZoneInfo) -> str:
    icon = SEVERITY_ICONS[item.severity]
    subject = (
        _escape(_clip(item.player_name, 80))
        if item.subject_confident and item.player_name
        else "League report"
    )
    event = EVENT_LABELS[item.event_type]
    headline = _escape(_clip(item.headline, 240))
    lines = [
        f"{icon} <b>{item.severity}/5 · {subject}</b>",
        f"<b>{_escape(event)}</b> · {headline}",
    ]

    body = _plain(item.body)
    headline_plain = _plain(item.headline)
    redundant = (
        not body
        or body == headline_plain
        or body.startswith(headline_plain[:120])
        or headline_plain.startswith(body[:120])
    )
    if not redundant:
        lines.append(f"<blockquote>{_escape(_clip(body, 280))}</blockquote>")

    displayed = item.attributions[:3]
    if displayed:
        sources = " · ".join(
            _attribution_markup(attribution, zone) for attribution in displayed
        )
        lines.append(f"🔗 {sources}")
    if item.report_count > 1:
        noun = "report" if item.report_count == 1 else "reports"
        lines.append(f"<i>{item.report_count} {noun} combined.</i>")
    if not item.subject_confident:
        lines.append(
            "⚠️ <i>Player attribution is unclear; no roster move is inferred.</i>"
        )
    return "\n".join(lines)


def _learn_note(items: Sequence[RecapItem]) -> str:
    events = {item.event_type for item in items}
    lessons: list[str] = []
    if events & {"injury", "inactive", "practice_report"}:
        lessons.append(
            "Injury news can open opportunity, but depth order alone does not "
            "guarantee snaps or touches."
        )
    if events & {"depth_chart", "usage", "signing", "trade", "release"}:
        lessons.append(
            "Role changes become meaningful when later reports confirm routes, "
            "carries, or playing time."
        )
    if "return" in events:
        lessons.append(
            "Active or returned status confirms availability, not a full workload."
        )
    if "suspension" in events:
        lessons.append(
            "Suspensions can create temporary roles that disappear when the starter returns."
        )
    if not lessons:
        return "A news headline is one data point; role and availability still need confirmation."
    return " ".join(lessons[:2])


def _header(now: datetime, hours: int, zone: ZoneInfo, *, continued: bool = False) -> str:
    local = now.astimezone(zone)
    hour = local.strftime("%I").lstrip("0") or "0"
    title = "📰 <b>DAILY FANTASY RECAP"
    if continued:
        title += " · CONTINUED"
    title += "</b>"
    return (
        f"{title}\n"
        f"Last {hours} hours · through {local:%b} {local.day}, "
        f"{hour}:{local:%M} {local:%p} {local:%Z}"
    )


def _pack_parts(
    *,
    now: datetime,
    hours: int,
    zone: ZoneInfo,
    sections: Sequence[tuple[str, Sequence[str]]],
    max_units: int,
) -> tuple[str, ...]:
    if max_units < 256:
        raise ValueError("max_units must be at least 256")

    parts: list[str] = []
    current = _header(now, hours, zone)

    for heading, blocks in sections:
        for index, block in enumerate(blocks):
            prefix = heading if index == 0 else ""
            additions = [value for value in (prefix, block) if value]
            candidate = "\n\n".join([current, *additions])
            if _visible_units(candidate) <= max_units:
                current = candidate
                continue

            # Logical item boundaries are never split.  With fields clipped by
            # ``_item_markup``, a single item comfortably fits Telegram's real
            # limit; this guard gives callers with a smaller test limit a clear
            # failure rather than emitting invalid or partial HTML.
            if current != _header(now, hours, zone):
                parts.append(current)
            current = "\n\n".join(
                [_header(now, hours, zone, continued=True), heading, block]
            )
            if _visible_units(current) > max_units:
                raise ValueError("one recap item exceeds max_units")

    if current and (not parts or current != _header(now, hours, zone)):
        parts.append(current)
    return tuple(parts)


def format_daily_recap(
    rows: Iterable[Mapping[str, Any]],
    *,
    now: datetime | None = None,
    hours: int = 24,
    timezone_name: str = DEFAULT_TIMEZONE,
    max_units: int = TELEGRAM_TEXT_LIMIT,
) -> DailyRecap:
    """Select and render an EventStore-style rolling recap.

    The returned ``parts`` are complete Telegram HTML messages.  They are
    split only between report blocks and each remains within ``max_units``
    visible UTF-16 units.
    """
    if hours < 1 or hours > 24 * 7:
        raise ValueError("hours must be between 1 and 168")
    generated_at = now or datetime.now(timezone.utc)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    generated_at = generated_at.astimezone(timezone.utc)
    zone = ZoneInfo(timezone_name)

    selected_big, selected_smaller = _select_items(
        rows,
        now=generated_at,
        hours=hours,
    )
    omitted_big = max(0, len(selected_big) - MAX_BIG_NEWS_ITEMS)
    omitted_smaller = max(0, len(selected_smaller) - MAX_SMALLER_MOVES_ITEMS)
    big = selected_big[:MAX_BIG_NEWS_ITEMS]
    smaller = selected_smaller[:MAX_SMALLER_MOVES_ITEMS]
    all_items = (*big, *smaller)
    learn = _learn_note(all_items)

    big_blocks = [_item_markup(item, zone) for item in big]
    smaller_blocks = [_item_markup(item, zone) for item in smaller]
    if not big_blocks:
        big_blocks = ["No major reports in this window."]
    if not smaller_blocks:
        smaller_blocks = ["No smaller fantasy-relevant moves in this window."]

    learn_block = f"🎓 <b>LEARN THE GAME</b>\n<i>{_escape(learn)}</i>"
    omitted_total = omitted_big + omitted_smaller
    overflow_blocks = (
        [
            f"<i>+ {omitted_total} more saved "
            f"{'report' if omitted_total == 1 else 'reports'}; use /news to search.</i>"
        ]
        if omitted_total
        else []
    )
    parts = _pack_parts(
        now=generated_at,
        hours=hours,
        zone=zone,
        sections=(
            ("🔥 <b>BIG NEWS</b>", big_blocks),
            ("🧩 <b>SMALLER MOVES</b>", smaller_blocks),
            ("", [*overflow_blocks, learn_block]),
        ),
        max_units=max_units,
    )
    return DailyRecap(
        generated_at=generated_at,
        window_start=generated_at - timedelta(hours=hours),
        big_news=big,
        smaller_moves=smaller,
        learn_note=learn,
        parts=parts,
        omitted_big_news=omitted_big,
        omitted_smaller_moves=omitted_smaller,
    )
