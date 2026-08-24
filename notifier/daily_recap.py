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
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from .dedupe import transaction_teams
from .matcher import compact_key
from .models import LeagueRef, NewsItem, RosterPlayer, RosterSnapshot

TELEGRAM_TEXT_LIMIT = 4096
DEFAULT_TIMEZONE = "America/Los_Angeles"
MAX_BIG_NEWS_ITEMS = 30
MAX_SMALLER_MOVES_ITEMS = 20
MAX_TEAM_IMPACT_ITEMS = 12

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
FANTASY_POSITION_ROOMS = frozenset({"QB", "RB", "WR", "TE"})
ROOM_IMPACT_EVENTS = USEFUL_SMALL_EVENTS
BENCH_SLOTS = frozenset({"BE", "BN", "BENCH"})
RESERVE_SLOTS = frozenset(
    {
        "IR",
        "INJURED RESERVE",
        "ER",
        "ROOKIE",
        "RES",
        "RESERVE",
        "TAXI",
        "NA",
        "PUP",
    }
)
INACTIVE_SLOTS = frozenset({"NFL_INACTIVE"})
TEAM_ALIASES = {
    "JAC": "JAX",
    "LA": "LAR",
    "WSH": "WAS",
}

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
    team_hints: tuple[str, ...] = ()
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
        event_type = _event_type(row.get("event_type"))
        subject_confident = _subject_confident(row, player_name)
        report = NewsItem(
            source=source,
            guid=_plain(row.get("guid")),
            player_name=player_name,
            headline=headline,
            body=body,
            url=_plain(row.get("url")),
            published_at=reported_at,
            subject_confident=subject_confident,
        )
        return cls(
            player_name=player_name,
            event_type=event_type,
            severity=severity,
            tier=_plain(row.get("tier")).casefold() or "league",
            headline=headline or body,
            body=body,
            reported_at=reported_at,
            subject_confident=subject_confident,
            attributions=(
                SourceAttribution(
                    source=source,
                    url=_safe_url(_plain(row.get("url"))),
                    reported_at=reported_at,
                ),
            ),
            team_hints=transaction_teams(report, event_type),
        )


@dataclass(frozen=True)
class RosterImpact:
    """One report connected to the user's roster by deterministic facts."""

    item: RecapItem
    direct_players: tuple[RosterPlayer, ...] = ()
    related_players: tuple[RosterPlayer, ...] = ()


@dataclass(frozen=True)
class DailyRecap:
    """Selected recap facts plus their already-safe Telegram HTML parts."""

    generated_at: datetime
    window_start: datetime
    big_news: tuple[RecapItem, ...]
    smaller_moves: tuple[RecapItem, ...]
    learn_note: str
    parts: tuple[str, ...]
    team_impacts: tuple[RosterImpact, ...] = ()
    omitted_team_impacts: int = 0
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
    team_hints = tuple(
        dict.fromkeys(team for item in group for team in item.team_hints)
    )
    return replace(
        newest,
        severity=severity,
        tier=tier,
        subject_confident=all(item.subject_confident for item in group),
        attributions=attributions,
        team_hints=team_hints,
        report_count=len(group),
    )


def _item_order(item: RecapItem) -> tuple[int, float, str]:
    return (-item.severity, -item.reported_at.timestamp(), _key(item.player_name))


def _merge_items(
    rows: Iterable[Mapping[str, Any]],
    *,
    now: datetime,
    hours: int,
) -> tuple[RecapItem, ...]:
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
    merged.sort(key=_item_order)
    return tuple(merged)


def _select_items(
    items: Iterable[RecapItem],
    *,
    excluded: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[tuple[RecapItem, ...], tuple[RecapItem, ...]]:
    big: list[RecapItem] = []
    smaller: list[RecapItem] = []
    for item in items:
        if _dedupe_key(item) in excluded:
            continue
        if item.severity >= 4 or (
            item.severity >= 3 and item.tier in BIG_TIERS
        ):
            big.append(item)
        elif 2 <= item.severity <= 3 and (
            item.event_type in USEFUL_SMALL_EVENTS
            or item.tier in BIG_TIERS
        ):
            smaller.append(item)

    big.sort(key=_item_order)
    smaller.sort(key=_item_order)
    return tuple(big), tuple(smaller)


def _normalized_team(value: Any) -> str:
    team = _plain(value).upper()
    return TEAM_ALIASES.get(team, team)


def _drafted_roster(snapshot: RosterSnapshot | None) -> tuple[RosterPlayer, ...]:
    if snapshot is None:
        return ()
    drafted_keys = {league.key for league in snapshot.drafted_leagues()}
    return tuple(
        player
        for player in snapshot.mine()
        if player.league_key in drafted_keys
    )


def _player_records_by_name(
    player_index: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    ambiguous: set[str] = set()
    for candidate in (player_index or {}).values():
        if not isinstance(candidate, Mapping):
            continue
        team = _normalized_team(candidate.get("team"))
        position = _plain(candidate.get("position")).upper()
        # The source matcher is also limited to fantasy positions. Ignoring a
        # same-named defensive player prevents him from masking the WR/RB/etc.
        # record used for a fantasy position-room comparison.
        if position not in FANTASY_POSITION_ROOMS:
            continue
        key = compact_key(_plain(candidate.get("full_name")))
        if not key or key in ambiguous:
            continue
        current = records.get(key)
        if current is None:
            records[key] = candidate
            continue
        current_identity = (
            _normalized_team(current.get("team")),
            _plain(current.get("position")).upper(),
        )
        if current_identity != (team, position):
            # Two fantasy players with the same normalized name but different
            # rooms are genuinely ambiguous, so infer no teammate impact.
            records.pop(key, None)
            ambiguous.add(key)
    return records


def _select_roster_impacts(
    items: Sequence[RecapItem],
    *,
    roster_snapshot: RosterSnapshot | None,
    player_index: Mapping[str, Any] | None,
) -> tuple[RosterImpact, ...]:
    roster = _drafted_roster(roster_snapshot)
    if not roster:
        return ()

    roster_by_name: dict[str, list[RosterPlayer]] = {}
    for player in roster:
        key = compact_key(player.name)
        if key:
            roster_by_name.setdefault(key, []).append(player)
    records = _player_records_by_name(player_index)

    impacts: list[RosterImpact] = []
    for item in items:
        if not item.subject_confident or item.severity < 2:
            continue
        subject_key = compact_key(item.player_name)
        if not subject_key:
            continue

        direct = tuple(roster_by_name.get(subject_key, ()))
        related: tuple[RosterPlayer, ...] = ()
        if item.severity >= 3 and item.event_type in ROOM_IMPACT_EVENTS:
            record = records.get(subject_key)
            if record is not None:
                teams = {
                    team
                    for team in (
                        _normalized_team(record.get("team")),
                        *(_normalized_team(hint) for hint in item.team_hints),
                    )
                    if team
                }
                position = _plain(record.get("position")).upper()
                if teams and position in FANTASY_POSITION_ROOMS:
                    related = tuple(
                        player
                        for player in roster
                        if compact_key(player.name) != subject_key
                        and _normalized_team(player.pro_team) in teams
                        and _plain(player.position).upper() == position
                    )

        if direct or related:
            impacts.append(
                RosterImpact(
                    item=item,
                    direct_players=direct,
                    related_players=related,
                )
            )

    impacts.sort(
        key=lambda impact: (
            -impact.item.severity,
            0 if impact.direct_players else 1,
            -impact.item.reported_at.timestamp(),
            _key(impact.item.player_name),
        )
    )
    return tuple(impacts)


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


def _item_markup(
    item: RecapItem,
    zone: ZoneInfo,
    *,
    context_lines: Sequence[str] = (),
) -> str:
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

    lines.extend(context_lines)

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


def _league_labels(leagues: Sequence[LeagueRef]) -> dict[str, str]:
    """Return stable human labels without collapsing same-named leagues."""
    bases = [league.short_label for league in leagues]
    counts = Counter(base.casefold() for base in bases)
    labels: dict[str, str] = {}
    for league, base in zip(leagues, bases):
        if counts[base.casefold()] == 1:
            labels[league.key] = base
            continue
        provider = (league.provider or "league").upper()
        suffix = league.league_id[-4:] if league.league_id else ""
        discriminator = f"{provider} {suffix}".strip()
        labels[league.key] = f"{base} ({discriminator})"
    return labels


def _roster_role(player: RosterPlayer) -> str:
    slot = _plain(player.lineup_slot).upper()
    if slot in BENCH_SLOTS:
        return "bench"
    if slot in RESERVE_SLOTS:
        return "reserve"
    if slot in INACTIVE_SLOTS:
        return "inactive"
    if slot:
        return "starter"
    return "rostered"


def _fantasy_team_identity(
    player: RosterPlayer,
    snapshot: RosterSnapshot,
    labels: Mapping[str, str],
) -> str:
    league = snapshot.league(player.league_key)
    if league is None:
        return player.fantasy_team or "Your team"
    team_name = (
        _plain(league.my_team_name)
        or _plain(player.fantasy_team)
        or "Your team"
    )
    league_label = labels.get(league.key, league.short_label)
    if team_name.casefold() == league_label.casefold():
        return team_name
    return f"{team_name} ({league_label})"


def _group_roster_players(
    players: Sequence[RosterPlayer],
) -> tuple[tuple[RosterPlayer, ...], ...]:
    groups: dict[str, list[RosterPlayer]] = {}
    for player in players:
        key = compact_key(player.name)
        if key:
            groups.setdefault(key, []).append(player)
    return tuple(tuple(group) for group in groups.values())


def _player_league_context(
    players: Sequence[RosterPlayer],
    snapshot: RosterSnapshot,
    labels: Mapping[str, str],
) -> str:
    contexts = [
        f"{_fantasy_team_identity(player, snapshot, labels)}: {_roster_role(player)}"
        for player in players
    ]
    return "; ".join(dict.fromkeys(contexts))


def _impact_context_lines(
    impact: RosterImpact,
    snapshot: RosterSnapshot,
    labels: Mapping[str, str],
) -> tuple[str, ...]:
    lines: list[str] = []
    if impact.direct_players:
        context = _player_league_context(impact.direct_players, snapshot, labels)
        lines.append(f"🏠 <b>Your player</b> · {_escape(context)}")

    for group in _group_roster_players(impact.related_players):
        player = group[0]
        team = _normalized_team(player.pro_team)
        position = _plain(player.position).upper()
        context = _player_league_context(group, snapshot, labels)
        lines.append(
            f"👀 <b>May affect {_escape(_clip(player.name, 80))}</b> · "
            f"same {_escape(team)} {_escape(position)} room · {_escape(context)}"
        )
    return tuple(lines)


def _roster_summary_markup(snapshot: RosterSnapshot, zone: ZoneInfo) -> str:
    leagues = snapshot.drafted_leagues()
    if not leagues:
        return (
            "No drafted roster is available yet. Team matching starts "
            "automatically after your draft."
        )

    labels = _league_labels(leagues)
    lines = ["<b>ROSTERS TRACKED</b>"]
    for league in leagues:
        players = snapshot.mine(league.key)
        team_name = _plain(league.my_team_name) or "Your team"
        identity = team_name
        league_label = labels.get(league.key, league.short_label)
        if team_name.casefold() != league_label.casefold():
            identity = f"{team_name} · {league_label}"
        count = len(players)
        noun = "player" if count == 1 else "players"
        lines.append(f"• <b>{_escape(identity)}</b> · {count} {noun}")
    refreshed_at = _timestamp(snapshot.generated_at)
    if refreshed_at is not None:
        lines.append(
            f"<i>Roster refreshed {_escape(_time_label(refreshed_at, zone))}</i>"
        )
    return "\n".join(lines)


def _impact_markup(
    impact: RosterImpact,
    zone: ZoneInfo,
    snapshot: RosterSnapshot,
    labels: Mapping[str, str],
) -> str:
    return _item_markup(
        impact.item,
        zone,
        context_lines=_impact_context_lines(impact, snapshot, labels),
    )


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
    roster_snapshot: RosterSnapshot | None = None,
    player_index: Mapping[str, Any] | None = None,
) -> DailyRecap:
    """Select and render an EventStore-style rolling recap.

    The returned ``parts`` are complete Telegram HTML messages.  They are
    split only between report blocks and each remains within ``max_units``
    visible UTF-16 units. Optional roster/player-index inputs are treated as
    read-only snapshots; this formatter still performs no provider requests.
    """
    if hours < 1 or hours > 24 * 7:
        raise ValueError("hours must be between 1 and 168")
    generated_at = now or datetime.now(timezone.utc)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    generated_at = generated_at.astimezone(timezone.utc)
    zone = ZoneInfo(timezone_name)

    merged = _merge_items(
        rows,
        now=generated_at,
        hours=hours,
    )
    selected_impacts = _select_roster_impacts(
        merged,
        roster_snapshot=roster_snapshot,
        player_index=player_index,
    )
    impact_keys = frozenset(_dedupe_key(impact.item) for impact in selected_impacts)
    selected_big, selected_smaller = _select_items(
        merged,
        excluded=impact_keys,
    )
    omitted_impacts = max(0, len(selected_impacts) - MAX_TEAM_IMPACT_ITEMS)
    omitted_big = max(0, len(selected_big) - MAX_BIG_NEWS_ITEMS)
    omitted_smaller = max(0, len(selected_smaller) - MAX_SMALLER_MOVES_ITEMS)
    impacts = selected_impacts[:MAX_TEAM_IMPACT_ITEMS]
    big = selected_big[:MAX_BIG_NEWS_ITEMS]
    smaller = selected_smaller[:MAX_SMALLER_MOVES_ITEMS]
    all_items = tuple(impact.item for impact in impacts) + big + smaller
    learn = _learn_note(all_items)

    team_section: tuple[tuple[str, Sequence[str]], ...] = ()
    if roster_snapshot is not None:
        leagues = roster_snapshot.drafted_leagues()
        labels = _league_labels(leagues)
        impact_blocks = [
            _impact_markup(impact, zone, roster_snapshot, labels)
            for impact in impacts
        ]
        if leagues and not impact_blocks:
            impact_blocks = [
                "No saved reports directly affecting your players or their "
                "position rooms in this window."
            ]
        if omitted_impacts:
            noun = "report" if omitted_impacts == 1 else "reports"
            impact_blocks.append(
                f"<i>+ {omitted_impacts} more team-impact {noun}; "
                "use /news to search.</i>"
            )
        team_section = (
            (
                "🏈 <b>YOUR TEAM IMPACT</b>",
                [_roster_summary_markup(roster_snapshot, zone), *impact_blocks],
            ),
        )

    big_blocks = [_item_markup(item, zone) for item in big]
    smaller_blocks = [_item_markup(item, zone) for item in smaller]
    if not big_blocks:
        big_blocks = [
            (
                "No additional major reports outside Your Team Impact."
                if impacts
                else "No major reports in this window."
            )
        ]
    if not smaller_blocks:
        smaller_blocks = [
            (
                "No additional smaller moves outside Your Team Impact."
                if impacts
                else "No smaller fantasy-relevant moves in this window."
            )
        ]

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
            *team_section,
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
        team_impacts=impacts,
        omitted_team_impacts=omitted_impacts,
        omitted_big_news=omitted_big,
        omitted_smaller_moves=omitted_smaller,
    )
