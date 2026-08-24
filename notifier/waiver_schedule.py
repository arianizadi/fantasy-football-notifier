"""Deadline discovery and live evidence assembly for scheduled waiver reports.

The report engine in :mod:`notifier.waiver_report` is pure.  This adapter owns
the provider calls needed to discover each league's real processing timestamp,
refresh ownership just before a recommendation, and translate ESPN/Sleeper
facts into that engine.  It never makes a FantasyPros request; it reads only
the already-budgeted cache.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import Counter
from datetime import datetime, time as clock_time, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import requests

from .config import Config
from .dedupe import semantic_event_fact_signature, semantic_event_status
from .event_store import EventStore
from .logging_utils import NotifierError, structured_log
from .matcher import compact_key
from .models import NewsItem, RosterPlayer, RosterSnapshot
from .roster import ESPN_DEFAULT_POSITIONS, ESPN_LEAGUE_URL, ESPN_PRO_TEAMS
from .sources import sleeper
from .sources import sleeper_league
from .sources.fantasypros import FantasyProsCache
from .sources.twitter import attributed_absence_subject
from .telegram_control import ScheduledReport
from .waiver_report import (
    CandidateEvidence,
    NewsFact,
    RosterAsset,
    WaiverReportContext,
    build_waiver_report,
    format_waiver_report_html,
)

ESPN_TIMEOUT = 25
CHECK_INTERVAL_SECONDS = 15 * 60
RETRY_INTERVAL_SECONDS = 5 * 60
SLEEPER_WAIVER_TIMEZONE = "America/Los_Angeles"
SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})
OPPORTUNITY_EVENTS = frozenset({"injury", "inactive", "suspension", "release"})
RESOLUTION_EVENTS = frozenset({"return"})
POSITION_CAPS = {"QB": 40, "RB": 100, "WR": 120, "TE": 50}


def _availability_key(value: Any) -> str:
    compact = re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())
    aliases = {
        "": "",
        "active": "",
        "healthy": "",
        "none": "",
        "normal": "",
        "injuryreserve": "ir",
        "injuredreserve": "ir",
        "reserveinjured": "ir",
        "ir": "ir",
        "physicallyunabletoperform": "pup",
        "pup": "pup",
        "suspension": "suspended",
        "suspended": "suspended",
    }
    return aliases.get(compact, compact)


def _record_availability(record: Mapping[str, Any] | None) -> str:
    """Return confirmed unavailable, concern, healthy, or unknown."""
    if not record:
        return "unknown"
    raw_status = str(record.get("status") or "").strip().casefold()
    status = _availability_key(raw_status)
    injury = _availability_key(record.get("injury_status"))
    if status in {"inactive", "suspended"} or injury in {
        "out",
        "ir",
        "pup",
        "inactive",
        "suspended",
    }:
        return "unavailable"
    if injury in {"questionable", "doubtful"}:
        return "concern"
    if raw_status == "active" and not injury:
        return "healthy"
    return "unknown"


def _int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _timestamp_ms(value: Any) -> datetime | None:
    milliseconds = _float(value, -1)
    if milliseconds <= 0:
        return None
    try:
        return datetime.fromtimestamp(milliseconds / 1000, timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _dominant_espn_deadline(
    entries: Iterable[Mapping[str, Any]],
    *,
    now: datetime,
) -> datetime | None:
    """Choose the broad-pool processing cohort, not one player's drop timer."""
    current = _utc(now)
    counts: Counter[int] = Counter()
    total = 0
    for entry in entries:
        deadline = _timestamp_ms(entry.get("waiverProcessDate"))
        if deadline is None or deadline <= current:
            continue
        total += 1
        counts[int(deadline.timestamp())] += 1
    if not counts:
        return None
    stamp, count = max(counts.items(), key=lambda item: (item[1], -item[0]))
    # A lone recently dropped player is not the league-wide weekly deadline.
    if count < max(5, round(total * 0.20)):
        return None
    return datetime.fromtimestamp(stamp, timezone.utc)


def _next_sleeper_waiver(
    now: datetime,
    settings: Mapping[str, Any],
    *,
    timezone_name: str,
) -> datetime | None:
    """Translate Sleeper's ``<weekday> After`` setting to 12:05 AM PT.

    Sleeper numbers Sunday as 0.  A Tuesday-After setting is therefore value
    2 and processes at 12:05 AM Wednesday Pacific (Python weekday 2).
    Custom daily waivers are intentionally withheld until their per-day map
    can be interpreted without guessing.
    """
    if _int(settings.get("daily_waivers"), 0) != 0:
        return None
    waiver_day = _int(settings.get("waiver_day_of_week"), -1)
    if not 0 <= waiver_day <= 6:
        return None
    zone = ZoneInfo(timezone_name)
    local = _utc(now).astimezone(zone)
    target_weekday = waiver_day % 7
    days_ahead = (target_weekday - local.weekday()) % 7
    target_date = local.date() + timedelta(days=days_ahead)
    candidate = datetime.combine(target_date, clock_time(0, 5), tzinfo=zone)
    if candidate <= local:
        candidate += timedelta(days=7)
    return candidate.astimezone(timezone.utc)


def _sleeper_quality(record: Mapping[str, Any] | None) -> float:
    if not record:
        return 0.0
    rank = _int(record.get("search_rank"), -1)
    if rank <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 - ((rank - 1) / 4.0)))


def _espn_rank(player: Mapping[str, Any]) -> int | None:
    raw = player.get("rankings")
    values: Iterable[Any]
    if isinstance(raw, Mapping):
        values = raw.values()
    elif isinstance(raw, list):
        values = raw
    else:
        values = ()
    ranks = [
        _int(value.get("rank"), -1)
        for value in values
        if isinstance(value, Mapping) and _int(value.get("rank"), -1) > 0
    ]
    return min(ranks) if ranks else None


def _espn_quality(player: Mapping[str, Any]) -> float:
    ownership = player.get("ownership") or {}
    if not isinstance(ownership, Mapping):
        ownership = {}
    owned = max(0.0, min(100.0, _float(ownership.get("percentOwned"))))
    rank = _espn_rank(player)
    rank_quality = (
        max(0.0, min(100.0, 100.0 - ((rank - 1) / 3.0)))
        if rank is not None
        else 0.0
    )
    return max(owned, rank_quality)


def _position_percentile(value: Any, position: str) -> float | None:
    if value is None:
        return None
    match = re.search(r"(\d+)", str(value))
    rank = _int(match.group(1), -1) if match else _int(value, -1)
    cap = POSITION_CAPS.get(position.upper(), 100)
    if rank <= 0:
        return None
    if cap <= 1:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (1 - ((rank - 1) / (cap - 1)))))


def _source_identity(row: Mapping[str, Any]) -> str:
    source = str(row.get("source") or "source").strip()
    url = str(row.get("url") or "").strip()
    if source.casefold() in {"twitter", "x"} and url:
        path = [part for part in urlsplit(url).path.split("/") if part]
        if path and path[0].casefold() not in {"i", "intent"}:
            return f"X/{path[0]}"
    return source


def _published(row: Mapping[str, Any]) -> datetime | None:
    raw = row.get("published_at") or row.get("received_at")
    if not raw:
        return None
    try:
        value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return _utc(value)


def _news_item(row: Mapping[str, Any]) -> NewsItem:
    return NewsItem(
        source=str(row.get("source") or ""),
        guid=str(row.get("guid") or ""),
        player_name=str(row.get("player_name") or ""),
        headline=str(row.get("headline") or ""),
        body=str(row.get("body") or ""),
        url=str(row.get("url") or ""),
        published_at=_published(row),
    )


class WaiverReportCoordinator:
    """Discover due league reports and assemble current, league-specific facts."""

    def __init__(
        self,
        config: Config,
        *,
        snapshot_provider: Callable[[], RosterSnapshot],
        player_index_provider: Callable[[], Mapping[str, Any]],
        refresh_provider: Callable[[set[str]], bool],
        event_store: EventStore,
        fantasypros: FantasyProsCache,
        completed_provider: Callable[[str], bool],
        session: requests.Session | None = None,
        clock: Callable[[], float] = time.time,
        now_provider: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._config = config
        self._snapshot_provider = snapshot_provider
        self._player_index_provider = player_index_provider
        self._refresh_provider = refresh_provider
        self._events = event_store
        self._fantasypros = fantasypros
        self._completed_provider = completed_provider
        self._session = session or requests.Session()
        self._owns_session = session is None
        self._clock = clock
        self._now_provider = now_provider
        self._lock = threading.Lock()
        self._next_check_at = 0.0
        self._cached_due: tuple[ScheduledReport, ...] = ()

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def due_reports(self, now: datetime) -> tuple[ScheduledReport, ...]:
        """Return currently due reports; provider checks are throttled."""
        if not bool(getattr(self._config, "waiver_report_enabled", True)):
            return ()
        current = _utc(now)
        with self._lock:
            stamp = self._clock()
            if stamp < self._next_check_at:
                # Cached content is useful through a transient Telegram
                # outage, but a waiver plan must never be sent after claims
                # have already processed.
                self._cached_due = tuple(
                    report
                    for report in self._cached_due
                    if self._report_deadline(report.key) > current
                )
                return self._cached_due
            self._next_check_at = stamp + CHECK_INTERVAL_SECONDS
            try:
                reports = self._discover_due(current)
            except (requests.RequestException, ValueError, TypeError) as error:
                # Retry sooner after a provider outage, without hammering it.
                self._next_check_at = stamp + 5 * 60
                structured_log(
                    logging.WARNING,
                    "waiver.report_discovery_failed",
                    errorType=type(error).__name__,
                    error=str(error)[:160],
                )
                self._cached_due = tuple(
                    report
                    for report in self._cached_due
                    if self._report_deadline(report.key) > current
                )
                return self._cached_due
            self._cached_due = reports
            return reports

    def _discover_due(self, now: datetime) -> tuple[ScheduledReport, ...]:
        snapshot = self._snapshot_provider()
        pools: list[dict[str, Any]] = []
        if bool(getattr(self._config, "espn_enabled", False)):
            league_key = f"espn:{getattr(self._config, 'espn_league_id', 0)}"
            if snapshot.is_drafted(league_key):
                try:
                    pool = self._fetch_espn_pool(now)
                except (requests.RequestException, ValueError, TypeError) as error:
                    structured_log(
                        logging.WARNING,
                        "waiver.espn_schedule_failed",
                        errorType=type(error).__name__,
                    )
                else:
                    if pool is not None:
                        pools.append(pool)
        drafted_sleeper = any(
            league.provider == "sleeper"
            for league in snapshot.drafted_leagues()
        )
        if (
            drafted_sleeper
            and str(getattr(self._config, "sleeper_username", "") or "").strip()
        ):
            try:
                pools.extend(self._fetch_sleeper_schedules(snapshot, now))
            except (
                requests.RequestException,
                NotifierError,
                ValueError,
                TypeError,
            ) as error:
                # One provider can be pre-draft or offline without suppressing
                # a due report from the other active league.
                structured_log(
                    logging.WARNING,
                    "waiver.sleeper_schedule_failed",
                    errorType=type(error).__name__,
                )

        lead = timedelta(
            hours=int(getattr(self._config, "waiver_report_lead_hours", 8))
        )
        upcoming = [
            pool["deadline"] - lead
            for pool in pools
            if pool["deadline"] - lead > now
        ]
        if upcoming:
            seconds_to_boundary = max(
                0.0,
                (min(upcoming) - now).total_seconds(),
            )
            # Polling remains bounded, but the last pre-window discovery wakes
            # the scheduler on the exact T-minus boundary rather than as much
            # as fifteen minutes late.
            self._next_check_at = min(
                self._next_check_at,
                self._clock() + seconds_to_boundary,
            )
        due = [
            pool
            for pool in pools
            if pool["deadline"] - lead <= now < pool["deadline"]
        ]
        if not due:
            return ()

        due = [
            pool
            for pool in due
            if not self._completed_provider(self._report_key(pool))
        ]
        if not due:
            return ()

        # Refresh one due league at a time. Once both providers are active, an
        # outage at either one must not suppress the other league's report.
        refreshed_due: list[dict[str, Any]] = []
        refresh_failed = False
        for pool in due:
            league_key = str(pool["league_key"])
            if self._refresh_provider({league_key}):
                if _utc(self._now_provider()) < _utc(pool["deadline"]):
                    refreshed_due.append(pool)
                else:
                    structured_log(
                        logging.INFO,
                        "waiver.report_expired_during_refresh",
                        leagueKey=league_key,
                    )
            else:
                refresh_failed = True
                structured_log(
                    logging.WARNING,
                    "waiver.ownership_refresh_failed",
                    leagueKey=league_key,
                )
        if refresh_failed:
            self._next_check_at = min(
                self._next_check_at,
                self._clock() + RETRY_INTERVAL_SECONDS,
            )
        if not refreshed_due:
            return ()
        due = refreshed_due
        snapshot = self._snapshot_provider()
        player_index = self._player_index_provider()
        history_start = min(
            _utc(pool["deadline"]) - timedelta(days=7)
            for pool in due
        )
        rows = self._events.recent(
            since=history_start,
            until=now + timedelta(seconds=1),
            limit=2000,
        )
        try:
            adds_6h = sleeper.trending_adds(
                self._session, dict(player_index), limit=200, lookback_hours=6
            )
            adds_24h = sleeper.trending_adds(
                self._session, dict(player_index), limit=300, lookback_hours=24
            )
        except (requests.RequestException, ValueError, TypeError):
            adds_6h = {}
            adds_24h = {}

        reports: list[ScheduledReport] = []
        for pool in due:
            league_key = str(pool["league_key"])
            if not snapshot.is_drafted(league_key):
                continue
            deadline = _utc(pool["deadline"])
            if _utc(self._now_provider()) >= deadline:
                continue
            # Use the previous processing cohort as the evidence boundary.
            # This includes all changes since last waivers while excluding the
            # prior Monday's already-actioned report window.
            pool_history_start = deadline - timedelta(days=7)
            pool_rows = [
                row
                for row in rows
                if (published := _published(row)) is not None
                and published >= pool_history_start
            ]
            candidates = self._candidate_evidence(
                pool,
                snapshot,
                player_index,
                pool_rows,
                adds_6h,
                adds_24h,
                now,
            )
            context = self._context(pool, snapshot, player_index, now)
            report = build_waiver_report(context, candidates)
            parts = format_waiver_report_html(report)
            if _utc(self._now_provider()) >= deadline:
                structured_log(
                    logging.INFO,
                    "waiver.report_expired_during_build",
                    leagueKey=league_key,
                )
                continue
            reports.append(
                ScheduledReport(
                    key=self._report_key(pool),
                    kind="waiver_report",
                    parts=parts,
                    notify_first=True,
                    expires_at=deadline,
                )
            )
        return tuple(reports)

    @staticmethod
    def _report_key(pool: Mapping[str, Any]) -> str:
        deadline = _utc(pool["deadline"])
        return f"waiver:{pool['league_key']}:{int(deadline.timestamp())}"

    @staticmethod
    def _report_deadline(key: str) -> datetime:
        try:
            stamp = int(str(key).rsplit(":", 1)[1])
            return datetime.fromtimestamp(stamp, timezone.utc)
        except (IndexError, TypeError, ValueError, OSError):
            return datetime.min.replace(tzinfo=timezone.utc)

    def _fetch_espn_pool(self, now: datetime) -> dict[str, Any] | None:
        league_id = int(getattr(self._config, "espn_league_id", 0))
        season = int(getattr(self._config, "espn_year", now.year))
        url = ESPN_LEAGUE_URL.format(year=season, league_id=league_id)
        fantasy_filter = {
            "players": {
                "filterStatus": {"value": ["FREEAGENT", "WAIVERS"]},
                "filterSlotIds": {"value": [0, 2, 4, 6]},
                "limit": 1200,
                "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
            }
        }
        response = self._session.get(
            url,
            params=[
                ("view", "kona_player_info"),
                ("view", "mSettings"),
                ("view", "mTeam"),
            ],
            cookies={
                "SWID": str(getattr(self._config, "espn_swid", "")),
                "espn_s2": str(getattr(self._config, "espn_s2", "")),
            },
            headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; fantasy-football-notifier/1.0)",
                "x-fantasy-filter": json.dumps(fantasy_filter, separators=(",", ":")),
            },
            timeout=ESPN_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        if not isinstance(payload, Mapping):
            raise ValueError("ESPN player pool had an unexpected shape")
        entries = [
            entry
            for entry in payload.get("players") or []
            if isinstance(entry, Mapping)
        ]
        deadline = _dominant_espn_deadline(entries, now=now)
        if deadline is None:
            return None
        settings = payload.get("settings") or {}
        acquisition = (
            settings.get("acquisitionSettings") or {}
            if isinstance(settings, Mapping)
            else {}
        )
        method = "Traditional rolling priority"
        if isinstance(acquisition, Mapping) and bool(
            acquisition.get("isUsingAcquisitionBudget")
        ):
            method = "FAAB"
        priority = None
        team_id = getattr(self._config, "espn_team_id", None)
        for team in payload.get("teams") or []:
            if not isinstance(team, Mapping):
                continue
            if team_id is not None and _int(team.get("id")) == int(team_id):
                priority = _int(team.get("waiverRank"), -1)
                break
            owners = team.get("owners") or []
            owner_values = [
                str(owner.get("id") if isinstance(owner, Mapping) else owner)
                .strip("{}")
                .casefold()
                for owner in owners
            ]
            swid = str(getattr(self._config, "espn_swid", "")).strip("{}").casefold()
            if swid and swid in owner_values:
                priority = _int(team.get("waiverRank"), -1)
                break
        if priority is not None and priority < 0:
            priority = None
        league_key = f"espn:{league_id}"
        return {
            "provider": "espn",
            "league_key": league_key,
            "deadline": deadline,
            "entries": entries,
            "method": method,
            "priority": priority,
        }

    def _fetch_sleeper_schedules(
        self,
        snapshot: RosterSnapshot,
        now: datetime,
    ) -> list[dict[str, Any]]:
        username = str(getattr(self._config, "sleeper_username", ""))
        user_id = sleeper_league.resolve_user_id(self._session, username)
        season = int(getattr(self._config, "espn_year", now.year))
        leagues = sleeper_league.list_leagues(self._session, user_id, season)
        allowed = set(getattr(self._config, "sleeper_league_ids", ()) or ())
        pools: list[dict[str, Any]] = []
        for league in leagues:
            if not isinstance(league, Mapping):
                continue
            league_id = str(league.get("league_id") or "")
            if allowed and league_id not in allowed:
                continue
            league_key = f"sleeper:{league_id}"
            if not snapshot.is_drafted(league_key):
                continue
            status = str(league.get("status") or "").strip().casefold()
            if status != "in_season":
                # A partial roster during a live draft is not an available
                # free-agent pool. ``complete`` is also withheld because it
                # means the fantasy season has ended, not that a draft ended.
                continue
            settings = league.get("settings") or {}
            if not isinstance(settings, Mapping):
                continue
            deadline = _next_sleeper_waiver(
                now,
                settings,
                timezone_name=SLEEPER_WAIVER_TIMEZONE,
            )
            if deadline is None:
                structured_log(
                    logging.WARNING,
                    "waiver.sleeper_schedule_unsupported",
                    leagueId=league_id,
                    customDaily=bool(_int(settings.get("daily_waivers"), 0)),
                )
                continue
            pools.append(
                {
                    "provider": "sleeper",
                    "league_key": league_key,
                    "deadline": deadline,
                    "method": (
                        "FAAB"
                        if _int(settings.get("waiver_type"), 0) == 2
                        else "Rolling priority"
                    ),
                    "priority": None,
                }
            )
        return pools

    @staticmethod
    def _records_by_name(
        player_index: Mapping[str, Any],
    ) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
        records: dict[str, Mapping[str, Any]] = {}
        ids: dict[str, str] = {}
        for player_id, raw in player_index.items():
            if not isinstance(raw, Mapping):
                continue
            key = compact_key(str(raw.get("full_name") or ""))
            if key:
                records[key] = raw
                ids[key] = str(player_id)
        return records, ids

    @staticmethod
    def _depth(record: Mapping[str, Any] | None) -> int | None:
        if not record:
            return None
        value = _int(record.get("depth_chart_order"), -1)
        return value if value > 0 else None

    def _facts_for_candidate(
        self,
        name: str,
        record: Mapping[str, Any],
        rows: Iterable[Mapping[str, Any]],
        records: Mapping[str, Mapping[str, Any]],
        teammates: Iterable[tuple[str, Mapping[str, Any]]],
        *,
        now: datetime | None = None,
        player_index_refreshed_at: datetime | None = None,
    ) -> tuple[tuple[NewsFact, ...], str, str, bool, str]:
        key = compact_key(name)
        team = str(record.get("team") or "").upper()
        position = str(record.get("position") or "").upper()
        depth = self._depth(record)
        facts: list[NewsFact] = []
        role = "confirmed_starter" if depth == 1 else "depth_two" if depth == 2 else "ordinary_backup"
        role_subject = ""
        inferred_without_name = False
        direct_availability_events: list[tuple[datetime, str]] = []
        teammate_entries = tuple(teammates)
        teammate_names = [
            str(possible.get("full_name") or "").strip()
            for _, possible in teammate_entries
            if str(possible.get("full_name") or "").strip()
        ]
        current = _utc(now or datetime.now(timezone.utc))

        for row in rows:
            # Ambiguously attributed multi-player reports are useful in the
            # recap, but cannot establish either a candidate's availability or
            # a mechanical next-man-up relationship.
            if _int(row.get("subject_confident"), 1) == 0:
                continue
            event = str(row.get("event_type") or "other").strip().casefold()
            if event not in OPPORTUNITY_EVENTS | RESOLUTION_EVENTS | {
                "usage",
                "depth_chart",
                "signing",
                "trade",
            }:
                continue
            published = _published(row)
            if published is None:
                continue
            text = " ".join(
                str(row.get(field) or "")
                for field in ("player_name", "headline", "body")
            )
            compact_text = compact_key(text)
            direct = bool(key and key in compact_text)
            reported_subject_key = compact_key(str(row.get("player_name") or ""))
            direct_subject = bool(key and reported_subject_key == key)
            row_subject_key = reported_subject_key
            subject_record = records.get(row_subject_key)
            absence_key = ""
            if event in OPPORTUNITY_EVENTS:
                absence_key = compact_key(
                    attributed_absence_subject(text, teammate_names)
                )

            # Beneficiary reports often name the backup as the row subject but
            # mention the injured starter in the body. Prefer the nearest
            # higher depth teammate only when deterministic clause-level
            # attribution says that teammate is the unavailable player. A
            # mere mention of the starter cannot erase a direct injury to the
            # backup itself.
            higher_mentions: list[tuple[int, str, Mapping[str, Any]]] = []
            if depth is not None:
                for possible_key, possible in teammate_entries:
                    possible_depth = self._depth(possible)
                    if (
                        possible_key != key
                        and possible_key == absence_key
                        and str(possible.get("team") or "").upper() == team
                        and str(possible.get("position") or "").upper() == position
                        and possible_depth is not None
                        and possible_depth < depth
                    ):
                        higher_mentions.append((possible_depth, possible_key, possible))
            if higher_mentions:
                _, row_subject_key, subject_record = max(higher_mentions)

            linked = False
            subject_name = str(row.get("player_name") or "")
            subject_depth = self._depth(subject_record)
            if subject_record is not None:
                subject_name = str(subject_record.get("full_name") or subject_name)
                linked = (
                    depth is not None
                    and subject_depth is not None
                    and str(subject_record.get("team") or "").upper() == team
                    and str(subject_record.get("position") or "").upper() == position
                    and 0 < depth - subject_depth <= 2
                )
            useful_direct = direct_subject and event in {
                "usage",
                "depth_chart",
                "signing",
                "trade",
            }
            direct_availability = direct_subject and (
                event in RESOLUTION_EVENTS
                or (
                    event in OPPORTUNITY_EVENTS
                    and (not absence_key or absence_key == key)
                )
            )
            if not linked and not useful_direct and not direct_availability:
                continue

            if direct_availability:
                direct_availability_events.append((published, event))

            item = _news_item(row)
            status = semantic_event_status(item, event)
            if useful_direct and event == "depth_chart":
                if status == "role_starter":
                    role = "confirmed_starter"
                elif status == "role_not_starter":
                    role = "ordinary_backup"
            signature = f"{event}:{semantic_event_fact_signature(item, event)}"
            creates = event in OPPORTUNITY_EVENTS and linked
            provisional_opportunity = False
            if creates and event in {"injury", "inactive"}:
                availability = _record_availability(subject_record)
                age_hours = max(
                    0.0,
                    (current - _utc(published)).total_seconds() / 3600,
                )
                index_is_current_for_fact = (
                    isinstance(player_index_refreshed_at, datetime)
                    and _utc(player_index_refreshed_at) >= _utc(published)
                )
                news_confirms_absence = status in {
                    "season_out",
                    "injured_reserve",
                    "inactive",
                }
                if index_is_current_for_fact and availability == "unavailable":
                    # A provider snapshot captured after the report confirms
                    # that the vacancy still exists.
                    pass
                elif (
                    index_is_current_for_fact
                    and availability == "healthy"
                    and age_hours > 24
                ):
                    # Only a provider snapshot at least as new as the report
                    # may resolve an old injury without a separate return
                    # story. A cached snapshot from before the report must
                    # never overrule faster breaking news.
                    creates = False
                elif news_confirms_absence and (
                    age_hours <= 24 or not index_is_current_for_fact
                ):
                    # Definitive report wording remains authoritative while
                    # provider data is older, and for the first day while a
                    # provider may still be catching up.
                    pass
                elif availability == "concern" or age_hours <= 24:
                    provisional_opportunity = True
                else:
                    # Without current provider confirmation, retain an old
                    # report only as a cautious watch rather than claiming a
                    # currently open starting role.
                    provisional_opportunity = True
            resolves = event in RESOLUTION_EVENTS and linked
            facts.append(
                NewsFact(
                    signature=signature,
                    event_type=event,
                    severity=max(1, min(5, _int(row.get("severity"), 2))),
                    published_at=published,
                    source=_source_identity(row),
                    subject_name=subject_name,
                    status=status,
                    directly_names_candidate=direct,
                    creates_opportunity=creates,
                    resolves_opportunity=resolves,
                    supports_candidate=creates or useful_direct,
                    description=str(row.get("headline") or ""),
                )
            )
            if creates:
                role_subject = subject_name
                if provisional_opportunity:
                    role = "uncertain_replacement"
                elif depth is not None and subject_depth is not None and depth - subject_depth == 1:
                    role = "immediate_replacement"
                else:
                    role = "uncertain_replacement"
                inferred_without_name = inferred_without_name or not direct
        latest_direct_event = (
            max(direct_availability_events, default=(datetime.min.replace(tzinfo=timezone.utc), ""))[1]
        )
        direct_concern = (
            latest_direct_event if latest_direct_event in OPPORTUNITY_EVENTS else ""
        )
        return tuple(facts), role, role_subject, inferred_without_name, direct_concern

    def _fantasypros_fields(
        self,
        name: str,
        *,
        team: str,
        position: str,
        scoring: str,
    ) -> tuple[float | None, float | None, datetime | None]:
        try:
            signal = self._fantasypros.signal(
                name,
                scoring=scoring,
                team=team,
                position=position,
            )
        except ValueError:
            return None, None, None
        if signal is None:
            return None, None, None
        return (
            _position_percentile(signal.waiver_pos_rank or signal.waiver_rank, position),
            _position_percentile(signal.ros_pos_rank or signal.ros_rank, position),
            signal.updated_at,
        )

    def _candidate_evidence(
        self,
        pool: Mapping[str, Any],
        snapshot: RosterSnapshot,
        player_index: Mapping[str, Any],
        rows: Iterable[Mapping[str, Any]],
        adds_6h: Mapping[str, int],
        adds_24h: Mapping[str, int],
        now: datetime,
    ) -> tuple[CandidateEvidence, ...]:
        league_key = str(pool["league_key"])
        scoring = snapshot.scoring_formats.get(league_key, "PPR")
        records, _ = self._records_by_name(player_index)
        depth_groups: dict[
            tuple[str, str], list[tuple[str, Mapping[str, Any]]]
        ] = {}
        for record_key, record in records.items():
            depth_groups.setdefault(
                (
                    str(record.get("team") or "").upper(),
                    str(record.get("position") or "").upper(),
                ),
                [],
            ).append((record_key, record))
        rostered = {
            compact_key(player.name)
            for player in snapshot.players
            if player.league_key == league_key
        }
        raw_candidates: list[tuple[str, str, str, float, Mapping[str, Any]]] = []
        if pool["provider"] == "espn":
            for entry in pool.get("entries") or []:
                player = entry.get("player") or {}
                if not isinstance(player, Mapping):
                    continue
                name = str(player.get("fullName") or "").strip()
                position = ESPN_DEFAULT_POSITIONS.get(
                    _int(player.get("defaultPositionId")), ""
                )
                team = ESPN_PRO_TEAMS.get(_int(player.get("proTeamId")), "")
                if name and position in SKILL_POSITIONS and team:
                    raw_candidates.append(
                        (name, position, team, _espn_quality(player), player)
                    )
        else:
            for record in player_index.values():
                if not isinstance(record, Mapping):
                    continue
                name = str(record.get("full_name") or "").strip()
                key = compact_key(name)
                position = str(record.get("position") or "").upper()
                team = str(record.get("team") or "").upper()
                status = str(record.get("status") or "").casefold()
                if (
                    not name
                    or key in rostered
                    or position not in SKILL_POSITIONS
                    or not team
                    or (status and status not in {"active", "inactive"})
                ):
                    continue
                quality = _sleeper_quality(record)
                if quality >= 15 or self._depth(record) in {1, 2, 3} or key in adds_24h:
                    raw_candidates.append((name, position, team, quality, record))

        refreshed_at = getattr(player_index, "refreshed_at", None)
        data_age = (
            (_utc(now) - _utc(refreshed_at)).total_seconds() / 3600
            if isinstance(refreshed_at, datetime)
            else None
        )
        result: list[CandidateEvidence] = []
        for name, position, team, provider_quality, raw in raw_candidates:
            key = compact_key(name)
            record = records.get(key, {})
            facts, role, role_subject, inferred, direct_concern = self._facts_for_candidate(
                name,
                record,
                rows,
                records,
                depth_groups.get((team.upper(), position.upper()), ()),
                now=now,
                player_index_refreshed_at=(
                    refreshed_at
                    if (
                        isinstance(refreshed_at, datetime)
                        and not bool(getattr(player_index, "stale", False))
                    )
                    else None
                ),
            )
            quality = max(provider_quality, _sleeper_quality(record))
            if quality < 15 and not facts and key not in adds_24h:
                continue
            waiver_percentile, ros_percentile, fantasy_updated = (
                self._fantasypros_fields(
                    name,
                    team=team,
                    position=position,
                    scoring=scoring,
                )
            )
            provider_status = str(record.get("status") or "").strip().casefold()
            espn_injury_raw = (
                str(raw.get("injuryStatus") or "").strip()
                if pool["provider"] == "espn"
                else ""
            )
            sleeper_injury_raw = str(record.get("injury_status") or "").strip()
            provider_injury_raw = (
                espn_injury_raw
                if pool["provider"] == "espn" and "injuryStatus" in raw
                else sleeper_injury_raw
            )
            provider_injury = _availability_key(provider_injury_raw)
            hard_unavailable = (
                direct_concern in {"release", "suspension", "inactive"}
                or provider_status in {"inactive", "suspended"}
                or provider_injury in {"inactive", "suspended"}
            )
            recommendation_blocked = (
                direct_concern == "injury"
                or provider_injury
                in {"out", "ir", "pup", "suspended", "suspension", "doubtful"}
            )
            injury_display = ""
            if provider_injury:
                injury_source = (
                    "ESPN"
                    if pool["provider"] == "espn" and "injuryStatus" in raw
                    else "Sleeper"
                )
                injury_display = (
                    f"{injury_source} "
                    + provider_injury_raw.replace("_", " ").title()
                ).strip()
            elif direct_concern:
                injury_display = (
                    f"Recent {direct_concern.replace('_', ' ')} report"
                )
            result.append(
                CandidateEvidence(
                    name=name,
                    position=position,
                    pro_team=team,
                    # A player who was released, suspended, or declared
                    # inactive cannot be a waiver recommendation merely
                    # because an old rank still looks attractive. An injury is
                    # kept visible as a WATCH item below, but never a claim.
                    available=(
                        key not in rostered
                        and not hard_unavailable
                    ),
                    role=role,
                    role_subject=role_subject,
                    depth_order=self._depth(record),
                    fantasypros_waiver_percentile=waiver_percentile,
                    fantasypros_ros_percentile=ros_percentile,
                    fantasypros_updated_at=fantasy_updated,
                    platform_quality_percentile=quality,
                    platform_quality_source=(
                        "ESPN rank/ownership"
                        if pool["provider"] == "espn"
                        else "Sleeper search rank"
                    ),
                    sleeper_adds_6h=max(0, _int(adds_6h.get(key), 0)),
                    sleeper_adds_24h=max(0, _int(adds_24h.get(key), 0)),
                    sleeper_data_age_hours=data_age,
                    injury_status=injury_display,
                    recommendation_blocked=recommendation_blocked,
                    blocked_reason=(
                        "Unresolved injury or availability status; monitor, but do not claim yet."
                        if recommendation_blocked
                        else ""
                    ),
                    committee_unresolved=role == "uncertain_replacement",
                    speculative=inferred,
                    facts=facts,
                )
            )
        # Limit pure-engine work while retaining more than any rendered report.
        result.sort(
            key=lambda candidate: (
                -max(
                    candidate.platform_quality_percentile or 0,
                    candidate.fantasypros_waiver_percentile or 0,
                ),
                -candidate.sleeper_adds_6h,
                compact_key(candidate.name),
            )
        )
        return tuple(result[:350])

    def _roster_assets(
        self,
        league_key: str,
        snapshot: RosterSnapshot,
        player_index: Mapping[str, Any],
        scoring: str,
        now: datetime,
    ) -> tuple[RosterAsset, ...]:
        records, _ = self._records_by_name(player_index)
        mine = snapshot.mine(league_key)
        owned_keys = {compact_key(player.name) for player in mine}
        assets: list[RosterAsset] = []
        for player in mine:
            record = records.get(compact_key(player.name), {})
            quality = _sleeper_quality(record)
            _, ros_percentile, fantasy_updated = self._fantasypros_fields(
                player.name,
                team=player.pro_team,
                position=player.position,
                scoring=scoring,
            )
            fantasy_fresh = False
            if fantasy_updated is not None:
                age_hours = (
                    _utc(now) - _utc(fantasy_updated)
                ).total_seconds() / 3600
                fantasy_fresh = (
                    -1
                    <= age_hours
                    <= int(getattr(self._config, "fantasypros_max_age_hours", 12))
                )
            has_sleeper_value = _int(record.get("search_rank"), -1) > 0
            roster_value = (
                ros_percentile
                if fantasy_fresh and ros_percentile is not None
                else quality if has_sleeper_value else None
            )
            depth = self._depth(record)
            protected = False
            if player.can_be_started_from_bench and depth == 2:
                for teammate in mine:
                    teammate_record = records.get(compact_key(teammate.name), {})
                    if (
                        compact_key(teammate.name) in owned_keys
                        and teammate.position.upper() == player.position.upper()
                        and teammate.pro_team.upper() == player.pro_team.upper()
                        and self._depth(teammate_record) == 1
                    ):
                        protected = True
                        break
            assets.append(
                RosterAsset(
                    name=player.name,
                    position=player.position,
                    lineup_slot=player.lineup_slot,
                    waiver_value=roster_value,
                    pro_team=player.pro_team,
                    protected=protected,
                    elite=quality >= 85 or compact_key(player.name) == "lamarjackson",
                )
            )
        return tuple(assets)

    def _context(
        self,
        pool: Mapping[str, Any],
        snapshot: RosterSnapshot,
        player_index: Mapping[str, Any],
        now: datetime,
    ) -> WaiverReportContext:
        league_key = str(pool["league_key"])
        league = snapshot.league(league_key)
        capacity = snapshot.capacities.get(league_key)
        zone = ZoneInfo(
            str(
                getattr(
                    self._config,
                    "daily_digest_timezone",
                    "America/Los_Angeles",
                )
            )
        )
        scoring = snapshot.scoring_formats.get(league_key, "PPR")
        assets = self._roster_assets(
            league_key,
            snapshot,
            player_index,
            scoring,
            now,
        )
        records, _ = self._records_by_name(player_index)
        qb_unavailable = any(
            asset.position.upper() == "QB"
            and asset.lineup_slot.upper() not in {"BE", "BN", "BENCH"}
            and str(
                records.get(compact_key(asset.name), {}).get("injury_status") or ""
            ).casefold()
            in {"out", "ir", "pup", "suspended"}
            for asset in assets
        )
        return WaiverReportContext(
            league_name=(league.name if league is not None else league_key),
            scoring_format=scoring,
            generated_at=_utc(now),
            expected_waiver_at=_utc(pool["deadline"]).astimezone(zone),
            bench_used=capacity.bench_used if capacity is not None else None,
            bench_limit=capacity.bench_limit if capacity is not None else None,
            ir_used=capacity.ir_used if capacity is not None else None,
            ir_limit=capacity.ir_limit if capacity is not None else None,
            roster=assets,
            starting_qb_unavailable=qb_unavailable,
            waiver_method=str(pool.get("method") or ""),
            waiver_priority=pool.get("priority"),
        )


__all__ = [
    "WaiverReportCoordinator",
    "_dominant_espn_deadline",
    "_next_sleeper_waiver",
]
